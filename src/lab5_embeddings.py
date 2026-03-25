"""
Lab 5 – OWL Ontology Reasoning & Knowledge Graph Embeddings
============================================================
Section A : Load family.owl → apply SWRL rule → infer OldPerson
Section B : Parse kb_expanse.nt → clean & split 80/10/10
Section C : Train TransE & DistMult (pure NumPy)
Section D : Evaluate, visualise (t-SNE, relation norms, KB-size sensitivity)

Usage:
    python src/lab5_embeddings.py [--section A|B|C|D|all]

    # Run all sections sequentially (default):
    python src/lab5_embeddings.py

    # Run a specific section:
    python src/lab5_embeddings.py --section C
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

random.seed(42)
np.random.seed(42)
os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A – OWL ontology + SWRL reasoning (family.owl)
# ══════════════════════════════════════════════════════════════════════════════

def run_section_a():
    print("\n" + "=" * 60)
    print("  SECTION A — OWL Reasoning (SWRL rule)")
    print("=" * 60)

    try:
        from owlready2 import Imp, get_ontology, sync_reasoner_pellet
    except ImportError:
        print("[SKIP] owlready2 not installed. Run: pip install owlready2")
        return

    ontology_path = os.path.abspath("family.owl")
    if not os.path.exists(ontology_path):
        print(f"[SKIP] family.owl not found at {ontology_path}")
        return

    onto = get_ontology(f"file://{ontology_path}").load()
    print(f"\n[OK] Ontology loaded : {onto.base_iri}")
    print(f"     Individuals : {[i.name for i in onto.individuals()]}")
    print(f"     Classes     : {[c.name for c in onto.classes()]}")

    # --- Display state BEFORE reasoning ---
    print("\n── Before reasoning ────────────────────────────────────")
    print(f"  {'Individual':<12} {'Age':>5}   Classes")
    print(f"  {'-'*12} {'-'*5}   {'-'*30}")
    for person in sorted(onto.individuals(), key=lambda p: p.name or ""):
        age_val = person.age if person.age is not None else "?"
        classes = [c.name for c in person.is_a if hasattr(c, "name")]
        print(f"  {person.name:<12} {str(age_val):>5}   {classes}")

    # --- Define SWRL rule ---
    with onto:
        rule = Imp()
        rule.set_as_rule(
            "Person(?p), age(?p, ?a), greaterThan(?a, 60) -> OldPerson(?p)",
            namespaces=[onto],
        )
    print("\n── SWRL rule defined ────────────────────────────────────")
    print("  Person(?p) ∧ age(?p, ?a) ∧ greaterThan(?a, 60) → OldPerson(?p)")

    # --- Run reasoner (Pellet or Python fallback) ---
    print("\n[INFO] Running reasoner…")
    try:
        with onto:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=0)
        reasoner = "Pellet"
        print("[OK] Pellet finished.")
    except Exception as e:
        print(f"[WARN] Pellet unavailable ({type(e).__name__}) — applying rule manually")
        reasoner = "Python fallback"
        with onto:
            for person in list(onto.Person.instances()):
                if person.age is not None and int(person.age) > 60:
                    if onto.OldPerson not in person.is_a:
                        person.is_a.append(onto.OldPerson)
                        print(f"  ✓ {person.name} (age={person.age}) → OldPerson")

    # --- Results ---
    old_persons = list(onto.OldPerson.instances())
    print(f"\n── After reasoning [{reasoner}] ───────────────────────")
    print(f"  {'Individual':<12} {'Age':>5}   {'OldPerson?':<14} Classes")
    for person in sorted(onto.individuals(), key=lambda p: p.name or ""):
        age_val = person.age if person.age is not None else "?"
        is_old  = "✓ OldPerson" if person in old_persons else ""
        classes = [c.name for c in person.is_a if hasattr(c, "name")]
        print(f"  {person.name:<12} {str(age_val):>5}   {is_old:<14} {classes}")
    print(f"\n  ➜ {len(old_persons)} inferred OldPerson(s):")
    for p in old_persons:
        print(f"     • {p.name} (age = {p.age})")

    # --- Save ---
    onto.save(file="family_inferred.owl", format="rdfxml")
    print("\n[OK] Saved → family_inferred.owl")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B – Parse kb_expanse.nt → clean & split
# ══════════════════════════════════════════════════════════════════════════════

NT_RE = re.compile(r'^(<[^>]+>)\s+(<[^>]+>)\s+(.+)\s+\.$')


def shorten(iri: str) -> str:
    iri = iri.strip("<>")
    m = re.search(r'/entity/(Q\w+)$', iri)
    if m: return m.group(1)
    m = re.search(r'/direct/(P\w+)$', iri)
    if m: return m.group(1)
    m = re.search(r'monprojet\.org/entity/(.+)$', iri)
    if m: return "mp_" + m.group(1)
    part = iri.rstrip("/").split("/")[-1]
    return part if part else iri


def save_txt(triples: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    print(f"[OK] → {path}  ({len(triples):,} lines)")


def run_section_b() -> bool:
    """Parse and split kb_expanse.nt. Returns True on success."""
    print("\n" + "=" * 60)
    print("  SECTION B — NT Parsing & Train/Valid/Test Split")
    print("=" * 60)

    if not os.path.exists("kb_expanse.nt"):
        print("[SKIP] kb_expanse.nt not found — run lab4 first.")
        return False

    print(f"\n[INFO] Parsing kb_expanse.nt …")
    raw, skipped_lit, skipped_bad = [], 0, 0
    with open("kb_expanse.nt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = NT_RE.match(line)
            if not m:
                skipped_bad += 1
                continue
            h, r, t = m.group(1), m.group(2), m.group(3).strip()
            if t.startswith('"') or t.startswith("'"):
                skipped_lit += 1
                continue
            raw.append((h, r, t))

    print(f"[OK] URI→URI triples   : {len(raw):,}")
    print(f"     Literals skipped  : {skipped_lit:,}")
    print(f"     Invalid lines     : {skipped_bad}")

    # Dedup + remove self-loops
    before = len(raw)
    raw = list(dict.fromkeys(raw))
    raw = [(h, r, t) for h, r, t in raw if h != t]
    print(f"[OK] Duplicates removed: {before - len(raw):,}")
    print(f"[OK] Clean triples     : {len(raw):,}")

    triples = [(shorten(h), shorten(r), shorten(t)) for h, r, t in raw]
    entities  = sorted(set(h for h,_,_ in triples) | set(t for _,_,t in triples))
    relations = sorted(set(r for _,r,_ in triples))

    print(f"\n[OK] Unique entities  : {len(entities):,}")
    print(f"[OK] Unique relations : {len(relations)}")

    rel_counts = Counter(r for _,r,_ in triples)
    wikidata_labels = {
        "P31": "instance of", "P166": "award received", "P17": "country",
        "P856": "official website", "P1346": "winner", "P571": "inception date",
        "P452": "industry", "P159": "headquarters", "P355": "subsidiary",
    }
    print(f"\n[INFO] Top 10 relations:")
    for rel, cnt in rel_counts.most_common(10):
        print(f"  {rel:<12} {cnt:>8,}  {wikidata_labels.get(rel,'')}")

    # Indexing
    e2id = {e: i for i, e in enumerate(entities)}
    r2id = {r: i for i, r in enumerate(relations)}

    # 80/10/10 split with no entity leakage
    random.shuffle(triples)
    n = len(triples)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    train = triples[:n_train]
    valid = triples[n_train:n_train + n_valid]
    test  = triples[n_train + n_valid:]

    train_ents = set(h for h,_,_ in train) | set(t for _,_,t in train)

    def filter_unseen(split):
        kept, moved = [], []
        for h, r, t in split:
            (kept if h in train_ents and t in train_ents else moved).append((h,r,t))
        return kept, moved

    valid_clean, mv = filter_unseen(valid)
    test_clean,  mt = filter_unseen(test)
    train = train + mv + mt

    all_te = set(h for h,_,_ in train) | set(t for _,_,t in train)
    assert set(h for h,_,_ in valid_clean).issubset(all_te)
    assert set(h for h,_,_ in test_clean).issubset(all_te)
    print(f"\n[OK] Train : {len(train):,}  Valid : {len(valid_clean):,}  Test : {len(test_clean):,}")
    print("[OK] No entity leakage ✓")

    save_txt(train,       "data/train.txt")
    save_txt(valid_clean, "data/valid.txt")
    save_txt(test_clean,  "data/test.txt")

    with open("data/entity2id.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(e2id)}\n")
        for e, i in e2id.items():
            f.write(f"{e}\t{i}\n")

    with open("data/relation2id.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(r2id)}\n")
        for r, i in r2id.items():
            f.write(f"{r}\t{i}\n")

    print(f"\n[OK] entity2id.txt   ({len(e2id):,} entities)")
    print(f"[OK] relation2id.txt ({len(r2id)} relations)")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C – TransE & DistMult training
# ══════════════════════════════════════════════════════════════════════════════

def norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def neg_sample(pos: np.ndarray, n_ent: int) -> np.ndarray:
    neg = pos.copy()
    mask = np.random.rand(len(pos)) < 0.5
    neg[mask,  2] = np.random.randint(0, n_ent, mask.sum())
    neg[~mask, 0] = np.random.randint(0, n_ent, (~mask).sum())
    return neg


class TransE:
    def __init__(self, n_ent, n_rel, dim=100, margin=2.0, lr=0.01):
        self.margin, self.lr = margin, lr
        b = 6 / np.sqrt(dim)
        self.E = np.random.uniform(-b, b, (n_ent, dim)).astype(np.float32)
        self.R = norm_rows(np.random.uniform(-b, b, (n_rel, dim)).astype(np.float32))

    def _score(self, h, r, t):
        return np.sum(np.abs(norm_rows(self.E[h]) + self.R[r] - norm_rows(self.E[t])), axis=1)

    def score_all_tails(self, h, r):
        return -np.sum(np.abs(norm_rows(self.E[[h]]) + self.R[[r]] - norm_rows(self.E)), axis=1)

    def score_all_heads(self, r, t):
        return -np.sum(np.abs(norm_rows(self.E) + self.R[[r]] - norm_rows(self.E[[t]])), axis=1)

    def train_step(self, pos, neg):
        sp, sn = self._score(pos[:,0], pos[:,1], pos[:,2]), self._score(neg[:,0], neg[:,1], neg[:,2])
        mask   = (self.margin + sp - sn) > 0
        En     = norm_rows(self.E)
        dp     = En[pos[:,0]] + self.R[pos[:,1]] - En[pos[:,2]]
        dn     = En[neg[:,0]] + self.R[neg[:,1]] - En[neg[:,2]]
        gp     = np.sign(dp) * mask[:,None]
        gn     = np.sign(dn) * mask[:,None]
        np.add.at(self.E, pos[:,0], -self.lr * gp)
        np.add.at(self.E, pos[:,2],  self.lr * gp)
        np.add.at(self.E, neg[:,0], -self.lr * gn)
        np.add.at(self.E, neg[:,2],  self.lr * gn)
        np.add.at(self.R, pos[:,1], -self.lr * gp)
        np.add.at(self.R, neg[:,1], -self.lr * gn)
        self.R = norm_rows(self.R)
        return float(np.mean(np.maximum(0, self.margin + sp - sn)))


class DistMult:
    def __init__(self, n_ent, n_rel, dim=100, lr=0.01, reg=1e-3):
        self.lr, self.reg = lr, reg
        sc = np.sqrt(2.0 / dim)
        self.E = np.random.normal(0, sc, (n_ent, dim)).astype(np.float32)
        self.R = np.random.normal(0, sc, (n_rel, dim)).astype(np.float32)

    @staticmethod
    def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _score(self, h, r, t): return np.sum(self.E[h] * self.R[r] * self.E[t], axis=1)

    def score_all_tails(self, h, r): return np.dot(self.E * self.R[r], self.E[h])
    def score_all_heads(self, r, t): return np.dot(self.E * self.R[r], self.E[t])

    def train_step(self, pos, neg):
        sp, sn = self._score(pos[:,0], pos[:,1], pos[:,2]), self._score(neg[:,0], neg[:,1], neg[:,2])
        ep, en = self._sig(sp) - 1.0, self._sig(sn)
        loss   = -np.log(self._sig(sp)+1e-9).mean() - np.log(1 - self._sig(sn)+1e-9).mean()
        lr = self.lr / len(pos)
        for err, hs, rs, ts in [(ep, pos[:,0], pos[:,1], pos[:,2]),
                                 (en, neg[:,0], neg[:,1], neg[:,2])]:
            g = err[:,None]
            np.add.at(self.E, hs, -lr * g * self.R[rs] * self.E[ts])
            np.add.at(self.E, ts, -lr * g * self.R[rs] * self.E[hs])
            np.add.at(self.R, rs, -lr * g * self.E[hs] * self.E[ts])
        self.E *= (1 - self.lr * self.reg)
        self.R *= (1 - self.lr * self.reg)
        return float(loss)


def _load_data():
    def load_txt(path):
        t = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) == 3: t.append(tuple(p))
        return t

    train_r = load_txt("data/train.txt")
    valid_r = load_txt("data/valid.txt")
    test_r  = load_txt("data/test.txt")
    all_r   = train_r + valid_r + test_r

    entities  = sorted(set(h for h,_,_ in all_r) | set(t for _,_,t in all_r))
    relations = sorted(set(r for _,r,_ in all_r))
    e2id = {e: i for i, e in enumerate(entities)}
    r2id = {r: i for i, r in enumerate(relations)}
    id2e = {i: e for e, i in e2id.items()}

    def encode(raw):
        return np.array(
            [(e2id[h], r2id[r], e2id[t]) for h,r,t in raw if h in e2id and r in r2id and t in e2id],
            dtype=np.int32
        )

    train_ids = encode(train_r)
    valid_ids = encode(valid_r)
    test_ids  = encode(test_r)

    filter_tail = defaultdict(set)
    filter_head = defaultdict(set)
    for h, r, t in np.vstack([train_ids, valid_ids, test_ids]):
        filter_tail[(int(h), int(r))].add(int(t))
        filter_head[(int(r), int(t))].add(int(h))

    return (train_ids, valid_ids, test_ids, entities, relations,
            e2id, r2id, id2e, filter_tail, filter_head)


def _train_model(model, name, train_ids, n_ent,
                 dim=100, epochs=50, batch=2048):
    print(f"\n── Training {name} ──────────────────────────────────────")
    idx = np.arange(len(train_ids))
    for epoch in range(1, epochs + 1):
        if epoch == 20: model.lr *= 0.5
        if epoch == 35: model.lr *= 0.5
        np.random.shuffle(idx)
        total, nb, t0 = 0.0, 0, time.time()
        for start in range(0, len(idx), batch):
            bi  = idx[start:start + batch]
            pos = train_ids[bi]
            neg = neg_sample(pos, n_ent)
            total += model.train_step(pos, neg); nb += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  Loss={total/nb:.4f}  ({time.time()-t0:.1f}s)")
    np.save(f"models/{name.lower()}_ent.npy", model.E)
    np.save(f"models/{name.lower()}_rel.npy", model.R)
    print(f"[OK] Saved → models/{name.lower()}_ent/rel.npy")
    return model


def _evaluate(model, test_ids, filter_tail, filter_head, n_eval=None):
    sample = test_ids if n_eval is None else test_ids[:n_eval]
    n = len(sample)
    mrr_h=mrr_t=h1_h=h3_h=h10_h=h1_t=h3_t=h10_t=0.0
    for i, (h,r,t) in enumerate(sample):
        if i % 50 == 0: print(f"  {i}/{n}…", end="\r")
        h, r, t = int(h), int(r), int(t)
        sc_t = model.score_all_tails(h, r)
        for ft in filter_tail[(h,r)]:
            if ft != t: sc_t[ft] = -1e9
        rank_t = int(np.sum(sc_t > sc_t[t])) + 1
        mrr_t += 1/rank_t; h1_t += rank_t<=1; h3_t += rank_t<=3; h10_t += rank_t<=10
        sc_h = model.score_all_heads(r, t)
        for fh in filter_head[(r,t)]:
            if fh != h: sc_h[fh] = -1e9
        rank_h = int(np.sum(sc_h > sc_h[h])) + 1
        mrr_h += 1/rank_h; h1_h += rank_h<=1; h3_h += rank_h<=3; h10_h += rank_h<=10
    return {
        "MRR":      (mrr_h+mrr_t)/(2*n), "Hits@1":  (h1_h+h1_t)/(2*n),
        "Hits@3":   (h3_h+h3_t)/(2*n),   "Hits@10": (h10_h+h10_t)/(2*n),
        "MRR_head": mrr_h/n,  "MRR_tail": mrr_t/n,
        "H1_head":  h1_h/n,   "H1_tail":  h1_t/n,
        "H10_head": h10_h/n,  "H10_tail": h10_t/n,
    }


def _print_results(res, name):
    print(f"\n── Results {name} (filtered) ─────────────────────────────")
    print(f"  {'Metric':<10} {'Global':>8}  {'Head':>8}  {'Tail':>8}")
    for m, hk, tk in [("MRR","MRR_head","MRR_tail"),("Hits@1","H1_head","H1_tail"),
                      ("Hits@3",None,None),("Hits@10","H10_head","H10_tail")]:
        g = res[m]
        hv = f"{res[hk]:.4f}" if hk else "  —   "
        tv = f"{res[tk]:.4f}" if tk else "  —   "
        print(f"  {m:<10} {g:>8.4f}  {hv:>8}  {tv:>8}")


def run_section_c():
    print("\n" + "=" * 60)
    print("  SECTION C — KGE Training (TransE + DistMult)")
    print("=" * 60)

    required = ["data/train.txt", "data/valid.txt", "data/test.txt"]
    if not all(os.path.exists(p) for p in required):
        print("[SKIP] data/ files not found — run Section B first.")
        return

    (train_ids, valid_ids, test_ids, entities, relations,
     e2id, r2id, id2e, filter_tail, filter_head) = _load_data()

    n_ent, n_rel = len(e2id), len(r2id)
    print(f"[OK] Entities: {n_ent:,}   Relations: {n_rel}")
    print(f"[OK] Train / Valid / Test : {len(train_ids):,} / {len(valid_ids):,} / {len(test_ids):,}")

    DIM, EPOCHS, BATCH = 100, 50, 2048

    transe   = TransE(n_ent, n_rel, dim=DIM)
    transe   = _train_model(transe,   "TransE",   train_ids, n_ent, DIM, EPOCHS, BATCH)
    res_te   = _evaluate(transe, test_ids, filter_tail, filter_head)
    _print_results(res_te, "TransE")

    distmult = DistMult(n_ent, n_rel, dim=DIM)
    distmult = _train_model(distmult, "DistMult", train_ids, n_ent, DIM, EPOCHS, BATCH)
    res_dm   = _evaluate(distmult, test_ids, filter_tail, filter_head)
    _print_results(res_dm, "DistMult")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║     MODEL COMPARISON (filtered test)                 ║")
    print("╠══════════════╦════════════╦════════════╦═════════════╣")
    print("║ Metric       ║  TransE    ║  DistMult  ║  Best       ║")
    print("╠══════════════╬════════════╬════════════╬═════════════╣")
    for m in ["MRR","Hits@1","Hits@3","Hits@10"]:
        vt, vd = res_te[m], res_dm[m]
        w = "TransE ✓" if vt >= vd else "DistMult ✓"
        print(f"║ {m:<12} ║  {vt:.4f}    ║  {vd:.4f}    ║ {w:<11} ║")
    print("╚══════════════╩════════════╩════════════╩═════════════╝")

    with open("models/results.json", "w") as f:
        json.dump(
            {"transe":   {k: float(v) for k,v in res_te.items()},
             "distmult": {k: float(v) for k,v in res_dm.items()}},
            f, indent=2
        )
    print("[DONE] models/results.json saved")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D – Visualisation & analysis
# ══════════════════════════════════════════════════════════════════════════════

WD_LABELS = {
    "Q5": "human", "Q16": "Canada", "Q30": "USA", "Q17": "Japan",
    "Q142": "France", "Q183": "Germany", "Q145": "UK", "Q60": "NYC",
    "Q61": "Washington DC",
}
WD_NAMES = {"P31":"instance_of","P166":"award_rcvd","P17":"country",
            "P856":"website","P1346":"winner","P452":"industry","P159":"HQ"}


def qlabel(qid): return WD_LABELS.get(qid, qid)


def run_section_d():
    print("\n" + "=" * 60)
    print("  SECTION D — Visualisation & Analysis")
    print("=" * 60)

    required_models = ["models/transe_ent.npy", "models/transe_rel.npy",
                       "models/distmult_ent.npy", "models/distmult_rel.npy"]
    if not all(os.path.exists(p) for p in required_models):
        print("[SKIP] Model files not found — run Section C first.")
        return

    (train_ids, _, test_ids, entities, relations,
     e2id, r2id, id2e, filter_tail, filter_head) = _load_data()

    n_ent = len(e2id)
    ent_emb_te = np.load("models/transe_ent.npy")
    rel_emb_te = np.load("models/transe_rel.npy")
    print("[OK] Embeddings loaded")

    # ── KB-size sensitivity ──
    def quick_mrr(sub_ids, n_e, n_r, epochs=15, n_eval=80):
        DIM = 50; LR = 0.01; MARGIN = 2.0; BATCH = 512
        b = 6 / np.sqrt(DIM)
        E = np.random.uniform(-b, b, (n_e, DIM)).astype(np.float32)
        R = norm_rows(np.random.uniform(-b, b, (n_r, DIM)).astype(np.float32))
        arr = np.array(sub_ids, dtype=np.int32)
        for ep in range(epochs):
            if ep == 10: LR *= 0.5
            np.random.shuffle(arr)
            for s in range(0, len(arr), BATCH):
                pos = arr[s:s+BATCH]
                neg = neg_sample(pos, n_e)
                sp  = np.sum(np.abs(norm_rows(E[pos[:,0]]) + R[pos[:,1]] - norm_rows(E[pos[:,2]])), 1)
                sn  = np.sum(np.abs(norm_rows(E[neg[:,0]]) + R[neg[:,1]] - norm_rows(E[neg[:,2]])), 1)
                mask = (MARGIN + sp - sn) > 0
                gp = np.sign(norm_rows(E[pos[:,0]]) + R[pos[:,1]] - norm_rows(E[pos[:,2]])) * mask[:,None]
                gn = np.sign(norm_rows(E[neg[:,0]]) + R[neg[:,1]] - norm_rows(E[neg[:,2]])) * mask[:,None]
                np.add.at(E, pos[:,0], -LR*gp); np.add.at(E, pos[:,2],  LR*gp)
                np.add.at(E, neg[:,0], -LR*gn); np.add.at(E, neg[:,2],  LR*gn)
                np.add.at(R, pos[:,1], -LR*gp); np.add.at(R, neg[:,1], -LR*gn)
                R[:] = norm_rows(R)
        E_n = norm_rows(E)
        mrr = 0.0
        for h, r, t in sub_ids[:n_eval]:
            sc = -np.sum(np.abs(E_n[[h]*n_e] + R[[r]*n_e] - E_n), axis=1)
            mrr += 1.0 / (int(np.sum(sc > sc[t])) + 1)
        return mrr / min(len(sub_ids), n_eval)

    sizes = [20_000, 50_000, len(train_ids)]
    labels_sz = ["20k", "50k", f"Full (~{len(train_ids)//1000}k)"]
    mrrs = []
    for sz, lbl in zip(sizes, labels_sz):
        sub  = list(train_ids[:sz])
        es   = sorted(set(h for h,_,_ in sub) | set(t for _,_,t in sub))
        rs   = sorted(set(r for _,r,_ in sub))
        e2s  = {e: i for i,e in enumerate(es)}
        r2s  = {r: i for i,r in enumerate(rs)}
        subr = [(e2s[h],r2s[r],e2s[t]) for h,r,t in sub if h in e2s and r in r2s and t in e2s]
        mrr  = quick_mrr(subr, len(es), len(rs))
        mrrs.append(mrr)
        print(f"  {lbl:<14} ({len(subr):>7,} triples)  MRR = {mrr:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(labels_sz, mrrs, marker="o", linewidth=2, color="steelblue", markersize=10)
    ax.set_title("Impact of KB size on MRR (TransE)")
    ax.set_xlabel("KB size"); ax.set_ylabel("MRR (filtered)")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(mrrs):
        ax.annotate(f"{v:.4f}", (labels_sz[i], v), textcoords="offset points", xytext=(0, 10), ha="center")
    plt.tight_layout()
    plt.savefig("outputs/kb_size_sensitivity.png", dpi=150)
    plt.close()
    print("[OK] → outputs/kb_size_sensitivity.png")

    # ── Nearest neighbours ──
    ent_freq = Counter(h for h,_,_ in train_ids) + Counter(t for _,_,t in train_ids)
    top_ents = [eid for eid, _ in ent_freq.most_common(10)]
    E_norm   = norm_rows(ent_emb_te)

    def nearest(eid, k=3):
        cos = E_norm @ E_norm[eid]
        cos[eid] = -1
        return [(id2e[i], float(cos[i])) for i in np.argsort(cos)[::-1][:k]]

    print(f"\n  {'Q-id':<10} {'Label':<18}  Nearest neighbours (cosine)")
    for eid in top_ents[:8]:
        nns = nearest(eid)
        nn_str = "  |  ".join(f"{qlabel(n)} ({s:.3f})" for n,s in nns)
        print(f"  {id2e[eid]:<10} {qlabel(id2e[eid]):<18}  {nn_str}")

    # ── t-SNE ──
    p31_id = r2id.get("P31")
    entity_type = {}
    if p31_id is not None:
        for h, r, t in train_ids:
            if r == p31_id:
                entity_type[h] = id2e[t]
    type_freq  = Counter(entity_type.values())
    top_types  = {t for t,_ in type_freq.most_common(6)}
    typed_ents = [eid for eid, typ in entity_type.items() if typ in top_types]
    if len(typed_ents) > 600:
        typed_ents = random.sample(typed_ents, 600)

    if len(typed_ents) > 20:
        X    = ent_emb_te[typed_ents]
        lbls = [qlabel(entity_type[i]) for i in typed_ents]
        le   = LabelEncoder(); lenc = le.fit_transform(lbls)
        print(f"\n  t-SNE on {len(typed_ents)} entities, {len(le.classes_)} types…")
        X2d  = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
        fig, ax = plt.subplots(figsize=(11, 8))
        cmap = plt.cm.get_cmap("tab10", len(le.classes_))
        for ci, cn in enumerate(le.classes_):
            mask = lenc == ci
            ax.scatter(X2d[mask,0], X2d[mask,1], c=[cmap(ci)], label=cn, alpha=0.5, s=15)
        ax.set_title("t-SNE of entity embeddings (TransE) — coloured by P31 type")
        ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig("outputs/tsne_entities.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[OK] → outputs/tsne_entities.png")

    # ── Relation norms ──
    rel_counts   = Counter(r for _,r,_ in (list(train_ids) + list(test_ids)))
    top_rels     = [r for r,_ in rel_counts.most_common(15) if r in r2id]
    norms_plot   = [np.linalg.norm(rel_emb_te[r2id[r]]) for r in top_rels]
    fig, ax      = plt.subplots(figsize=(10, 4))
    ax.barh(top_rels, norms_plot, color="steelblue")
    ax.set_title("Relation embedding norms (TransE) — Top 15")
    ax.set_xlabel("‖r‖")
    plt.tight_layout()
    plt.savefig("outputs/relation_norms.png", dpi=150)
    plt.close()
    print("[OK] → outputs/relation_norms.png")

    # ── Results summary ──
    try:
        with open("models/results.json") as f:
            results = json.load(f)
        print(f"\n  {'Metric':<10} {'TransE':>8}  {'DistMult':>8}")
        for m in ["MRR","Hits@1","Hits@3","Hits@10"]:
            print(f"  {m:<10} {results['transe'][m]:>8.4f}  {results['distmult'][m]:>8.4f}")
    except FileNotFoundError:
        pass

    print("\n[DONE] Files in ./outputs/:")
    for f in sorted(os.listdir("outputs")):
        print(f"  • {f}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["A","B","C","D","all"], default="all")
    args = parser.parse_args()

    sections = ["A","B","C","D"] if args.section == "all" else [args.section]
    for sec in sections:
        if sec == "A": run_section_a()
        elif sec == "B": run_section_b()
        elif sec == "C": run_section_c()
        elif sec == "D": run_section_d()
