"""
Lab 1 – From the Unstructured Web to Structured Entities
=========================================================
Phase 1 : Web crawling & cleaning  → outputs/crawler_output.jsonl
Phase 2 : NER + relation extraction → outputs/extracted_knowledge.csv
                                      outputs/relations.csv

The pipeline:
  1. Check robots.txt for each seed URL (ethics)
  2. Fetch & extract main content with trafilatura (boilerplate removal)
  3. Filter pages by usefulness (>= 500 words)
  4. Run spaCy NER (PERSON, ORG, GPE, DATE) with entity-level cleaning
  5. Extract (subject, verb, object) triples via dependency parsing

Usage:
    python src/lab1_scraping.py
"""

import csv
import json
import os
import re
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import spacy
import trafilatura

# ── Configuration ──────────────────────────────────────────────────────────────
MIN_WORDS = 500
OUTPUT_DIR = "outputs"
CORPUS_FILE = os.path.join(OUTPUT_DIR, "crawler_output.jsonl")
ENTITIES_CSV = os.path.join(OUTPUT_DIR, "extracted_knowledge.csv")
RELATIONS_CSV = os.path.join(OUTPUT_DIR, "relations.csv")

USER_AGENT = "KBLabBot/1.0 (student project; contact@example.com)"

# Entity types relevant to our domain
ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "DATE"}

# Domain: Bitcoin & Cryptocurrency — 7 seed URLs with rich text
SEED_URLS = [
    "https://www.gemini.com/en-GB/cryptopedia/what-is-bitcoin",
    "https://bitcoinmagazine.com/guides/what-is-bitcoin",
    "https://www.ledger.com/academy/what-is-bitcoin",
    "https://bitcoin.org/en/secure-your-wallet",
    "https://www.investopedia.com/terms/b/bitcoin.asp",
    "https://ethereum.org/developers/docs/consensus-mechanisms/pow/",
    "https://river.com/learn/what-is-bitcoin/",
]

# spaCy model — en_core_web_trf (transformer) recommended by lab instructions;
# fall back to en_core_web_lg if not installed
NLP_MODEL_PREFERRED = "en_core_web_trf"
NLP_MODEL_FALLBACK = "en_core_web_lg"

# ── Noise filters ──────────────────────────────────────────────────────────────
# Generic terms that spaCy may tag as ORG/GPE but are really common nouns
ENTITY_BLACKLIST = {
    "bitcoin", "blockchain", "cryptocurrency", "crypto", "miners",
    "btc", "eth", "digital", "virtual", "internet", "web",
    "proof of work", "proof of stake", "consensus", "protocol",
    "blockchain technology", "digital currency", "virtual currency",
    "a 'virtual' currency", "the blockchain",
}

# Strings typically found in boilerplate leftovers
NOISE_PATTERNS = re.compile(
    r"(read more|share|skip to|cookie|newsletter|subscribe|sign up|"
    r"log ?in|menu|privacy policy|terms of service|buy bitcoin|"
    r"support bitcoin|events|community|resources|documentation)",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def reset_output_files():
    """Remove previously generated files to avoid appending duplicates."""
    for path in [CORPUS_FILE, ENTITIES_CSV, RELATIONS_CSV]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  [RESET] Removed: {path}")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def has_bad_encoding(text: str) -> bool:
    return bool(re.search(r"[Ã¢ÐÑ�¦]", text))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Web Crawling & Cleaning
# ══════════════════════════════════════════════════════════════════════════════

def check_robots_txt(url: str) -> bool:
    """Return True if our user-agent is allowed to fetch *url* per robots.txt."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)
        if not allowed:
            print(f"  [BLOCKED] robots.txt disallows: {url}")
        return allowed
    except Exception:
        # If robots.txt is unreachable, assume allowed (common convention)
        return True


def fetch_and_extract(url: str) -> Optional[dict]:
    """Fetch HTML with trafilatura, extract main content, return record or None."""
    try:
        html = trafilatura.fetch_url(url)
    except Exception as e:
        print(f"  [ERROR] Fetch failed: {url} — {e}")
        return None

    if not html:
        print(f"  [SKIP]  Empty response: {url}")
        return None

    # Extract main text (boilerplate removal)
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=True,
        favor_precision=True,
    )
    if not text:
        print(f"  [SKIP]  No main content extracted: {url}")
        return None

    text = normalize_ws(text)
    word_count = len(text.split())

    # Usefulness check: at least MIN_WORDS words
    if word_count < MIN_WORDS:
        print(f"  [SKIP]  Only {word_count} words (min {MIN_WORDS}): {url}")
        return None

    # Try to extract title from metadata
    metadata = trafilatura.extract(html, output_format="json", include_comments=False)
    title = ""
    if metadata:
        try:
            meta_dict = json.loads(metadata)
            title = meta_dict.get("title", "")
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "url": url,
        "title": title,
        "word_count": word_count,
        "text": text,
    }


def save_jsonl_record(record: dict, filepath: str):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_crawling_pipeline(urls: list[str]) -> list[dict]:
    """Crawl all URLs, respecting robots.txt. Returns loaded documents."""
    print("\n── Phase 1: Web Crawling & Cleaning ──")
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)

        # Ethics: check robots.txt first
        if not check_robots_txt(url):
            continue

        record = fetch_and_extract(url)
        if record:
            save_jsonl_record(record, CORPUS_FILE)
            print(f"  [OK]    {record['word_count']} words — {url}")

    # Read back all records (handles dedup across runs)
    return read_jsonl(CORPUS_FILE)


def read_jsonl(path: str) -> list[dict]:
    docs = []
    if not os.path.exists(path):
        return docs
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                key = d.get("url", "")
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
            except json.JSONDecodeError:
                pass
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Information Extraction
# ══════════════════════════════════════════════════════════════════════════════

# ── 2.1 Named Entity Recognition ──────────────────────────────────────────────

def clean_entity(text: str) -> str:
    """Normalize entity surface form."""
    text = normalize_ws(text)
    text = text.strip(" \n\t\r-–—|,:;[](){}\"'")
    return text


def is_valid_entity(text: str, label: str) -> bool:
    """Filter out noisy or generic entities."""
    if not text or len(text) < 2:
        return False

    lowered = text.lower()

    # Blacklisted domain terms that are common nouns, not named entities
    if lowered in ENTITY_BLACKLIST:
        return False

    # Must start with uppercase (proper noun) — except for DATE
    if label != "DATE" and text[0].islower():
        return False

    # All lowercase → likely a common noun mis-tagged
    if label != "DATE" and text.islower():
        return False

    # Too long → likely a sentence fragment
    if len(text) > 60:
        return False

    # Contains boilerplate patterns
    if NOISE_PATTERNS.search(text):
        return False

    # URLs, brackets, broken encoding
    if "http" in text or "www." in text:
        return False
    if re.search(r"[\[\]{}<>]", text):
        return False
    if has_bad_encoding(text):
        return False

    # Mostly digits (except DATE)
    if label != "DATE" and sum(c.isdigit() for c in text) > len(text) * 0.5:
        return False

    return True


def extract_entities(documents: list[dict], nlp) -> list[dict]:
    """Extract and deduplicate named entities from all documents."""
    results = []
    seen = set()

    for doc_data in documents:
        url = doc_data["url"]
        doc = nlp(doc_data["text"][:100_000])  # cap for memory

        for ent in doc.ents:
            if ent.label_ not in ALLOWED_LABELS:
                continue

            entity_text = clean_entity(ent.text)
            if not is_valid_entity(entity_text, ent.label_):
                continue

            # Get the sentence context
            sentence = normalize_ws(ent.sent.text)
            if len(sentence.split()) < 4:
                continue

            # Dedup key: (entity, label, source_url)
            key = (entity_text, ent.label_, url)
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "entity": entity_text,
                "label": ent.label_,
                "sentence": sentence,
                "source_url": url,
            })

    return results


# ── 2.2 Relation Extraction ──────────────────────────────────────────────────

def _entity_span_map(sent) -> dict[int, str]:
    """Map token indices → entity text for all valid named entities in *sent*."""
    tok2ent = {}
    for ent in sent.ents:
        if ent.label_ not in ALLOWED_LABELS:
            continue
        ent_text = clean_entity(ent.text)
        if not is_valid_entity(ent_text, ent.label_):
            continue
        for token in ent:
            tok2ent[token.i] = ent_text
    return tok2ent


def _resolve_entity(token, tok2ent: dict) -> Optional[str]:
    """Return entity text if *token* or one of its children is part of an entity."""
    if token.i in tok2ent:
        return tok2ent[token.i]
    for child in token.children:
        if child.i in tok2ent:
            return tok2ent[child.i]
    return None


def extract_relations(documents: list[dict], nlp) -> list[dict]:
    """
    Extract (subject, relation, object) triples using dependency parsing.
    For each VERB token, try to find:
      - nsubj / nsubjpass → subject entity
      - dobj / pobj / attr → object entity
    Only keep triples where both endpoints are named entities.
    """
    relations = []
    seen = set()

    for doc_data in documents:
        url = doc_data["url"]
        doc = nlp(doc_data["text"][:100_000])

        for sent in doc.sents:
            sentence = normalize_ws(sent.text)
            if len(sentence.split()) < 5:
                continue

            tok2ent = _entity_span_map(sent)
            # Need at least 2 distinct entities in the sentence
            if len(set(tok2ent.values())) < 2:
                continue

            for token in sent:
                if token.pos_ not in {"VERB", "AUX"}:
                    continue

                relation = normalize_ws(token.lemma_.lower())
                if len(relation) < 2 or has_bad_encoding(relation):
                    continue

                subject = None
                obj = None

                for child in token.children:
                    # Subject
                    if child.dep_ in {"nsubj", "nsubjpass"} and subject is None:
                        subject = _resolve_entity(child, tok2ent)

                    # Direct object
                    if child.dep_ in {"dobj", "obj", "attr", "oprd"} and obj is None:
                        obj = _resolve_entity(child, tok2ent)

                    # Prepositional object (e.g. "located in France")
                    if child.dep_ == "prep" and obj is None:
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                obj = _resolve_entity(gc, tok2ent)
                                if obj:
                                    # Include preposition in relation label
                                    relation = f"{relation} {child.text.lower()}"
                                    break

                # Passive / relative clause fallback
                if subject is None and token.dep_ in {"acl", "relcl"}:
                    subject = _resolve_entity(token.head, tok2ent)

                if subject and obj and subject != obj:
                    key = (subject, relation, obj, url)
                    if key not in seen:
                        seen.add(key)
                        relations.append({
                            "subject": subject,
                            "relation": relation,
                            "object": obj,
                            "sentence": sentence,
                            "source_url": url,
                        })

    return relations


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(data: list[dict], path: str, fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"  [OK] {path}  ({len(data)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_output_dir()
    reset_output_files()

    # ── Phase 1 ──
    print("=" * 60)
    print("  Lab 1 — From the Unstructured Web to Structured Entities")
    print("=" * 60)

    documents = run_crawling_pipeline(SEED_URLS)
    if not documents:
        print("[ERROR] No documents in corpus — check URLs / network.")
        raise SystemExit(1)

    print(f"\n  Corpus: {len(documents)} documents saved to {CORPUS_FILE}")
    for i, d in enumerate(documents, 1):
        print(f"    {i}. [{d.get('word_count', '?')} words] {d.get('url')}")

    # ── Phase 2 ──
    print("\n── Phase 2: Information Extraction (NER + Relations) ──")

    # Load spaCy model
    try:
        nlp = spacy.load(NLP_MODEL_PREFERRED)
        print(f"  Model: {NLP_MODEL_PREFERRED}")
    except OSError:
        print(f"  {NLP_MODEL_PREFERRED} not found, falling back to {NLP_MODEL_FALLBACK}")
        nlp = spacy.load(NLP_MODEL_FALLBACK)

    # NER
    entities = extract_entities(documents, nlp)
    print(f"\n  Entities extracted: {len(entities)}")

    # Show entity type distribution
    from collections import Counter
    label_counts = Counter(e["label"] for e in entities)
    for lbl, cnt in label_counts.most_common():
        print(f"    {lbl:<8} {cnt}")

    # Relations
    relations = extract_relations(documents, nlp)
    print(f"\n  Relations extracted: {len(relations)}")
    if relations:
        print("  Sample triples:")
        for r in relations[:5]:
            print(f"    ({r['subject']}) —[{r['relation']}]→ ({r['object']})")

    # ── Save outputs ──
    print("\n── Saving outputs ──")
    save_csv(
        entities,
        ENTITIES_CSV,
        ["entity", "label", "sentence", "source_url"],
    )
    save_csv(
        relations,
        RELATIONS_CSV,
        ["subject", "relation", "object", "sentence", "source_url"],
    )

    print(f"\n[DONE] Files: {CORPUS_FILE}, {ENTITIES_CSV}, {RELATIONS_CSV}")
