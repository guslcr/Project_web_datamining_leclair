"""
Lab 6 – RAG Chatbot: SPARQL Generation over Wikidata KB
========================================================
- Loads kb_expanse.nt into rdflib
- Summarises the schema (predicates, classes, sample triples)
- Converts natural-language questions to SPARQL via a local LLM (Ollama)
- Executes queries against rdflib, auto-repairs on failure
- Provides Baseline (LLM only) vs RAG (SPARQL + rdflib) comparison

Prerequisites:
    ollama serve                          # terminal 1
    ollama pull gemma:2b                  # or any supported model
    python src/lab6_rag_chatbot.py        # terminal 2

Usage:
    python src/lab6_rag_chatbot.py [--model MODEL] [--eval]
"""

import argparse
import re
from typing import List, Tuple

import requests
from rdflib import Graph

# ── Configuration ──────────────────────────────────────────────────────────────
NT_FILE         = "kg_artifacts/kb_expanse.nt"
OLLAMA_URL      = "http://localhost:11434/api/generate"
DEFAULT_MODEL   = "gemma:2b"   # alternatives: qwen2.5:0.5b, deepseek-r1:1.5b
MAX_PREDICATES  = 80
MAX_CLASSES     = 40
SAMPLE_TRIPLES  = 20
TRY_REPAIR      = True

# ── Evaluation questions (IRIs confirmed in kb_expanse.nt) ────────────────────
EVAL_QUESTIONS = [
    "Which entities are instances of human (Q5)? Give the first 10.",
    "Which entities received an award (P166)? Give the first 10.",
    "Which entities are located in Canada (Q16)?",
    "Which entities have a known industry (P452)? Give the first 10.",
    "Which entities have subsidiaries (P355)?",
]

# ── Prefix expansion map ───────────────────────────────────────────────────────
PREFIX_MAP = {
    "wdt:":    "http://www.wikidata.org/prop/direct/",
    "wd:":     "http://www.wikidata.org/entity/",
    "rdf:":    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:":   "http://www.w3.org/2000/01/rdf-schema#",
    "xsd:":    "http://www.w3.org/2001/XMLSchema#",
    "owl:":    "http://www.w3.org/2002/07/owl#",
    "schema:": "http://schema.org/",
    "skos:":   "http://www.w3.org/2004/02/skos/core#",
}

CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

SPARQL_SYSTEM_PROMPT = """
You are a SPARQL 1.1 generator for a Wikidata RDF knowledge graph.
Given the SCHEMA SUMMARY below, convert the QUESTION into a valid SELECT query.

Rules:
- NEVER use prefix shortcuts like wdt: or wd: — always write the full IRI in angle brackets.
- Wikidata properties MUST be written as: <http://www.wikidata.org/prop/direct/P159>
- Wikidata entities MUST be written as:   <http://www.wikidata.org/entity/Q142>
- Only use IRIs that appear in the SCHEMA SUMMARY.
- Return ONLY a single fenced code block labelled ```sparql.
- No explanation, no text outside the code block.
"""

REPAIR_SYSTEM_PROMPT = """
The SPARQL query below raised an error. Fix it using the SCHEMA SUMMARY.
Return only the corrected SPARQL in a single ```sparql code block.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1 – Ollama LLM interface
# ══════════════════════════════════════════════════════════════════════════════

def ask_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "[ERROR] Ollama unreachable on http://localhost:11434\n"
            "        Start it first:  ollama serve"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2 – Graph loading
# ══════════════════════════════════════════════════════════════════════════════

def load_graph(nt_path: str) -> Graph:
    g = Graph()
    g.parse(nt_path, format="nt")
    print(f"[OK] {len(g):,} triples loaded from '{nt_path}'")
    return g


# ══════════════════════════════════════════════════════════════════════════════
# 3 – Schema summary (fed into LLM prompts)
# ══════════════════════════════════════════════════════════════════════════════

def _get_prefix_block(g: Graph) -> str:
    defaults = {
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd":  "http://www.w3.org/2001/XMLSchema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
        "wd":   "http://www.wikidata.org/entity/",
        "wdt":  "http://www.wikidata.org/prop/direct/",
    }
    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    for k, v in defaults.items():
        ns_map.setdefault(k, v)
    return "\n".join(sorted(f"PREFIX {p}: <{ns}>" for p, ns in ns_map.items()))


def build_schema_summary(g: Graph) -> str:
    prefixes = _get_prefix_block(g)

    preds = "\n".join(
        f"- {r.p}"
        for r in g.query(f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o }} LIMIT {MAX_PREDICATES}")
    )
    clss = "\n".join(
        f"- {r.cls}"
        for r in g.query(f"SELECT DISTINCT ?cls WHERE {{ ?s a ?cls }} LIMIT {MAX_CLASSES}")
    )
    samples = "\n".join(
        f"- {r.s}  {r.p}  {r.o}"
        for r in g.query(f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {SAMPLE_TRIPLES}")
    )

    return (
        f"{prefixes}\n\n"
        f"# Predicates (up to {MAX_PREDICATES})\n{preds}\n\n"
        f"# Classes / rdf:type (up to {MAX_CLASSES})\n{clss}\n\n"
        f"# Sample triples (up to {SAMPLE_TRIPLES})\n{samples}"
    ).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 4 – NL → SPARQL
# ══════════════════════════════════════════════════════════════════════════════

def expand_prefixes(sparql: str) -> str:
    """Replace prefix shortcuts with full IRIs in non-PREFIX lines."""
    lines = []
    for line in sparql.splitlines():
        if line.strip().upper().startswith("PREFIX"):
            lines.append(line)
            continue
        for prefix, uri in PREFIX_MAP.items():
            line = re.sub(
                rf"(?<![<'/]){re.escape(prefix)}([A-Za-z0-9_]+)",
                lambda m, u=uri: f"<{u}{m.group(1)}>",
                line,
            )
        lines.append(line)
    return "\n".join(lines)


def extract_sparql(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    raw = m.group(1).strip() if m else text.strip()
    return expand_prefixes(raw)


def generate_sparql(question: str, schema: str, model: str) -> str:
    prompt = (
        f"{SPARQL_SYSTEM_PROMPT}\n\n"
        f"SCHEMA SUMMARY:\n{schema}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Return only the SPARQL query in a code block."
    )
    return extract_sparql(ask_llm(prompt, model))


# ══════════════════════════════════════════════════════════════════════════════
# 5 – SPARQL execution + self-repair
# ══════════════════════════════════════════════════════════════════════════════

def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    res  = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows  = [tuple(str(c) for c in r) for r in res]
    return vars_, rows


def repair_sparql(schema: str, question: str, bad_query: str, error: str, model: str) -> str:
    prompt = (
        f"{REPAIR_SYSTEM_PROMPT}\n\n"
        f"SCHEMA SUMMARY:\n{schema}\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"BAD SPARQL:\n{bad_query}\n\n"
        f"ERROR:\n{error}\n\n"
        f"Return only the corrected SPARQL in a code block."
    )
    return extract_sparql(ask_llm(prompt, model))


# ══════════════════════════════════════════════════════════════════════════════
# 6 – RAG orchestration
# ══════════════════════════════════════════════════════════════════════════════

def answer_rag(g: Graph, schema: str, question: str, model: str,
               try_repair: bool = TRY_REPAIR) -> dict:
    sparql = generate_sparql(question, schema, model)
    try:
        vars_, rows = run_sparql(g, sparql)
        return {"query": sparql, "vars": vars_, "rows": rows, "repaired": False, "error": None}
    except Exception as e:
        if try_repair:
            repaired = repair_sparql(schema, question, sparql, str(e), model)
            try:
                vars_, rows = run_sparql(g, repaired)
                return {"query": repaired, "vars": vars_, "rows": rows, "repaired": True, "error": None}
            except Exception as e2:
                return {"query": repaired, "vars": [], "rows": [], "repaired": True, "error": str(e2)}
        return {"query": sparql, "vars": [], "rows": [], "repaired": False, "error": str(e)}


def answer_baseline(question: str, model: str) -> str:
    return ask_llm(f"Answer the following question as precisely as possible:\n\n{question}", model)


# ══════════════════════════════════════════════════════════════════════════════
# 7 – Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def pretty_print(result: dict):
    tag = "(repaired) " if result["repaired"] else ""
    print(f"\n  [SPARQL {tag}used]")
    print(result["query"])
    if result.get("error"):
        print(f"\n  [ERROR] {result['error']}")
        return
    vars_, rows = result["vars"], result["rows"]
    if not rows:
        print("\n  ↳ No results returned.")
        return
    header = " | ".join(f"{v:^30}" for v in vars_)
    print(f"\n  {header}")
    print("  " + "-" * len(header))
    for r in rows[:20]:
        print("  " + " | ".join(f"{c:^30}" for c in r))
    if len(rows) > 20:
        print(f"  … ({len(rows)} rows total, display capped at 20)")


# ══════════════════════════════════════════════════════════════════════════════
# 8 – Evaluation (Baseline vs RAG)
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(g: Graph, schema: str, model: str):
    print("\n" + "=" * 70)
    print("  EVALUATION — Baseline (LLM only)  vs  SPARQL-RAG")
    print("=" * 70)
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print("\n" + "-" * 70)
        print(f"  Q{i}: {q}")
        print("-" * 70)
        print("\n  [Baseline — direct LLM]")
        baseline = answer_baseline(q, model)
        print("  " + baseline[:400].replace("\n", "\n  "))
        print("\n  [RAG — SPARQL + rdflib]")
        pretty_print(answer_rag(g, schema, q, model))
    print("\n" + "=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 9 – Interactive CLI
# ══════════════════════════════════════════════════════════════════════════════

def cli_loop(g: Graph, schema: str, model: str):
    print(f"\n[RAG Chatbot — Wikidata KB]  model={model}")
    print("Commands: 'eval' to run evaluation, 'quit' to exit.\n")
    while True:
        try:
            q = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q.lower() == "eval":
            run_evaluation(g, schema, model)
            continue
        if not q:
            continue
        print("\n── Baseline (LLM only) " + "─" * 45)
        print(answer_baseline(q, model))
        print("\n── RAG (SPARQL-generation) " + "─" * 40)
        pretty_print(answer_rag(g, schema, q, model))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama model name (default: gemma:2b)")
    parser.add_argument("--eval", action="store_true",
                        help="Run automated evaluation then exit")
    args = parser.parse_args()

    g      = load_graph(NT_FILE)
    schema = build_schema_summary(g)

    print("\n[Schema summary (excerpt)]")
    print(schema[:600])
    print("…\n")

    if args.eval:
        run_evaluation(g, schema, args.model)
    else:
        cli_loop(g, schema, args.model)
