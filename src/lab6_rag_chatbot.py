"""
Lab 6 – RAG Chatbot: SPARQL Generation over Wikidata KB
========================================================
- Loads kb_expanse.nt into rdflib
- Summarises the schema (predicates, classes, sample triples)
- Converts natural-language questions to SPARQL via a local LLM (Ollama)
- Executes queries against rdflib, auto-repairs on failure
- Falls back to template-based SPARQL when LLM fails
- Provides Baseline (LLM only) vs RAG (SPARQL + rdflib) comparison

Prerequisites:
    ollama serve                          # terminal 1
    ollama pull mistral:7b                # or any supported model
    python src/lab6_rag_chatbot.py        # terminal 2

Usage:
    python src/lab6_rag_chatbot.py [--model MODEL] [--eval]
"""

import argparse
import os
import re
from typing import List, Tuple
from urllib.parse import unquote

import requests
from rdflib import Graph, URIRef

# ── Configuration ──────────────────────────────────────────────────────────────
NT_FILE_CANDIDATES = [
    "kg_artifacts/kb_expanse.nt",
    "kb_expanse.nt",
]
OLLAMA_URL      = "http://localhost:11434/api/generate"
DEFAULT_MODEL   = "mistral:7b"  # alternatives: llama3:8b, gemma:7b, gemma:2b
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

# Well-known Wikidata property labels (used in schema summary + templates)
KNOWN_PROPERTIES = {
    "P31": "instance of",
    "P17": "country",
    "P571": "inception date",
    "P856": "official website",
    "P159": "headquarters location",
    "P452": "industry",
    "P166": "award received",
    "P1346": "winner",
    "P355": "has subsidiary",
    "P184": "doctoral advisor",
    "P112": "founded by",
    "P127": "owned by",
    "P176": "manufacturer",
    "P138": "named after",
    "P279": "subclass of",
    "P361": "part of",
    "P527": "has part",
    "P1092": "total produced",
    "P8891": "energy consumption",
    "P366": "has use",
    "P4527": "UK Parliament ID",
    "P6200": "BBC Things ID",
    "P3280": "BnF ID",
    "P13561": "IGDB game ID",
    "P14178": "CoinGecko ID",
}

# Well-known Wikidata entity labels
KNOWN_ENTITIES = {
    "Q5": "human",
    "Q4": "artwork",
    "Q16": "Canada",
    "Q30": "United States",
    "Q38": "Italy",
    "Q60": "New York City",
    "Q61": "Washington, D.C.",
    "Q131723": "Bitcoin",
    "Q2283": "Ethereum (not in KB - use local entity)",
    "Q7590": "financial services",
    "Q11034": "electronics",
    "Q11023": "engineering",
    "Q11425": "animation",
    "Q68": "computer science",
    "Q75": "Internet",
}

CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SELECT_RE     = re.compile(r"(SELECT\s.+)", re.IGNORECASE | re.DOTALL)

SPARQL_SYSTEM_PROMPT = """
You are a SPARQL 1.1 generator for a Wikidata-based RDF knowledge graph.
Given the SCHEMA SUMMARY below, convert the QUESTION into a valid SELECT query.

Rules:
- NEVER use prefix shortcuts like wdt: or wd: — always write the FULL IRI in angle brackets.
- Wikidata properties: <http://www.wikidata.org/prop/direct/P159>
- Wikidata entities:   <http://www.wikidata.org/entity/Q142>
- Only use IRIs that appear in the SCHEMA SUMMARY (predicates list or entity list).
- Return ONLY a single fenced code block labelled ```sparql.
- No explanation, no text outside the code block.

## Examples

QUESTION: Which entities are instances of human?
```sparql
SELECT ?entity WHERE {
  ?entity <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> .
} LIMIT 10
```

QUESTION: What is Bitcoin?
```sparql
SELECT ?property ?value WHERE {
  <http://www.wikidata.org/entity/Q131723> ?property ?value .
} LIMIT 20
```

QUESTION: Which entities are located in Canada?
```sparql
SELECT ?entity WHERE {
  ?entity <http://www.wikidata.org/prop/direct/P17> <http://www.wikidata.org/entity/Q16> .
} LIMIT 10
```

QUESTION: Who founded Bitcoin?
```sparql
SELECT ?founder WHERE {
  <http://www.wikidata.org/entity/Q131723> <http://www.wikidata.org/prop/direct/P112> ?founder .
} LIMIT 10
```
"""

REPAIR_SYSTEM_PROMPT = """
The SPARQL query below raised a parse error against an rdflib Graph.
Fix the SPARQL so it is syntactically valid and uses only IRIs from the SCHEMA SUMMARY.

Common mistakes to fix:
- Missing SELECT ... WHERE { } structure
- Using prefix shortcuts (wdt:, wd:) instead of full IRIs in angle brackets
- Mismatched braces or missing dots between triple patterns
- Using IRIs not present in the knowledge graph
- Having more than 3 elements in a triple pattern

Return ONLY the corrected query in a single ```sparql code block.
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

def find_nt_file() -> str:
    for path in NT_FILE_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Cannot find kb_expanse.nt in: {NT_FILE_CANDIDATES}"
    )


def load_graph(nt_path: str) -> Graph:
    g = Graph()
    g.parse(nt_path, format="nt")
    print(f"[OK] {len(g):,} triples loaded from '{nt_path}'")
    return g


# ══════════════════════════════════════════════════════════════════════════════
# 3 – Entity search (find IRIs by keyword in URI fragments + labels)
# ══════════════════════════════════════════════════════════════════════════════

def uri_to_label(uri: str) -> str:
    """Extract a human-readable label from a URI fragment."""
    fragment = uri.rsplit("/", 1)[-1]
    decoded = unquote(fragment).replace("_", " ")
    return decoded


def search_entity(g: Graph, keyword: str, limit: int = 5) -> list[dict]:
    """Search for entities in the graph matching a keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    matches = []

    # First: check if it's a known Q-code
    for qid, label in KNOWN_ENTITIES.items():
        if keyword_lower in label.lower() or keyword_lower == qid.lower():
            uri = f"http://www.wikidata.org/entity/{qid}"
            # Verify it actually exists in the graph
            if (URIRef(uri), None, None) in g or (None, None, URIRef(uri)) in g:
                matches.append({"uri": uri, "label": label, "source": "wikidata"})

    # Second: search in local entity URIs (lab_web4.org)
    seen = set()
    for s in g.subjects():
        s_str = str(s)
        if s_str in seen:
            continue
        seen.add(s_str)
        label = uri_to_label(s_str)
        if keyword_lower in label.lower():
            matches.append({"uri": s_str, "label": label, "source": "local"})
        if len(matches) >= limit:
            break

    return matches[:limit]


# ══════════════════════════════════════════════════════════════════════════════
# 4 – Schema summary (fed into LLM prompts)
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


def _get_label(g: Graph, uri: str) -> str:
    """Return a human-readable label: rdfs:label if available, else URI fragment, else known label."""
    # Try rdfs:label first
    for row in g.query(
        f'SELECT ?lbl WHERE {{ <{uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?lbl }} LIMIT 1'
    ):
        return str(row.lbl)
    # Try known properties / entities
    fragment = uri.rsplit("/", 1)[-1]
    if fragment in KNOWN_PROPERTIES:
        return KNOWN_PROPERTIES[fragment]
    if fragment in KNOWN_ENTITIES:
        return KNOWN_ENTITIES[fragment]
    return ""


def build_schema_summary(g: Graph) -> str:
    prefixes = _get_prefix_block(g)

    # Predicates with labels
    pred_lines = []
    for r in g.query(f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o }} LIMIT {MAX_PREDICATES}"):
        lbl = _get_label(g, str(r.p))
        tag = f' ("{lbl}")' if lbl else ""
        pred_lines.append(f"- <{r.p}>{tag}")
    preds = "\n".join(pred_lines)

    # Classes with labels
    cls_lines = []
    for r in g.query(f"SELECT DISTINCT ?cls WHERE {{ ?s a ?cls }} LIMIT {MAX_CLASSES}"):
        lbl = _get_label(g, str(r.cls))
        tag = f' ("{lbl}")' if lbl else ""
        cls_lines.append(f"- <{r.cls}>{tag}")
    clss = "\n".join(cls_lines)

    # Sample triples with labels
    sample_lines = []
    for r in g.query(f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {SAMPLE_TRIPLES}"):
        s_lbl = _get_label(g, str(r.s))
        p_lbl = _get_label(g, str(r.p))
        o_lbl = _get_label(g, str(r.o)) if str(r.o).startswith("http") else str(r.o)
        s_tag = f" ({s_lbl})" if s_lbl else ""
        p_tag = f" ({p_lbl})" if p_lbl else ""
        o_tag = f" ({o_lbl})" if o_lbl else ""
        sample_lines.append(f"- <{r.s}>{s_tag}  <{r.p}>{p_tag}  <{r.o}>{o_tag}")
    samples = "\n".join(sample_lines)

    # Known entity catalogue
    entity_lines = []
    for qid, label in KNOWN_ENTITIES.items():
        uri = f"http://www.wikidata.org/entity/{qid}"
        if (URIRef(uri), None, None) in g or (None, None, URIRef(uri)) in g:
            entity_lines.append(f'- "{label}" → <{uri}>')
    entity_map = "\n".join(entity_lines)

    return (
        f"{prefixes}\n\n"
        f"# Predicates (up to {MAX_PREDICATES})\n{preds}\n\n"
        f"# Classes / rdf:type (up to {MAX_CLASSES})\n{clss}\n\n"
        f"# Known entities (name → IRI)\n{entity_map}\n\n"
        f"# Sample triples (up to {SAMPLE_TRIPLES})\n{samples}"
    ).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 5 – NL → SPARQL (LLM-based)
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
    # Try fenced code block first
    m = CODE_BLOCK_RE.search(text)
    if m:
        raw = m.group(1).strip()
    else:
        # Fallback: extract everything starting from SELECT
        m2 = SELECT_RE.search(text)
        raw = m2.group(1).strip() if m2 else text.strip()
    return expand_prefixes(raw)


def generate_sparql(question: str, schema: str, model: str) -> str:
    prompt = (
        f"{SPARQL_SYSTEM_PROMPT}\n\n"
        f"SCHEMA SUMMARY:\n{schema}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Look up the known entities list above to find the correct IRI for entities "
        f"mentioned in the question. Write a valid SELECT query. "
        f"Return ONLY the SPARQL inside a ```sparql code block."
    )
    return extract_sparql(ask_llm(prompt, model))


# ══════════════════════════════════════════════════════════════════════════════
# 6 – Template-based SPARQL fallback
# ══════════════════════════════════════════════════════════════════════════════

# Patterns: (regex_on_question, template_function)
# Template functions receive (g, question, regex_match) and return a SPARQL string or None

def _extract_keywords(question: str) -> list[str]:
    """Extract meaningful keywords from a question (skip stop words)."""
    stop = {"what", "which", "who", "is", "are", "the", "a", "an", "of", "in",
            "give", "me", "first", "list", "all", "find", "show", "does", "do",
            "has", "have", "had", "was", "were", "been", "be", "to", "for",
            "with", "from", "by", "about", "how", "many", "much", "can", "could"}
    words = re.findall(r"[A-Za-z0-9]+", question.lower())
    return [w for w in words if w not in stop and len(w) > 1]


def template_about_entity(g: Graph, question: str) -> str | None:
    """Handle 'What is X?' / 'Tell me about X' / 'Who is X?' questions."""
    keywords = _extract_keywords(question)
    for kw in keywords:
        results = search_entity(g, kw, limit=1)
        if results:
            uri = results[0]["uri"]
            return (
                f"SELECT ?property ?value WHERE {{\n"
                f"  <{uri}> ?property ?value .\n"
                f"}} LIMIT 20"
            )
    return None


def template_instances_of(g: Graph, question: str) -> str | None:
    """Handle 'Which entities are instances of X?' questions."""
    keywords = _extract_keywords(question)
    for kw in keywords:
        # Try to match a known class
        for qid, label in KNOWN_ENTITIES.items():
            if kw in label.lower():
                uri = f"http://www.wikidata.org/entity/{qid}"
                return (
                    f"SELECT ?entity WHERE {{\n"
                    f"  ?entity <http://www.wikidata.org/prop/direct/P31> <{uri}> .\n"
                    f"}} LIMIT 10"
                )
    return None


def template_property_search(g: Graph, question: str) -> str | None:
    """Handle questions about a specific property (award, country, industry, etc.)."""
    q_lower = question.lower()
    # Map question keywords to Wikidata properties
    property_keywords = {
        "award":       "P166",
        "country":     "P17",
        "located":     "P17",
        "industry":    "P452",
        "subsidiary":  "P355",
        "subsidiaries":"P355",
        "headquarters":"P159",
        "website":     "P856",
        "founded":     "P112",
        "founder":     "P112",
        "winner":      "P1346",
    }
    for keyword, pid in property_keywords.items():
        if keyword in q_lower:
            # Check if the question also mentions a specific entity
            other_keywords = [k for k in _extract_keywords(question) if k != keyword]
            for kw in other_keywords:
                results = search_entity(g, kw, limit=1)
                if results:
                    uri = results[0]["uri"]
                    prop_uri = f"http://www.wikidata.org/prop/direct/{pid}"
                    return (
                        f"SELECT ?result WHERE {{\n"
                        f"  <{uri}> <{prop_uri}> ?result .\n"
                        f"}} LIMIT 10"
                    )
            # No specific entity — list all entities with this property
            prop_uri = f"http://www.wikidata.org/prop/direct/{pid}"
            return (
                f"SELECT ?entity ?value WHERE {{\n"
                f"  ?entity <{prop_uri}> ?value .\n"
                f"}} LIMIT 10"
            )
    return None


def template_fallback(g: Graph, question: str) -> str | None:
    """Last resort: search for any entity matching a keyword and show its triples."""
    keywords = _extract_keywords(question)
    for kw in keywords:
        results = search_entity(g, kw, limit=1)
        if results:
            uri = results[0]["uri"]
            return (
                f"SELECT ?property ?value WHERE {{\n"
                f"  <{uri}> ?property ?value .\n"
                f"}} LIMIT 20"
            )
    return None


TEMPLATE_CHAIN = [
    template_property_search,   # Most specific — tries to match a property + entity
    template_instances_of,      # "instances of X"
    template_about_entity,      # "what is X"
    template_fallback,          # Last resort
]


def generate_sparql_template(g: Graph, question: str) -> str | None:
    """Try template-based SPARQL generation as fallback."""
    for template_fn in TEMPLATE_CHAIN:
        result = template_fn(g, question)
        if result:
            return result
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 7 – SPARQL execution + self-repair
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
# 8 – RAG orchestration
# ══════════════════════════════════════════════════════════════════════════════

def answer_rag(g: Graph, schema: str, question: str, model: str,
               try_repair: bool = TRY_REPAIR) -> dict:
    last_query = ""

    # Step 1: Try LLM-generated SPARQL
    sparql = generate_sparql(question, schema, model)
    last_query = sparql
    try:
        vars_, rows = run_sparql(g, sparql)
        if rows:
            return {"query": sparql, "vars": vars_, "rows": rows,
                    "repaired": False, "error": None, "method": "llm", "hybrid": None}
    except Exception as e:
        if try_repair:
            try:
                repaired = repair_sparql(schema, question, sparql, str(e), model)
                last_query = repaired
                vars_, rows = run_sparql(g, repaired)
                if rows:
                    return {"query": repaired, "vars": vars_, "rows": rows,
                            "repaired": True, "error": None, "method": "llm-repaired", "hybrid": None}
            except Exception:
                pass

    # Step 2: Fallback to template-based SPARQL
    template_sparql = generate_sparql_template(g, question)
    if template_sparql:
        last_query = template_sparql
        try:
            vars_, rows = run_sparql(g, template_sparql)
            if rows:
                return {"query": template_sparql, "vars": vars_, "rows": rows,
                        "repaired": False, "error": None, "method": "template", "hybrid": None}
        except Exception:
            pass

    # Step 3: Hybrid — use KB triples as LLM context
    hybrid_answer = answer_hybrid(g, question, model)
    if hybrid_answer:
        return {"query": last_query, "vars": [], "rows": [],
                "repaired": False, "error": None, "method": "hybrid", "hybrid": hybrid_answer}

    # Step 4: Nothing at all
    return {"query": last_query, "vars": [], "rows": [],
            "repaired": False, "error": "No relevant data found in KB", "method": "none", "hybrid": None}


def _format_triples_as_context(g: Graph, question: str) -> str:
    """Search the KB for entities related to the question and format their triples as text."""
    keywords = _extract_keywords(question)
    context_lines = []
    seen_entities = set()

    for kw in keywords:
        for match in search_entity(g, kw, limit=3):
            uri = match["uri"]
            if uri in seen_entities:
                continue
            seen_entities.add(uri)
            label = match["label"]
            context_lines.append(f"\n## {label} (<{uri}>)")
            # Get all triples where this entity is the subject
            for row in g.query(
                f"SELECT ?p ?o WHERE {{ <{uri}> ?p ?o }} LIMIT 15"
            ):
                p_lbl = _get_label(g, str(row.p)) or uri_to_label(str(row.p))
                o_str = str(row.o)
                o_lbl = _get_label(g, o_str) or uri_to_label(o_str) if o_str.startswith("http") else o_str
                context_lines.append(f"- {p_lbl}: {o_lbl}")
        if len(seen_entities) >= 5:
            break

    return "\n".join(context_lines) if context_lines else ""


def answer_hybrid(g: Graph, question: str, model: str) -> str:
    """Use KB triples as context for the LLM to generate a grounded answer."""
    kb_context = _format_triples_as_context(g, question)
    if not kb_context:
        return ""

    prompt = (
        f"Use the following KNOWLEDGE BASE FACTS to answer the question.\n"
        f"If the facts are relevant, base your answer on them. "
        f"If they don't contain enough information, say what you found and complement with your knowledge.\n\n"
        f"KNOWLEDGE BASE FACTS:\n{kb_context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer concisely:"
    )
    return ask_llm(prompt, model)


def answer_baseline(question: str, model: str) -> str:
    return ask_llm(f"Answer the following question as precisely as possible:\n\n{question}", model)


# ══════════════════════════════════════════════════════════════════════════════
# 9 – Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def pretty_print(result: dict):
    method = result.get("method", "")
    tag = f"({method}) " if method else ""
    repaired = "(repaired) " if result.get("repaired") else ""

    # Hybrid mode: show the KB-grounded LLM answer
    if method == "hybrid":
        print(f"\n  [Method: hybrid — KB context + LLM]")
        if result.get("query"):
            print(f"  [SPARQL attempted (no direct results)]")
            print(f"  {result['query'][:200]}")
        print(f"\n  [Answer grounded on KB facts]")
        print("  " + result["hybrid"].replace("\n", "\n  "))
        return

    print(f"\n  [SPARQL {repaired}{tag}used]")
    print(result["query"])
    if result.get("error"):
        print(f"\n  [ERROR] {result['error']}")
        return
    vars_, rows = result["vars"], result["rows"]
    if not rows:
        print("\n  -> No results returned.")
        return
    header = " | ".join(f"{v:^40}" for v in vars_)
    print(f"\n  {header}")
    print("  " + "-" * len(header))
    for r in rows[:20]:
        print("  " + " | ".join(f"{c:^40}" for c in r))
    if len(rows) > 20:
        print(f"  ... ({len(rows)} rows total, display capped at 20)")


# ══════════════════════════════════════════════════════════════════════════════
# 10 – Evaluation (Baseline vs RAG)
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
# 11 – Interactive CLI
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
                        help="Ollama model name (default: mistral:7b)")
    parser.add_argument("--eval", action="store_true",
                        help="Run automated evaluation then exit")
    args = parser.parse_args()

    nt_path = find_nt_file()
    g       = load_graph(nt_path)
    schema  = build_schema_summary(g)

    print("\n[Schema summary (excerpt)]")
    print(schema[:800])
    print("...\n")

    if args.eval:
        run_evaluation(g, schema, args.model)
    else:
        cli_loop(g, schema, args.model)
