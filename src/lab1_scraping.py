"""
Lab 1 – Web Scraping & Knowledge Extraction Pipeline
=====================================================
Part I  : Web scraping → corpus.jsonl
Part II : Named entity extraction + relation extraction → extracted_knowledge.csv, relations.csv

Usage:
    python src/lab1_scraping.py
"""

import json
import csv
import trafilatura
import spacy
import en_core_web_sm

# ── Configuration ──────────────────────────────────────────────────────────────
MIN_WORDS   = 500
CORPUS_FILE = "corpus.jsonl"
OUTPUT_CSV  = "extracted_knowledge.csv"
RELATIONS_CSV = "relations.csv"

ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "DATE"}

BLACKLIST = {
    "bitcoin blockchain",
    "blockchain",
    "cryptocurrency",
    "miners",
}

SEED_URLS = [
    "https://www.gemini.com/en-GB/cryptopedia/what-is-bitcoin",
    "https://bitcoinmagazine.com/guides/what-is-bitcoin",
    "https://www.ledger.com/academy/what-is-bitcoin",
    "https://bitcoin.org/en/secure-your-wallet#backup",
    "https://www.investopedia.com/terms/b/bitcoin.asp",
    "https://ethereum.org/developers/docs/consensus-mechanisms/pow/",
    "https://river.com/learn/what-is-bitcoin/",
]


# ══════════════════════════════════════════════════════════════════════════════
# PART I – Scraping
# ══════════════════════════════════════════════════════════════════════════════

def fetch_html(url: str) -> str | None:
    """Download raw HTML from a URL via trafilatura."""
    try:
        return trafilatura.fetch_url(url)
    except Exception as e:
        print(f"[ERROR] Fetch failed for {url}: {e}")
        return None


def extract_main_text(html: str) -> str | None:
    """Extract main text content from raw HTML."""
    return trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=True,
    )


def is_useful(text: str, min_words: int = MIN_WORDS) -> bool:
    """Return True if the text contains at least min_words words."""
    if not text:
        return False
    return len(text.split()) >= min_words


def save_to_jsonl(record: dict, filepath: str) -> None:
    """Append a JSON record to a JSONL file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_url(url: str) -> None:
    """Scrape, filter, and save one URL."""
    html = fetch_html(url)
    if not html:
        return

    text = extract_main_text(html)
    if not is_useful(text):
        print(f"[SKIP] Not enough content: {url}")
        return

    record = {
        "url":        url,
        "word_count": len(text.split()),
        "text":       text,
    }
    save_to_jsonl(record, CORPUS_FILE)
    print(f"[OK]   Saved: {url}")


def run_scraping_pipeline(urls: list[str]) -> None:
    """Run the scraping pipeline for all URLs."""
    for url in urls:
        process_url(url)


# ══════════════════════════════════════════════════════════════════════════════
# PART II – Corpus reading
# ══════════════════════════════════════════════════════════════════════════════

def read_jsonl(file_path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                print("[WARNING] Invalid JSON line — skipped")
    return documents


def display_documents(documents: list[dict], show_text: bool = True, max_chars: int = 500) -> None:
    """Print each document with its URL and a text preview."""
    for i, doc in enumerate(documents, 1):
        print(f"\n--- Document {i} ---")
        print(f"URL:        {doc.get('url')}")
        print(f"Word count: {doc.get('word_count')}")
        if show_text:
            text    = doc.get("text", "")
            preview = text[:max_chars] + ("..." if len(text) > max_chars else "")
            print(f"Text preview:\n{preview}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART II – Entity extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities(documents: list[dict], nlp) -> list[dict]:
    """Extract filtered named entities from all documents."""
    results = []
    for doc_data in documents:
        url = doc_data["url"]
        doc = nlp(doc_data["text"])
        for sent in doc.sents:
            for ent in sent.ents:
                if ent.label_ in ALLOWED_LABELS:
                    if len(ent.text) > 2 and not ent.text.islower():
                        results.append({
                            "entity":     ent.text,
                            "label":      ent.label_,
                            "sentence":   sent.text.strip(),
                            "source_url": url,
                        })
    return results


def clean_entities(entities: list[dict]) -> list[dict]:
    """Remove entities that appear in the blacklist."""
    return [e for e in entities if e["entity"].lower() not in BLACKLIST]


# ══════════════════════════════════════════════════════════════════════════════
# PART II – Relation extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_relations(documents: list[dict], nlp) -> list[dict]:
    """Extract (subject, verb, object) triples from co-occurring entities."""
    relations = []
    for doc_data in documents:
        url = doc_data["url"]
        doc = nlp(doc_data["text"])
        for sent in doc.sents:
            ents  = [ent for ent in sent.ents if ent.label_ in ALLOWED_LABELS]
            verbs = [token for token in sent if token.pos_ == "VERB"]
            if len(ents) >= 2 and verbs:
                verb = verbs[0].lemma_
                for i in range(len(ents) - 1):
                    relations.append({
                        "subject":    ents[i].text,
                        "relation":   verb,
                        "object":     ents[i + 1].text,
                        "source_url": url,
                    })
    return relations


# ══════════════════════════════════════════════════════════════════════════════
# PART II – CSV export
# ══════════════════════════════════════════════════════════════════════════════

def save_to_csv(data: list[dict], path: str, fieldnames: list[str]) -> None:
    """Save a list of dicts to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"[OK]   Saved: {path} ({len(data)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- Part I: scraping ---
    print("=" * 55)
    print("  Part I : Web Scraping")
    print("=" * 55)
    run_scraping_pipeline(SEED_URLS)

    # --- Read corpus ---
    print("\n" + "=" * 55)
    print("  Reading corpus")
    print("=" * 55)
    documents = read_jsonl(CORPUS_FILE)
    if not documents:
        print("[ERROR] corpus.jsonl is empty — check that scraping succeeded.")
        exit(1)
    display_documents(documents, show_text=True, max_chars=1000)

    # --- Part II: NLP ---
    print("\n" + "=" * 55)
    print("  Part II : Entity & Relation Extraction")
    print("=" * 55)
    nlp = spacy.load("en_core_web_lg")

    entities = extract_entities(documents, nlp)
    entities = clean_entities(entities)
    print(f"  {len(entities)} entities extracted after cleaning.")

    relations = extract_relations(documents, nlp)
    print(f"  {len(relations)} relations extracted.")
    if relations:
        print("  Sample:", relations[:3])

    # --- Save ---
    save_to_csv(entities, OUTPUT_CSV,   ["entity", "label", "sentence", "source_url"])
    save_to_csv(relations, RELATIONS_CSV, ["subject", "relation", "object", "source_url"])
    print(f"\n[DONE] Files generated: {OUTPUT_CSV}, {RELATIONS_CSV}")
