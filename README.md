# Web Datamining & Semantics — Knowledge Graph Pipeline

A four-module pipeline: web scraping, knowledge graph construction, KG embeddings, and a RAG chatbot powered by a local LLM.

```
Web pages  →  Corpus  →  Knowledge Graph  →  KG Embeddings  →  RAG Chatbot
 (Lab 1)        ↓           (Lab 4)            (Lab 5)           (Lab 6)
           crawler_output   kb_expanse.nt      TransE/DistMult   SPARQL-gen
              .jsonl                                              + hybrid
```

---

## Repository Structure

```
project-root/
├── src/
│   ├── lab1_scraping.py        # Web scraping + NER + relation extraction
│   ├── lab4_knowledge_graph.py # RDF graph + ontology + alignment + expansion
│   ├── lab5_embeddings.py      # SWRL reasoning + KGE training + analysis
│   └── lab6_rag_chatbot.py     # RAG chatbot: NL → SPARQL → rdflib → answer
├── kg_artifacts/               # Deliverables: ontologie.ttl, alignement.ttl, kb_expanse.nt
├── data/                       # Generated: train/valid/test splits
├── models/                     # Generated: embedding .npy files
├── outputs/                    # Generated: CSVs, plots, crawler_output.jsonl
├── family.owl                  # OWL ontology for SWRL reasoning (Lab 5 Section A)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1 — Clone & create a virtual environment

```bash
git clone <repo-url>
cd Project_web_datamining_leclair
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Download spaCy models

```bash
python -m spacy download en_core_web_trf   # recommended (transformer-based)
python -m spacy download en_core_web_lg    # fallback
python -m spacy download en_core_web_sm    # used by lab4
```

### 4 — Install & start Ollama (required for Lab 6)

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama serve                              # keep this terminal open
ollama pull mistral:7b                    # recommended model (~4 GB)
```

> Alternative smaller models: `gemma:2b`, `qwen2.5:0.5b`, `deepseek-r1:1.5b`.

---

## How to Run Each Module

### Lab 1 — Web Scraping & Knowledge Extraction

Scrapes Bitcoin/crypto articles (7 seed URLs), checks `robots.txt`, extracts named entities (PERSON, ORG, GPE, DATE) and (subject, verb, object) triples via dependency parsing.

```bash
python src/lab1_scraping.py
```

**Outputs:** `outputs/crawler_output.jsonl`, `outputs/extracted_knowledge.csv`, `outputs/relations.csv`

---

### Lab 4 — Knowledge Graph Construction & Wikidata Enrichment

Builds an RDF graph from scraped text, links entities to Wikidata (`owl:sameAs` with confidence scores), aligns predicates (`owl:equivalentProperty`), and expands via 1-hop / 2-hop / property-based SPARQL queries.

```bash
# Full pipeline (scrape → NLP → RDF → Wikidata expansion)
python src/lab4_knowledge_graph.py

# Skip scraping, load existing graphe.ttl and run expansion only
python src/lab4_knowledge_graph.py --expand-only
```

**Outputs:** `kg_artifacts/{kb_expanse.nt, ontologie.ttl, alignement.ttl, graphe.ttl, statistiques_kb.json}`, `outputs/{mapping_entites.csv, alignement_predicats.csv}`

> `kb_expanse.nt` is required by Labs 5 and 6. Target: 50k–200k triples.

---

### Lab 5 — OWL Reasoning & Knowledge Graph Embeddings

Five sections, run together or individually.

```bash
python src/lab5_embeddings.py                # all sections
python src/lab5_embeddings.py --section A    # SWRL reasoning (needs family.owl)
python src/lab5_embeddings.py --section B    # Parse & split kb_expanse.nt
python src/lab5_embeddings.py --section C    # Train TransE + DistMult
python src/lab5_embeddings.py --section D    # Visualisation & analysis
python src/lab5_embeddings.py --section E    # Relation behavior + rule-vs-embedding comparison
```

| Section | What | Input | Output |
|---------|------|-------|--------|
| A | SWRL rule on family.owl | `family.owl` | `family_inferred.owl` |
| B | Parse & split KB | `kb_expanse.nt` | `data/{train,valid,test}.txt` |
| C | Train TransE + DistMult | `data/*.txt` | `models/{transe,distmult}_{ent,rel}.npy` |
| D | t-SNE, nearest neighbors, KB-size sensitivity | `models/*.npy` | `outputs/{tsne_entities,relation_norms,kb_size_sensitivity}.png` |
| E | Relation behavior + SWRL on own KB + rule-vs-embedding | `models/*.npy` | terminal output (analysis) |

---

### Lab 6 — RAG Chatbot (SPARQL generation)

Requires `kb_expanse.nt` (from Lab 4) and a running Ollama server.

```bash
# Interactive chatbot (default: mistral:7b)
python src/lab6_rag_chatbot.py

# Use a different model
python src/lab6_rag_chatbot.py --model gemma:2b

# Run automated 7-question evaluation (Baseline vs RAG) and exit
python src/lab6_rag_chatbot.py --eval
```

The chatbot uses a 4-step cascade: LLM SPARQL → self-repair → template fallback → hybrid (KB context + LLM).

---

## How to Run the RAG Demo

```bash
# 1. Make sure Ollama is running
ollama serve &

# 2. Generate the knowledge base (if not already done)
python src/lab4_knowledge_graph.py --expand-only

# 3. Start the chatbot
python src/lab6_rag_chatbot.py
```

**Inside the chatbot:**
- Type any natural-language question (e.g., "What is Bitcoin?", "Who received an award?")
- Type `eval` to run the 7-question benchmark (Baseline vs RAG comparison table)
- Type `quit` to exit

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 5 GB (code + venv) | 10 GB (+ models + KB) |
| CPU | Any modern 4-core | 8-core+ for KGE training |
| GPU | Not required | CUDA GPU speeds up t-SNE |
| OS | Linux / macOS / Windows (WSL) | macOS or Ubuntu 22.04+ |

### Notes
- **Lab 4** makes many HTTP requests to `query.wikidata.org` — expect ~10 min.
- **Lab 5 Section C** trains TransE and DistMult for 50 epochs — ~5–15 min per model on CPU.
- **Lab 6 (Ollama)** requires ~4 GB RAM for `mistral:7b`. Use `gemma:2b` (~2 GB) for lighter setup.

---

## End-to-End Pipeline

```bash
python src/lab1_scraping.py                       # Step 1: build corpus
python src/lab4_knowledge_graph.py                # Step 2: build & expand KG
python src/lab5_embeddings.py                     # Step 3: reasoning + embeddings
python src/lab6_rag_chatbot.py --eval             # Step 4: RAG evaluation
```

---

## Key Dependencies

| Library | Role |
|---------|------|
| `trafilatura` | Web scraping & boilerplate removal |
| `spacy` | NER & dependency parsing |
| `rdflib` | RDF graph storage & SPARQL |
| `SPARQLWrapper` | Wikidata SPARQL queries |
| `owlready2` | OWL ontology + SWRL reasoning |
| `numpy` | KGE model implementation |
| `scikit-learn` | t-SNE visualisation |
| `matplotlib` | Plots |
| `requests` | Ollama API calls |
