# 🕸️ Web Datamining & Semantics

A four-module pipeline that goes from raw web pages all the way to a RAG chatbot powered by a local LLM answering questions over a Wikidata knowledge graph.

```
Web pages  →  Corpus  →  Knowledge Graph  →  KG Embeddings  →  RAG Chatbot
 (Lab 1)        ↓           (Lab 4)            (Lab 5)           (Lab 6)
           corpus.jsonl   kb_expanse.nt      TransE/DistMult   SPARQL-gen
```

---

## 📁 Repository Structure

```
datamining-semantics/
├── src/
│   ├── lab1_scraping.py        # Web scraping + NER + relation extraction
│   ├── lab4_knowledge_graph.py # RDF graph construction + Wikidata enrichment
│   ├── lab5_embeddings.py      # OWL reasoning + KGE training + visualisation
│   └── lab6_rag_chatbot.py     # RAG chatbot: NL → SPARQL → rdflib → answer
├── data/                       # Generated: train/valid/test splits (git-ignored)
├── models/                     # Generated: embedding .npy files (git-ignored)
├── outputs/                    # Generated: plots (git-ignored)
├── assets/
│   └── screenshots/
│       └── rag_demo.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Installation

### 1 — Clone & create a virtual environment

```bash
git clone https://github.com/your-username/datamining-semantics.git
cd datamining-semantics
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Download spaCy models

```bash
python -m spacy download en_core_web_sm   # used by lab4
python -m spacy download en_core_web_lg   # used by lab1
```

### 4 — Install & start Ollama (required for Lab 6 only)

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama serve                              # keep this terminal open
ollama pull gemma:2b                      # ~1.6 GB
```

> Any model supported by Ollama works. Lighter alternatives: `qwen2.5:0.5b`, `deepseek-r1:1.5b`.

---

## ▶️ How to Run Each Module

### Lab 1 — Web Scraping & Knowledge Extraction

Scrapes Bitcoin/crypto articles, extracts named entities (PERSON, ORG, GPE, DATE) and (subject, verb, object) relations.

```bash
python src/lab1_scraping.py
```

**Outputs:** `corpus.jsonl`, `extracted_knowledge.csv`, `relations.csv`

---

### Lab 4 — Knowledge Graph Construction & Wikidata Enrichment

Builds an RDF graph from scraped text, links entities to Wikidata via `owl:sameAs`, aligns predicates, and expands the graph with 1-hop / 2-hop / property-based Wikidata queries.

```bash
# Full pipeline (scrape → NLP → RDF → Wikidata expansion)
python src/lab4_knowledge_graph.py

# Skip scraping, load existing graphe.ttl and run expansion only
python src/lab4_knowledge_graph.py --expand-only
```

**Outputs:** `graphe.ttl`, `kb_expanse.nt`, `ontologie.ttl`, `alignement.ttl`, `mapping_entites.csv`, `alignement_predicats.csv`, `statistiques_kb.json`

> ⚠️ `kb_expanse.nt` is required by Labs 5 and 6.

---

### Lab 5 — OWL Reasoning & Knowledge Graph Embeddings

Four independent sections, run together or individually.

```bash
# Run all sections
python src/lab5_embeddings.py

# Run a specific section
python src/lab5_embeddings.py --section A   # OWL SWRL reasoning (needs family.owl)
python src/lab5_embeddings.py --section B   # Parse & split kb_expanse.nt
python src/lab5_embeddings.py --section C   # Train TransE + DistMult
python src/lab5_embeddings.py --section D   # Visualisation & analysis
```

| Section | Input | Output |
|---------|-------|--------|
| A | `family.owl` | `family_inferred.owl` |
| B | `kb_expanse.nt` | `data/{train,valid,test}.txt`, `entity2id.txt`, `relation2id.txt` |
| C | `data/*.txt` | `models/{transe,distmult}_{ent,rel}.npy`, `models/results.json` |
| D | `models/*.npy` | `outputs/tsne_entities.png`, `outputs/relation_norms.png`, `outputs/kb_size_sensitivity.png` |

---

### Lab 6 — RAG Chatbot (SPARQL generation)

Requires `kb_expanse.nt` and a running Ollama server.

```bash
# Interactive chatbot
python src/lab6_rag_chatbot.py

# Use a different model
python src/lab6_rag_chatbot.py --model qwen2.5:0.5b

# Run automated 5-question evaluation (Baseline vs RAG) and exit
python src/lab6_rag_chatbot.py --eval
```

---

## 🤖 How to Run the RAG Demo

The RAG demo compares a **Baseline** (plain LLM answer) against a **RAG** answer (LLM generates SPARQL → executed against rdflib → structured results).

```bash
# 1. Make sure Ollama is running
ollama serve &

# 2. Generate the knowledge base (if not already done)
python src/lab4_knowledge_graph.py --expand-only

# 3. Start the chatbot
python src/lab6_rag_chatbot.py
```

**Inside the chatbot:**
- Type any natural-language question about Bitcoin, Ethereum, companies, awards, etc.
- Type `eval` to run the 5-question benchmark
- Type `quit` to exit

**Screenshot — RAG Demo in action:**

![RAG Demo](assets/screenshots/rag_demo.png)

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 5 GB (code + venv) | 10 GB (+ models + KB) |
| CPU | Any modern 4-core | 8-core+ for KGE training |
| GPU | Not required | CUDA GPU speeds up t-SNE & KGE |
| OS | Linux / macOS / Windows (WSL) | Ubuntu 22.04+ |

### Notes
- **Lab 4 (Wikidata expansion)** makes many HTTP requests to `query.wikidata.org`. With the default settings (`P31, P17, P166, P452, P159, P355` at 10 000 triples each) expect ~10 min and a `kb_expanse.nt` of ~50–200 MB.
- **Lab 5 Section C (KGE training)** trains TransE and DistMult for 50 epochs on ~80 k triples. On a modern laptop CPU this takes ~5–15 min per model.
- **Lab 6 (Ollama)** requires ~2 GB RAM for `gemma:2b`. Using `qwen2.5:0.5b` reduces this to ~1 GB.

---

## 🔄 End-to-End Pipeline

```bash
# Step 1 — build corpus
python src/lab1_scraping.py

# Step 2 — build & expand the knowledge graph
python src/lab4_knowledge_graph.py

# Step 3 — train embeddings (sections B → C → D)
python src/lab5_embeddings.py

# Step 4 — run the RAG chatbot
python src/lab6_rag_chatbot.py
```

---

## 📚 Key Dependencies

| Library | Version | Role |
|---------|---------|------|
| `trafilatura` | ≥1.7 | Web scraping |
| `spacy` | ≥3.7 | Named entity recognition |
| `rdflib` | ≥7.0 | RDF graph storage & SPARQL |
| `SPARQLWrapper` | ≥2.0 | Wikidata queries |
| `owlready2` | ≥0.46 | OWL ontology + Pellet reasoner |
| `numpy` | ≥1.26 | KGE model implementation |
| `scikit-learn` | ≥1.4 | t-SNE visualisation |
| `requests` | ≥2.31 | Ollama API calls |

---

## 📝 License

MIT
