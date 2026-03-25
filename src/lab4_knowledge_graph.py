"""
Lab 4 – Knowledge Graph Construction & Wikidata Enrichment
===========================================================
Step 1 : Scrape URLs → extract NLP triples → RDF graph (graphe.ttl)
Step 2 : Entity linking (owl:sameAs → Wikidata)
Step 3 : Predicate alignment (owl:equivalentProperty)
Step 4 : 1-hop / 2-hop / property-based expansion → kb_expanse.nt

Usage:
    # Run everything end-to-end
    python src/lab4_knowledge_graph.py

    # Or just the expansion stage (requires graphe.ttl to exist)
    python src/lab4_knowledge_graph.py --expand-only
"""

import argparse
import json
import time
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from SPARQLWrapper import JSON, SPARQLWrapper

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_NS   = "http://lab_web4.org/entity/"
BASE_PRED = "http://lab_web4.org/predicate/"
NS        = Namespace(BASE_NS)
PRED      = Namespace(BASE_PRED)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "KBLabBot/1.0 (student project; contact@example.com)"}

NLP_MODEL = "en_core_web_sm"

URLS = [
    "https://www.gemini.com/en-GB/cryptopedia/what-is-bitcoin",
    "https://bitcoinmagazine.com/guides/what-is-bitcoin",
    "https://www.ledger.com/academy/what-is-bitcoin",
    "https://bitcoin.org/en/secure-your-wallet#backup",
    "https://www.investopedia.com/terms/b/bitcoin.asp",
    "https://ethereum.org/developers/docs/consensus-mechanisms/pow/",
    "https://river.com/learn/what-is-bitcoin/",
    "https://en.wikipedia.org/wiki/Bitcoin",
]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Web scraping + triple extraction
# ══════════════════════════════════════════════════════════════════════════════

def fetch_text(url: str) -> str:
    """Download and clean the body text of a URL."""
    print(f"  → Fetching: {url}")
    time.sleep(1.0)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer":        "https://www.google.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.Session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())
    except requests.HTTPError as e:
        code = e.response.status_code if e.response else 0
        print(f"  ✗ HTTP {code} — skipped")
        return ""
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return ""


def extract_triples(text: str, nlp) -> list[dict]:
    """Extract (subject, verb, object) triples via spaCy dependency parsing."""
    doc = nlp(text[:50_000])  # cap to avoid memory issues
    triples = []
    for sent in doc.sents:
        ents  = [e for e in sent.ents if e.label_ in {"PERSON", "ORG", "GPE"}]
        verbs = [t for t in sent if t.pos_ == "VERB"]
        if len(ents) >= 2 and verbs:
            verb = verbs[0].lemma_
            for i in range(len(ents) - 1):
                triples.append({
                    "sujet":    ents[i].text,
                    "relation": verb,
                    "objet":    ents[i + 1].text,
                })
    return triples


def make_uri(text: str) -> URIRef:
    clean = text.strip().replace(" ", "_")
    return URIRef(BASE_NS + quote(clean))


def build_rdf_graph(triples: list[dict]) -> Graph:
    g = Graph()
    g.bind("lab", NS)
    for t in triples:
        g.add((make_uri(t["sujet"]), make_uri(t["relation"]), make_uri(t["objet"])))
    print(f"  → {len(g)} triples added to RDF graph")
    return g


def save_files(triples: list[dict], graph: Graph):
    df = pd.DataFrame(triples).drop_duplicates()
    df.to_csv("triplets.csv", index=False, encoding="utf-8")
    print(f"  ✓ triplets.csv  ({len(df)} unique triples)")

    uniq = list({(t["sujet"], t["relation"], t["objet"]): t for t in triples}.values())
    with open("triplets.json", "w", encoding="utf-8") as f:
        json.dump(uniq, f, ensure_ascii=False, indent=2)
    print("  ✓ triplets.json")

    graph.serialize(destination="graphe.ttl", format="turtle")
    print("  ✓ graphe.ttl")


def run_sparql_examples(graph: Graph):
    from rdflib.plugins.sparql import prepareQuery
    print("\n── SPARQL sample (first 10 triples) ──")
    q = prepareQuery("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10")
    for row in graph.query(q):
        s = str(row.s).replace(BASE_NS, "")
        p = str(row.p).replace(BASE_NS, "")
        o = str(row.o).replace(BASE_NS, "")
        print(f"   {s}  →[{p}]→  {o}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Entity linking
# ══════════════════════════════════════════════════════════════════════════════

def search_wikidata(label: str) -> list[dict]:
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json",
        "limit": 5,
    }
    try:
        r = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=10)
        results = []
        for i, item in enumerate(r.json().get("search", [])):
            exact = item.get("label", "").lower() == label.lower()
            confidence = round(0.99 if exact else max(0.5, 0.9 - i * 0.1), 2)
            results.append({
                "label":       item.get("label"),
                "description": item.get("description", ""),
                "wikidata_id": item.get("id"),
                "wikidata_uri": f"http://www.wikidata.org/entity/{item.get('id')}",
                "confidence":  confidence,
            })
        return results
    except Exception as e:
        print(f"  ✗ Wikidata API error for '{label}': {e}")
        return []


def link_entities(graph: Graph, entities: list[str]) -> tuple[Graph, pd.DataFrame]:
    print("\n── Step 2: Entity Linking ──")
    mappings = []
    for entity in entities:
        print(f"  Searching: {entity}")
        ent_uri = URIRef(BASE_NS + quote(entity.replace(" ", "_")))
        candidates = search_wikidata(entity)
        if candidates:
            best = candidates[0]
            graph.add((ent_uri, OWL.sameAs, URIRef(best["wikidata_uri"])))
            mappings.append({
                "Private Entity":  f":{entity.replace(' ', '_')}",
                "External URI":    best["wikidata_uri"],
                "Wikidata ID":     best["wikidata_id"],
                "Description":     best["description"],
                "Confidence":      best["confidence"],
                "Status":          "aligned",
            })
            print(f"    ✓ → {best['wikidata_id']} (confidence={best['confidence']})")
        else:
            graph.add((ent_uri, RDF.type, NS.CustomEntity))
            graph.add((ent_uri, RDFS.label, Literal(entity, lang="en")))
            graph.add((NS.CustomEntity, RDFS.subClassOf, NS.Entity))
            mappings.append({
                "Private Entity":  f":{entity.replace(' ', '_')}",
                "External URI":    "N/A",
                "Wikidata ID":     "N/A",
                "Description":     "Not found — defined locally",
                "Confidence":      0.0,
                "Status":          "local",
            })
            print(f"    ✗ Not found — defined locally")
        time.sleep(0.5)
    return graph, pd.DataFrame(mappings)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Predicate alignment
# ══════════════════════════════════════════════════════════════════════════════

def align_predicate(keyword: str) -> list[dict]:
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.addCustomHttpHeader("User-Agent", HEADERS["User-Agent"])
    query = f"""
    SELECT ?property ?propertyLabel WHERE {{
        ?property a wikibase:Property .
        ?property rdfs:label ?propertyLabel .
        FILTER(CONTAINS(LCASE(?propertyLabel), "{keyword.lower()}"))
        FILTER(LANG(?propertyLabel) = "en")
    }}
    LIMIT 10
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return [
            {
                "property_uri":   r["property"]["value"],
                "property_label": r["propertyLabel"]["value"],
            }
            for r in results["results"]["bindings"]
        ]
    except Exception as e:
        print(f"  ✗ SPARQL error for '{keyword}': {e}")
        return []


def align_predicates(predicates: dict) -> tuple[Graph, pd.DataFrame]:
    """predicates: { local_name: search_keyword }"""
    print("\n── Step 3: Predicate Alignment ──")
    g_align = Graph()
    g_align.bind("owl", OWL)
    rows = []
    for pred, keyword in predicates.items():
        print(f"  Searching '{pred}' (keyword: '{keyword}')…")
        pred_uri   = URIRef(BASE_PRED + pred)
        candidates = align_predicate(keyword)
        if candidates:
            best = candidates[0]
            g_align.add((pred_uri, OWL.equivalentProperty, URIRef(best["property_uri"])))
            rows.append({
                "Private Predicate": f":{pred}",
                "Wikidata Property": best["property_uri"].split("/")[-1],
                "Wikidata Label":    best["property_label"],
                "Relation":         "owl:equivalentProperty",
                "Validate":         "manual check needed",
            })
            print(f"    ✓ → {best['property_uri'].split('/')[-1]}: {best['property_label']}")
        else:
            rows.append({
                "Private Predicate": f":{pred}",
                "Wikidata Property": "N/A",
                "Wikidata Label":    "Not found",
                "Relation":         "N/A",
                "Validate":         "—",
            })
        time.sleep(1)
    return g_align, pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Graph expansion
# ══════════════════════════════════════════════════════════════════════════════

def expand_1hop(wikidata_ids: list[str], limit: int = 1000) -> list[tuple]:
    print("\n── Step 4a: 1-hop Expansion ──")
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.addCustomHttpHeader("User-Agent", HEADERS["User-Agent"])
    sparql.setTimeout(60)
    all_triples = []
    for wd_id in wikidata_ids:
        print(f"  Expanding {wd_id}…")
        query = f"""
        SELECT ?p ?o WHERE {{
            wd:{wd_id} ?p ?o .
            FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
            FILTER(!isLiteral(?o) || LANG(?o) = "" || LANG(?o) = "en")
        }}
        LIMIT {limit}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            for r in results["results"]["bindings"]:
                s = f"http://www.wikidata.org/entity/{wd_id}"
                all_triples.append((s, r["p"]["value"], r["o"]["value"]))
            print(f"    ✓ {len(results['results']['bindings'])} triples retrieved")
        except Exception as e:
            print(f"    ✗ Error for {wd_id}: {e}")
        time.sleep(1)
    return all_triples


def expand_by_property(properties: list[str], limit: int = 10000) -> list[tuple]:
    print("\n── Step 4b: Property-based Expansion ──")
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.addCustomHttpHeader("User-Agent", HEADERS["User-Agent"])
    sparql.setTimeout(60)
    all_triples = []
    for prop in properties:
        print(f"  Expanding via {prop}…")
        query = f"""
        SELECT ?s ?o WHERE {{
            ?s wdt:{prop} ?o .
        }}
        LIMIT {limit}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            for r in results["results"]["bindings"]:
                p = f"http://www.wikidata.org/prop/direct/{prop}"
                all_triples.append((r["s"]["value"], p, r["o"]["value"]))
            print(f"    ✓ {len(results['results']['bindings'])} triples retrieved")
        except Exception as e:
            print(f"    ✗ Error for {prop}: {e}")
        time.sleep(2)
    return all_triples


def expand_2hop(wikidata_ids: list[str], property_id: str, limit: int = 5000) -> list[tuple]:
    print(f"\n── Step 4c: 2-hop Expansion via {property_id} ──")
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.addCustomHttpHeader("User-Agent", HEADERS["User-Agent"])
    sparql.setTimeout(60)
    ids_str = " ".join(f"wd:{i}" for i in wikidata_ids)
    query = f"""
    SELECT ?intermediaire ?p ?o WHERE {{
        VALUES ?personne {{ {ids_str} }}
        ?personne wdt:{property_id} ?intermediaire .
        ?intermediaire ?p ?o .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
    }}
    LIMIT {limit}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    all_triples = []
    try:
        results = sparql.query().convert()
        for r in results["results"]["bindings"]:
            all_triples.append((r["intermediaire"]["value"], r["p"]["value"], r["o"]["value"]))
        print(f"  ✓ {len(all_triples)} 2-hop triples retrieved")
    except Exception as e:
        print(f"  ✗ 2-hop error: {e}")
    return all_triples


def merge_graph(private_graph: Graph, expansion_triples: list[tuple]) -> tuple[Graph, dict]:
    print("\n── Merging and deduplicating graph ──")
    g = Graph()
    for triple in private_graph:
        g.add(triple)
    for s, p, o in expansion_triples:
        try:
            obj = URIRef(o) if o.startswith("http") else Literal(o)
            g.add((URIRef(s), URIRef(p), obj))
        except Exception:
            pass
    entities  = set(str(s) for s, _, _ in g) | set(str(o) for _, _, o in g if isinstance(o, URIRef))
    relations = set(str(p) for _, p, _ in g)
    stats = {"triples": len(g), "entities": len(entities), "relations": len(relations)}
    print(f"  Triples:   {stats['triples']:,}")
    print(f"  Entities:  {stats['entities']:,}")
    print(f"  Relations: {stats['relations']}")
    return g, stats


def save_deliverables(
    expanded_graph: Graph,
    ontology_graph: Graph,
    alignment_graph: Graph,
    df_mapping: pd.DataFrame,
    df_predicates: pd.DataFrame,
    stats: dict,
):
    print("\n── Saving deliverables ──")
    expanded_graph.serialize("kb_expanse.nt",     format="nt");      print("  ✓ kb_expanse.nt")
    ontology_graph.serialize("ontologie.ttl",     format="turtle");  print("  ✓ ontologie.ttl")
    alignment_graph.serialize("alignement.ttl",   format="turtle");  print("  ✓ alignement.ttl")
    df_mapping.to_csv("mapping_entites.csv",      index=False);      print("  ✓ mapping_entites.csv")
    df_predicates.to_csv("alignement_predicats.csv", index=False);   print("  ✓ alignement_predicats.csv")
    with open("statistiques_kb.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("  ✓ statistiques_kb.json")
    print(f"\n{'='*50}\n  FINAL STATISTICS\n{'='*50}")
    for k, v in stats.items():
        print(f"  {k:<20} : {v:,}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

ENTITIES_TO_LINK = [
    "Bitcoin", "Ethereum", "Satoshi Nakamoto", "Jack Dorsey", "Elon Musk",
    "Adam Back", "Joseph Stiglitz", "Microsoft", "University of Cambridge",
    "Bank of England", "Bank of China", "Visa",
]

PREDICATES_TO_ALIGN = {
    "avoir":      "has part",
    "consommer":  "energy consumption",
    "permettre":  "use",
    "encourager": "promotes",
    "supply":     "total produced",
    "blockchain": "blockchain",
    "node":       "instance of",
}

EXPANSION_PROPERTIES = ["P31", "P17", "P571", "P856", "P159", "P452", "P166", "P1346", "P355"]


def main(expand_only: bool = False):
    print("=" * 55)
    print("  Lab 4 – Knowledge Graph Construction")
    print("=" * 55)

    # ── Step 1: scrape + NLP ──
    if not expand_only:
        print("\n[1/4] Loading spaCy model…")
        import spacy
        try:
            nlp = spacy.load(NLP_MODEL)
        except OSError:
            print(f"  ✗ Model not found. Run: python -m spacy download {NLP_MODEL}")
            return

        print("\n[2/4] Extracting triples from URLs…")
        all_triples = []
        for url in URLS:
            text = fetch_text(url)
            if text:
                triples = extract_triples(text, nlp)
                print(f"      {len(triples)} triples extracted")
                all_triples.extend(triples)

        print(f"\n  Total: {len(all_triples)} triples")
        if not all_triples:
            print("  ✗ No triples found.")
            return

        print("\n[3/4] Building RDF graph…")
        g_private = build_rdf_graph(all_triples)

        print("\n[4/4] Saving files…")
        save_files(all_triples, g_private)
        run_sparql_examples(g_private)
    else:
        g_private = Graph()
        try:
            g_private.parse("graphe.ttl", format="turtle")
            print(f"\n✓ Private graph loaded: {len(g_private)} triples")
        except FileNotFoundError:
            print("\n⚠ graphe.ttl not found — starting with empty graph")
            g_private.add((URIRef(BASE_NS + "Bitcoin"), URIRef(BASE_PRED + "above"), URIRef(BASE_NS + "Trump")))
            g_private.add((URIRef(BASE_NS + "Satoshi"), URIRef(BASE_PRED + "explain"), URIRef(BASE_NS + "Bitcoin")))

    # ── Step 2: Entity linking ──
    g_private, df_mapping = link_entities(g_private, ENTITIES_TO_LINK)
    print(f"\nEntity mapping table:\n{df_mapping.to_string(index=False)}")

    # ── Step 3: Predicate alignment ──
    g_alignment, df_predicates = align_predicates(PREDICATES_TO_ALIGN)
    print(f"\nPredicate alignment:\n{df_predicates.to_string(index=False)}")

    # ── Step 4: Graph expansion ──
    wd_ids = df_mapping[df_mapping["Confidence"] > 0.8]["Wikidata ID"].tolist()
    wd_ids = [i for i in wd_ids if i != "N/A"]
    print(f"\n  {len(wd_ids)} aligned entities for expansion")

    expansion = []
    if wd_ids:
        expansion += expand_1hop(wd_ids)
    expansion += expand_by_property(EXPANSION_PROPERTIES)
    if wd_ids:
        expansion += expand_2hop(wd_ids, "P166")

    # ── Merge ──
    g_expanded, stats = merge_graph(g_private, expansion)

    # ── Minimal ontology ──
    g_ontology = Graph()
    g_ontology.add((NS.Entity,        RDF.type,         OWL.Class))
    g_ontology.add((NS.CustomEntity,  RDFS.subClassOf,  NS.Entity))
    g_ontology.add((PRED.wonAward,    RDF.type,         OWL.ObjectProperty))
    g_ontology.add((PRED.wonAward,    RDFS.domain,      NS.Person))
    g_ontology.add((PRED.wonAward,    RDFS.range,       NS.Award))

    # ── Save ──
    save_deliverables(g_expanded, g_ontology, g_alignment, df_mapping, df_predicates, stats)
    print("\n[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expand-only", action="store_true",
                        help="Skip scraping/NLP, load graphe.ttl and run expansion only")
    args = parser.parse_args()
    main(expand_only=args.expand_only)
