# product-matcher-faiss

Hybrid **BM25 + FAISS HNSW** product matcher for heterogeneous catalogs. Match identical products across retailers that use different naming conventions and have no shared GTIN/UPC — the same problem I solved for a Dutch grocery comparison platform.

## The problem

Two retailers. Both sell the same can of tomatoes. One calls it:

> Heinz Tomato Ketchup 460ML

The other calls it:

> HEINZ KETCHUP - TOMATO (460 ml)

Different strings, same product. Multiply that by 100,000 SKUs across 5 retailers and you have a matching problem that string equality and naive LIKE queries can't solve.

## The approach

**Blocking first, scoring second.** Don't compare every record to every other record (N² is fatal). Instead:

1. **Lexical candidates** — BM25 on the product name pulls the top-K similar-looking candidates
2. **Semantic candidates** — sentence-transformer embeddings + FAISS HNSW index pull the top-K semantically-similar candidates
3. **Merge** — union the two candidate sets
4. **Score** — a combined similarity score ranks the merged set
5. **Accept** — a threshold decides match / no-match (tunable per verticale)

Using both BM25 and embeddings catches two failure modes that either method alone misses:
- BM25 alone misses **rephrasings** ("sparkling water" vs "carbonated mineral water")
- Embeddings alone miss **SKU codes and specs** ("460ml" vs "500ml" — same brand, different product)

## Quickstart

```bash
pip install -r requirements.txt
python -m matcher.demo
```

You'll see the demo index 40 synthetic retailer-A products, then run matching queries from retailer B. Output:

```
Query: HEINZ KETCHUP - TOMATO (460 ml)
  → Heinz Tomato Ketchup 460ML                      (score=0.94) [MATCH]
  → Heinz Ketchup 1000ml                            (score=0.62) [no match]
  → Hellmann's Mayo 450ml                           (score=0.31) [no match]
```

## Stack

- [`sentence-transformers`](https://www.sbert.net/) for embeddings (default: `all-MiniLM-L6-v2`, 384-dim, CPU-friendly)
- [`faiss`](https://github.com/facebookresearch/faiss) for the HNSW ANN index
- [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) for lexical scoring
- Pure Python otherwise

## Project layout

```
product-matcher-faiss/
├── matcher/
│   ├── __init__.py
│   ├── embed.py          # sentence-transformer wrapper
│   ├── index.py          # FAISS HNSW index + ID mapping
│   ├── bm25.py           # BM25 candidate generator
│   ├── hybrid.py         # merge + score
│   └── demo.py           # end-to-end runnable demo
├── fixtures/
│   ├── retailer_a.json
│   └── retailer_b.json
├── tests/
│   └── test_hybrid.py
├── requirements.txt
└── README.md
```

## Tuning notes

- **Embedding model** — `all-MiniLM-L6-v2` is a reasonable default. For non-English catalogs, use `paraphrase-multilingual-MiniLM-L12-v2`. For premium accuracy, `all-mpnet-base-v2` (slower, 768-dim).
- **HNSW parameters** — `M=32, efConstruction=200, efSearch=64` is a solid default for <1M items. Bump `efSearch` to trade latency for recall.
- **BM25 weight** — the hybrid score is `α·cosine + (1-α)·bm25_norm`. Start at `α=0.5`. Raise it for paraphrase-heavy domains (apparel descriptions), lower it for spec-heavy domains (electronics).
- **Threshold** — pick it from a hand-labelled validation set, not a default. Wrong defaults cause silent recall / precision cliffs.

## What this isn't

This isn't a complete production matching system. In production you also need:
- De-duplication within each retailer before cross-matching
- Brand / category gating (don't compare a TV to shampoo)
- Active learning loop to label ambiguous pairs
- Persisted index that survives process restarts
- Monitoring on match-rate drift

I've built all of those for clients; this repo shows the core algorithmic shape.

## Who wrote this

Hamed Daabies — Data Engineer ([Upwork](https://www.upwork.com/freelancers/hameddaabies) · [LinkedIn](https://www.linkedin.com/in/hameddaabies/)).

I build production matching systems for a living. Reach out if you need one.

## License

MIT
