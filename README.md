# Seed & Seek (remastered)

Data augmentation from a seed file. You give it a small **seed CSV**; it finds other tables that
describe the same entities, picks the ones that fulfil what the seed needs, and uses them to
**add new columns to the seed**. Every added value comes from stored source tables, not from a
language model, so the values are true.

## The idea

Two stores hold the same tables, 1:1:

| Store | Holds | Role |
|---|---|---|
| **DuckDB** (`data/seedandseek.duckdb`) | the real, cleaned table data | source of truth for values |
| **ChromaDB** (`data/chroma_seed_seek`) | embeddings of natural-language descriptions of those tables and columns | search index |

Each Chroma card carries a `table_id` (and `column_name` for column cards) that maps straight to a
DuckDB table (`t_<table_id>`, see `db_table_name` in `preprocessing/dbms_store.py`). Search happens
in Chroma, but the values you end up with are always read from DuckDB, so retrieval can be fuzzy
while the augmented data stays exact.

```
                         ┌──────────────────────────── INDEX SIDE (built once) ───────────────────────────┐
  Wikipedia pages        │                                                                                 │
        │ scrape.py      │   data/imputed/*.csv ──► dbms_store.py ──► DuckDB  (true values)               │
        ▼                │          │                                      ▲                              │
  data/tables ─ clean.py ─► data/clean ─ quality.py ─► data/imputed        │ 1:1 by table_id              │
                         │          │                                      │                              │
                         │          └─ NL descriptions of tables/columns ─► ChromaDB                      │
                         │             (table_cards, column_cards)         (embeddings)                   │
                         └─────────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────────── QUERY SIDE (per seed file) ────────────────────────┐
  seed CSV ─ profiler.py ─► stats ─ query.py / querygroq.py ─► one NL query per seed column               │
                                         │                                                                │
                                         ▼                                                                │
                         equationextractorforembedding.py   (belongs_to / subset_of relations)           │
                                         │                                                                │
  queries q1..qN ───────► retrieve_chroma.py ──► score matrix  (queries x tables, 0-100)                 │
                                         │                                                                │
                                         ▼                                                                │
                              select_tables.py ──► ranked table combinations                             │
                                         │          (every query fulfilled, best average first)          │
                                         ▼                                                                │
                              augmentation (DuckDB)  ──► seed + new columns        [not built yet]       │
                         └─────────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline

### 1. Build the corpus (once)

Run from the project root; paths are relative to it.

| Step | Script | In → Out |
|---|---|---|
| Scrape Wikipedia tables | `preprocessing/scrape.py` | URLs → `data/raw`, `data/tables`, `tables_metadata.json` |
| Clean (units, multipliers, types) | `preprocessing/clean.py` | `data/tables` → `data/clean`, `clean_metadata.json` |
| Quality: TabPFN imputation + outlier scores | `preprocessing/quality.py` | `data/clean` → `data/imputed`, `imputed_metadata.json`, `outliers.json` |
| Store | `preprocessing/dbms_store.py` | `data/imputed` → `data/seedandseek.duckdb` (`table_registry`, `column_registry`, one `t_*` table per source table) |
| Describe + embed into Chroma | `index_chroma.py` | DuckDB → `data/chroma_seed_seek` (`table_cards`, `column_cards`) — **not in the repo yet** |

The Chroma collections must record `embedding_model` in their metadata and use cosine distance
(`hnsw:space = cosine`); `retrieve_chroma.py` refuses to run otherwise, so queries are always
embedded with the same model as the index.

### 2. Turn the seed into queries

```
python profiler.py data/<seed>.csv                     # -> output/profiler_output.json
python querygroq.py                                    # -> output/column_queries_groq.json   (needs GROQ_API_KEY)
python query.py                                        # same, with Gemini                    (needs GEMINI_API_KEY)
python equationextractorforembedding.py                # -> output/column_relationship_equations.json
```

- `profiler.py` computes per-column stats (numeric min/max/avg, TF-IDF terms for text, boolean counts, sample rows).
- `querygroq.py` / `query.py` ask an LLM to describe each column and write one natural-language
  query for it. They also record which entity owns the column (`belongs_to`) and whether it is a
  subset of or derived from another column (`subset_of`, `is_derived`). `--non-derived-output`
  writes a file with only the columns that are not derived.
- `equationextractorforembedding.py` turns those ownership and subset links into short predicates
  that are easy to embed.

Put API keys in a `.env` file next to the scripts (it is git-ignored).

### 3. Match queries against the index

```
python retrieve_chroma.py --queries output/column_queries_groq.json
```

Embeds each query and scores it against **every** table card, so there are no gaps. Output is a
matrix with rows `q1…qN` and columns `T1…TM`; each entry is cosine similarity × 100 (higher is
better). Written to `output/score_matrix.json` and `output/score_matrix.csv`.

Needs `sentence-transformers` (not in `requirements.txt` yet).

### 4. Sample the tables that satisfy the queries

```
python select_tables.py --matrix output/score_matrix.json --threshold 50 --top 10
```

A table **fulfils** a query when its score is at least `--threshold`. The script lists table
combinations where:

- **every query is fulfilled** (completeness is never traded away),
- **no table is redundant** (drop any one and a query is left unfulfilled),
- the number of tables **varies**, whatever it takes to cover the queries.

Each combination is returned as its own queries × tables sub-matrix and the list is **sorted by
average score**: each query counts its best score within the combination, and those best scores
are averaged. A query that no table reaches the threshold for is covered by its best table and
reported under `weak_queries`. If the search is cut short (`--max-tables` too small, or the size
limit is hit) it falls back to a greedy cover, so the result is never incomplete or empty.
Output: `output/ranked_table_matrices.json`.

### 5. Augment from DuckDB (planned)

Take the top-ranked combination, read the matched tables from DuckDB by `table_id`, join them to
the seed on the entity key, and append the columns that fulfilled the queries. Because Chroma and
DuckDB map 1:1, the appended values are the stored true values.

## Status

| Part | State |
|---|---|
| Scrape / clean / quality / DuckDB store | done |
| Seed profiling, query generation, relation equations | done |
| Chroma indexing (`index_chroma.py`) | **missing from the repo** — `retrieve_chroma.py` expects it |
| Score matrix (`retrieve_chroma.py`) | done, not yet run against a real index |
| Table selection (`select_tables.py`) | done, tested on synthetic matrices |
| Relation-aware selection (using `belongs_to` / `subset_of`) | not yet; selection currently uses scores only |
| Augmentation from DuckDB | not built |

## Setup

```
pip install -r requirements.txt
pip install sentence-transformers chromadb
```

`chromadb` and `sentence-transformers` are used by the retrieval step but are not listed in
`requirements.txt` yet.

## Layout

```
data/            seed CSVs, plus (generated) raw/tables/clean/imputed, the DuckDB file and the Chroma index
output/          generated JSON/CSV (git-ignored)
preprocessing/   scrape.py, clean.py, quality.py, dbms_store.py
profiler.py  query.py  querygroq.py  equationextractorforembedding.py   seed -> queries
retrieve_chroma.py  select_tables.py                                    queries -> scores -> tables
```
