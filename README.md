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
| Describe + embed into Chroma | `index_chroma.py` | DuckDB → `data/chroma_seed_seek` (`table_cards`, `column_cards`) |

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

To run steps 1-2 on every table at once: `python run_all_tables.py` (profiler + querygroq, writes
`output/tables/<table>/`) and `python run_all_gemini.py` (query.py for every table, same folders).

### 3. Match queries against the index

```
python retrieve_chroma.py --queries output/column_queries_groq.json
```

Embeds each query and scores it against **every** table card, so there are no gaps. Output is a
matrix with rows `q1…qN` and columns `T1…TM`; each entry is cosine similarity × 100 (higher is
better). Written to `output/score_matrix.json` and `output/score_matrix.csv`.

To compare a seed's Groq queries directly against the Groq query files for all source tables:

```
python match_groq_queries.py
```

By default, this reads `output/column_queries_groq.json` and every `*.json` file in
`data/queries/queries/groq/`. Each source file is assigned a `T1`, `T2`, ... column in
case-insensitive filename order; each input query becomes a `q1`, `q2`, ... row. Both sets of
query texts are embedded into the dedicated `data/chroma_groq_queries` ChromaDB collection.
Each matrix cell selects the nearest individual source query from that file. The JSON includes
cosine distances (lower is closer), similarity scores (`1 - distance`, scaled by 100), and the
matched source column/query. CSV outputs are written to
`output/groq_query_score_matrix.csv` and `output/groq_query_distance_matrix.csv`; the readable
similarity matrix is also written to `output/groq_query_score_matrix.txt`. This flow is separate
from `retrieve_chroma.py`, which compares against table cards indexed from DuckDB.

### 4. Sample the tables that satisfy the queries

```
python select_tables.py --matrix output/score_matrix.json --threshold 50 --top 10
```

For the Groq-to-Groq matrix, use:

```
python select_tables.py --matrix output/groq_query_score_matrix.json --threshold 60 \
  --output output/ranked_groq_table_matrices.json
```

A table **fulfils** a query when its score is at least `--threshold`. The selector reads either
the normalized `scores` from `retrieve_chroma.py` or the 0-100 `similarities` from
`match_groq_queries.py`. It considers every threshold-qualified table by default, enumerates all
minimal covers, and ranks them with a max-priority heap by the average of each query's best score.
No redundant supersets are included. By default, every minimal cover is written; use `--top N`
to limit the output. If any query has no table at the threshold, selection fails rather than
silently covering it below threshold. Table combinations must meet these conditions:

- **every query is fulfilled** (completeness is never traded away),
- **no table is redundant** (drop any one and a query is left unfulfilled),
- the number of tables **varies**, whatever it takes to cover the queries.

Each combination is returned as its own queries × tables sub-matrix and the list is **sorted by
average score**: each query counts its best score within the combination, and those best scores
are averaged. If the exact search exceeds its state limit, it raises an error instead of
returning a partial or greedy result. Output: `output/ranked_table_matrices.json`.

### 5. Augment from DuckDB (planned)

Take the top-ranked combination, read the matched tables from DuckDB by `table_id`, join them to
the seed on the entity key, and append the columns that fulfilled the queries. Because Chroma and
DuckDB map 1:1, the appended values are the stored true values.

## Status

| Part | State |
|---|---|
| Scrape / clean / quality / DuckDB store | done |
| Seed profiling, query generation, relation equations | done |
| Chroma indexing (`index_chroma.py`) | done: 10 table cards, 68 column cards |
| Score matrix (`retrieve_chroma.py`) | done, tested against the real index |
| Table selection (`select_tables.py`) | done; runs on the real index, but key-column queries (e.g. country name) pull in unrelated tables |
| Relation-aware selection (using `belongs_to` / `subset_of`) | not yet; selection currently uses scores only |
| Augmentation from DuckDB | not built |

## Setup

```
pip install -r requirements.txt
```

## Layout

```
data/            seed CSVs, plus (generated) raw/tables/clean/imputed, the DuckDB file and the Chroma index
output/          generated JSON/CSV (git-ignored)
preprocessing/   scrape.py, clean.py, quality.py, dbms_store.py
profiler.py  query.py  querygroq.py  equationextractorforembedding.py   seed -> queries
retrieve_chroma.py  match_groq_queries.py  select_tables.py             queries -> scores -> tables
```
