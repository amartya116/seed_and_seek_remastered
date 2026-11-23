# Seed & Seek (remastered)

Data augmentation from a seed file. You give it a small **seed CSV**; it finds other tables that
describe the same entities and picks the combinations of tables that can supply every seed column.
Augmentation (joining those tables onto the seed) comes next and is not built yet.

## Pipeline

```
seed CSV ─ profiler.py ─► stats ─ querygroq.py / query.py ─► one NL query + relations per seed column
                                              │
                                              ├─► equationextractorforembedding.py ─► seed equation
                                              │                                      (belongs_to / subset_of)
                                              ▼
            match_groq_queries.py (ChromaDB) ─► score matrix  (seed columns x source tables, 0-100)
                                              │
                                              ▼
            select_tables.py + table_equations.py ─► max-heap of minimal table sets that satisfy the equation
```

Run from the project root; paths are relative to it. Put API keys in a `.env` file next to the
scripts (it is git-ignored).

### 1. Profile the seed

```
python profiler.py data/List_of_countries_by_life_expectancy__t4.csv   # -> output/profiler_output.json
```

Per-column stats: numeric min/max/avg, TF-IDF terms for text, boolean counts, sample rows.

### 2. Queries and relations

```
python querygroq.py     # -> output/column_queries_groq.json and ..._non_derived.json   (needs GROQ_API_KEY)
python query.py         # same, with Gemini                                              (needs GEMINI_API_KEY)
```

An LLM describes each column, writes one natural-language query for it, and records which entity
owns it (`belongs_to`), whether it is a narrower scope of another column (`subset_of`), and whether it
is computed from other columns (`is_derived`, `derived_from`). The `_non_derived` file keeps only the
columns that cannot be computed from the others.

The source tables were run through the same step; their query files are in
`data/queries/queries/groq/` (one JSON per table) and `data/queries/queries/gemini/`.

### 3. Seed equation

```
python equationextractorforembedding.py   # -> output/column_relationship_equations.json
```

Turns the relations into one equation, e.g.
`Table(columns=[...]) AND belongs_to("... - M", "Countries") AND subset_of("... - M", "... - All")`.

### 4. Matching and score matrix

```
python match_groq_queries.py
```

Reads `output/column_queries_groq_non_derived.json` (derived seed columns are computed later from
their inputs) and every file in `data/queries/queries/groq/`. Each source file becomes a column `T1`,
`T2`, ... (filename order); each seed column becomes a row `q1`, `q2`, .... On the source side every
column is a candidate, derived or not.

What is embedded (`--embed full`, default) is `<column name>. <description>. <query>`, because the
query alone shares a template ("... for this country?") with every other query. Embeddings live in
`data/chroma_groq_queries` (cosine, `all-MiniLM-L6-v2`).

Each cell is the best score among that table's columns; the JSON also keeps the score of **every**
source column (`column_scores`). Outputs: `output/groq_query_score_matrix.json` / `.csv` / `.txt` and
`output/groq_query_distance_matrix.csv`.

### 5. Heap of table sets (candidates)

```
python select_tables.py --threshold 60
# -> output/ranked_groq_table_matrices.json, output/groq_table_set_max_heap.txt
```

Enumerates all **minimal** table sets in which every seed column has a table scoring >= `--threshold`
(dropping any table would leave a seed column uncovered), and ranks them in a max-heap by the average
of each seed column's best score. These are the candidates that step 6 joins.

`--mode equation` instead keeps only sets that satisfy the seed equation (rules in `table_equations.py`:
`subset_of` groups inside one table, `belongs_to` the table's key, key matches the seed entity,
one-to-one columns).

### 6. Join every heap node

```
python join_heap_nodes.py --seed data/List_of_countries_by_life_expectancy__t4.csv
# -> output/join_report.txt, output/join_report.json
```

Reads each node's tables from `data/seedandseek.duckdb` **read-only** (the database is never
modified), and joins them in memory on their entity key with inner, left, right and outer joins.
Keys are normalised (case, accents, punctuation, footnotes, parenthetical notes). Each seed column is
represented by the best-scoring column among the node's tables. Per node and join type the report gives
rows, distinct keys, % of seed entities matched, duplicate keys, NULL %, whether `belongs_to` holds (one
value per entity), and agreement with the seed's own values. It also flags a source column that stands
for several seed columns, and lists seed keys a table does not contain.

### Source tables in DuckDB

```
python store_duckdb.py --input "C:/Users/amart/Final Data/imputed"   # -> data/seedandseek.duckdb
```

Loads the cleaned/imputed source tables (`<table_id>.csv` + `imputed_metadata.json`) into DuckDB:
one table `t_<table_id>` per source table, plus `table_registry` and `column_registry`.
`store_duckdb.load_table(table_id)` reads one back as a DataFrame. The database file is git-ignored.

### 7. Augmentation (next)

Join each heap node's tables on the entity key (inner / left / right / outer) and append the bound
columns to the seed. This needs the source tables' rows, which are not in this repo.

## Web app (one click)

```
python app.py      # then open http://127.0.0.1:8000
```

Upload a seed CSV, set the match threshold, and click **Run pipeline**. It runs steps 1-6 in order
(each script with the same Python interpreter, from the project root), shows each step's status and
log, and displays the heap (`output/groq_table_set_max_heap.txt`) and the join report
(`output/join_report.txt`); both can be downloaded. Uploads are saved
to `data/uploads/` (git-ignored). It needs `GROQ_API_KEY` in `.env`, uses only the standard library,
and listens on 127.0.0.1 only. One run at a time.

## Setup

```
pip install -r requirements.txt
```

## Layout

```
data/List_of_countries_by_life_expectancy__t4.csv   seed table used for testing
data/queries/queries/{groq,gemini}/                 query + relation files of the source tables
data/chroma_groq_queries/                           Chroma index (generated, git-ignored)
output/                                             generated JSON/CSV/TXT (git-ignored)
profiler.py  querygroq.py  query.py                 seed -> stats -> queries + relations
equationextractorforembedding.py                    relations -> seed equation
match_groq_queries.py                               queries -> score matrix
select_tables.py  table_equations.py                score matrix (+ equations) -> heap
join_heap_nodes.py                                  heap nodes -> joins from DuckDB (read-only) -> report
store_duckdb.py                                     source tables -> data/seedandseek.duckdb
app.py  web/index.html                               web UI: upload CSV, run everything, show the heap
```

## AI assistance

Parts of this project were developed with help from Claude (Anthropic), used as a coding assistant:

- Claude reviewed the matching and table-selection steps and implemented the agreed changes:
  embedding column name and description with the query, per-column scores, and equation-aware
  table selection (`table_equations.py`: `subset_of` groups, entity-key checks, one-to-one binding).
- An explainer PDF for the team was generated with Claude.

The project direction, the choice of approach, and the review of all results are the team's own.
Commits that Claude contributed to carry a `Co-Authored-By: Claude` trailer.
