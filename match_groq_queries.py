"""Match seed-column queries against the queries generated for every Groq source table.

Each source JSON file becomes one T-column. For every input query, a matrix cell is
the nearest (lowest cosine distance) source query from that file.

    python match_groq_queries.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_query_records(path: Path, table_label: Optional[str] = None) -> list[dict]:
    """Load column query records from one query JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of column records in {path}")

    queries = []
    for record_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Column item {record_index} in {path} is not an object")

        column = item.get("column")
        texts = item.get("queries", item.get("query"))
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(column, str) or not isinstance(texts, list):
            raise ValueError(f"Column item {record_index} in {path} is missing column or query/queries")

        for query_index, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                queries.append({
                    "column": column,
                    "query": text.strip(),
                    "record_index": record_index,
                    "query_index": query_index,
                    "table_label": table_label or "",
                    "source_file": path.name,
                })

    if not queries:
        raise ValueError(f"No usable natural-language queries found in {path}")
    return queries


def create_collection(client, name: str):
    """Create a fresh cosine collection so removed or changed source files do not linger."""
    existing_names = {
        getattr(collection, "name", collection)
        for collection in client.list_collections()
    }
    if name in existing_names:
        client.delete_collection(name)
    return client.create_collection(
        name=name,
        metadata={"embedding_model": EMBEDDING_MODEL, "hnsw:space": "cosine"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="output/column_queries_groq.json",
                        help="Input seed-column query JSON")
    parser.add_argument("--sources", default="data/queries/queries/groq",
                        help="Folder containing one Groq query JSON per source table")
    parser.add_argument("--chroma", default="data/chroma_groq_queries",
                        help="Dedicated persistent ChromaDB directory")
    parser.add_argument("--collection", default="groq_query_match",
                        help="Chroma collection to rebuild and use")
    parser.add_argument("--output", default="output/groq_query_score_matrix.json",
                        help="Output JSON with similarity, distance, and nearest-match matrices")
    parser.add_argument("--csv", default="output/groq_query_score_matrix.csv",
                        help="Output CSV containing cosine similarity scores scaled by 100")
    parser.add_argument("--distance-csv", default="output/groq_query_distance_matrix.csv",
                        help="Output CSV containing cosine distances (lower is better)")
    parser.add_argument("--txt", default="output/groq_query_score_matrix.txt",
                        help="Plain-text similarity matrix for easy viewing")
    args = parser.parse_args()

    input_path = Path(args.queries)
    source_dir = Path(args.sources)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input query file not found: {input_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source query folder not found: {source_dir}")

    source_files = sorted(source_dir.glob("*.json"), key=lambda path: path.name.casefold())
    if not source_files:
        raise FileNotFoundError(f"No JSON source query files found in {source_dir}")

    input_queries = load_query_records(input_path)
    source_tables = []
    source_queries = []
    for index, source_file in enumerate(source_files, start=1):
        label = f"T{index}"
        records = load_query_records(source_file, label)
        source_tables.append({"id": label, "source_file": source_file.name})
        source_queries.extend(records)

    print(f"Loading embedding model {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    all_records = [
        *({"kind": "source", **record} for record in source_queries),
        *({"kind": "input", **record} for record in input_queries),
    ]
    vectors = model.encode(
        [record["query"] for record in all_records],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    chroma_dir = Path(args.chroma)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = create_collection(client, args.collection)
    collection.add(
        ids=[f"query_{index:06d}" for index in range(len(all_records))],
        documents=[record["query"] for record in all_records],
        embeddings=vectors,
        metadatas=[
            {
                "kind": record["kind"],
                "column": record["column"],
                "table_label": record["table_label"],
                "source_file": record["source_file"],
                "record_index": record["record_index"],
                "query_index": record["query_index"],
            }
            for record in all_records
        ],
    )

    source_count = len(source_queries)
    if collection.count() != len(all_records):
        raise RuntimeError("Not all input and source queries were stored in ChromaDB")

    distances = []
    similarities = []
    best_matches = []
    for query_index, input_query in enumerate(input_queries):
        result = collection.query(
            query_embeddings=[vectors[source_count + query_index]],
            n_results=source_count,
            where={"kind": "source"},
            include=["metadatas", "distances", "documents"],
        )

        best_by_table = {}
        for distance, metadata, document in zip(
            result["distances"][0], result["metadatas"][0], result["documents"][0]
        ):
            table_label = metadata["table_label"]
            if table_label not in best_by_table or distance < best_by_table[table_label]["distance"]:
                best_by_table[table_label] = {
                    "distance": float(distance),
                    "source_column": metadata["column"],
                    "source_query": document,
                    "source_file": metadata["source_file"],
                }

        if len(best_by_table) != len(source_tables):
            raise RuntimeError(
                f"Expected a nearest source query from all {len(source_tables)} files "
                f"for input query q{query_index + 1}; found {len(best_by_table)}"
            )

        row_distances = []
        row_similarities = []
        row_matches = []
        for table in source_tables:
            match = best_by_table[table["id"]]
            row_distances.append(match["distance"])
            row_similarities.append((1.0 - match["distance"]) * 100.0)
            row_matches.append(match)
        distances.append(row_distances)
        similarities.append(row_similarities)
        best_matches.append(row_matches)

    output = {
        "embedding_model": EMBEDDING_MODEL,
        "metric": "cosine distance (lower is closer); similarity = (1 - distance) * 100",
        "cell_rule": "Each cell uses the nearest individual source query in that T file.",
        "query_source": str(input_path),
        "queries": [
            {"id": f"q{index}", "seed_column": query["column"], "query": query["query"]}
            for index, query in enumerate(input_queries, start=1)
        ],
        "tables": source_tables,
        "similarities": similarities,
        "distances": distances,
        "best_matches": best_matches,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    row_ids = [query["id"] for query in output["queries"]]
    table_ids = [table["id"] for table in source_tables]
    score_df = pd.DataFrame(similarities, columns=table_ids, index=row_ids)
    score_df.insert(0, "query", [query["query"] for query in input_queries])
    distance_df = pd.DataFrame(distances, columns=table_ids, index=row_ids)
    distance_df.insert(0, "query", [query["query"] for query in input_queries])

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(csv_path, index_label="id", encoding="utf-8")
    distance_csv_path = Path(args.distance_csv)
    distance_csv_path.parent.mkdir(parents=True, exist_ok=True)
    distance_df.to_csv(distance_csv_path, index_label="id", encoding="utf-8")

    text_path = Path(args.txt)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    source_legend = "\n".join(
        f"  {table['id']} = {table['source_file']}" for table in source_tables
    )
    matrix_df = pd.DataFrame(similarities, columns=table_ids, index=row_ids)
    matrix_df.insert(
        0,
        "seed_column",
        [query["column"] for query in input_queries],
    )
    text_path.write_text(
        "Groq query match matrix\n"
        "Cell value: cosine similarity (%) = (1 - cosine distance) * 100; higher is closer.\n\n"
        f"Source files:\n{source_legend}\n\n"
        "Similarity matrix:\n"
        f"{matrix_df.round(2).to_string(index=True, index_names=False)}\n",
        encoding="utf-8",
    )

    print(
        f"Saved {len(input_queries)} x {len(source_tables)} matrices to "
        f"{output_path}, {csv_path}, {distance_csv_path}, and {text_path}"
    )
    print(score_df.drop(columns="query").round(2).to_string())


if __name__ == "__main__":
    main()
