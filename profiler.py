"""Simple CSV column stats.

Numeric columns: avg, min, max.
Text columns:    top TF-IDF terms (each cell is one document).
Boolean columns: true / false counts.
Every column also gets a few example values.

Usage:
    python simple_profiler.py data/example_seed.csv
    python simple_profiler.py data/example_seed.csv --output output/simple_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from typing import Any

import pandas as pd

TRUE_VALUES = {"true", "t", "yes", "y"}
FALSE_VALUES = {"false", "f", "no", "n"}
EXAMPLES = 3
TOP_TERMS = 5
SAMPLE_GROUPS = 3
ROWS_PER_GROUP = 3


def is_boolean(values: pd.Series) -> bool:
    """True for bool dtype or text made only of true/false style words."""
    if pd.api.types.is_bool_dtype(values):
        return True
    if pd.api.types.is_numeric_dtype(values):
        return False
    return set(values.astype(str).str.strip().str.lower().unique()) <= TRUE_VALUES | FALSE_VALUES


def boolean_stats(values: pd.Series) -> dict[str, Any]:
    if pd.api.types.is_bool_dtype(values):
        true_count = int(values.sum())
    else:
        true_count = int(values.astype(str).str.strip().str.lower().isin(TRUE_VALUES).sum())
    return {"true": true_count, "false": len(values) - true_count}


def numeric_stats(values: pd.Series) -> dict[str, Any]:
    return {"avg": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def tfidf_stats(values: pd.Series, top: int = TOP_TERMS) -> dict[str, Any]:
    """Top terms by mean TF-IDF, treating each cell as one document."""
    documents = [re.findall(r"\w+", text.lower()) for text in values.astype(str)]
    documents = [doc for doc in documents if doc]
    if not documents:
        return {"tfidf_top_terms": {}}
    document_frequency = Counter(term for doc in documents for term in set(doc))
    scores: Counter[str] = Counter()
    for doc in documents:
        for term, count in Counter(doc).items():
            idf = math.log((1 + len(documents)) / (1 + document_frequency[term])) + 1
            scores[term] += (count / len(doc)) * idf
    return {"tfidf_top_terms": {term: round(score / len(documents), 4) for term, score in scores.most_common(top)}}


def sample_row_groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Whole rows that share a value in a repeating column.

    Comparing such rows shows which columns stay constant for that value (they belong
    to the grouping column's entity) and which change (they belong to something finer).
    Columns with the fewest distinct values are grouped first.
    """
    candidates = sorted(
        (column for column in frame.columns if 1 < frame[column].nunique() < len(frame)),
        key=lambda column: frame[column].nunique(),
    )
    clean = frame.astype(object).where(frame.notna(), None)
    groups = []
    seen: set[tuple[int, ...]] = set()
    for column in candidates:
        if len(groups) == SAMPLE_GROUPS:
            break
        value = frame[column].value_counts().index[0]
        rows = clean[frame[column] == value].head(ROWS_PER_GROUP)
        if len(rows) < 2 or tuple(rows.index) in seen:
            continue
        seen.add(tuple(rows.index))
        groups.append({"shared_column": column, "shared_value": value, "rows": rows.to_dict("records")})
    return groups


def profile_csv(path: str) -> dict[str, Any]:
    """Read a CSV with pandas and return per-column stats plus sample row groups."""
    frame = pd.read_csv(path)
    result: dict[str, Any] = {}
    for name in frame.columns:
        values = frame[name].dropna()
        if values.empty:
            result[name] = {"type": "empty", "examples": []}
        elif is_boolean(values):
            result[name] = {"type": "boolean", **boolean_stats(values)}
        elif pd.api.types.is_numeric_dtype(values):
            result[name] = {"type": "numeric", **numeric_stats(values)}
        else:
            result[name] = {"type": "text", **tfidf_stats(values)}
        result[name]["examples"] = values.drop_duplicates().head(EXAMPLES).tolist()
    return {"columns": result, "sample_row_groups": sample_row_groups(frame)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple per-column stats for a CSV file.")
    parser.add_argument(
        "input",
        nargs="?",
        default="data/List_of_countries_by_life_expectancy__t4.csv",
        help="CSV file to profile (default: data/List_of_countries_by_life_expectancy__t4.csv)",
    )
    parser.add_argument("--output", "-o",
                         nargs="?",
                                default="output/profiler_output.json",
                         help="Optional path to save the stats as JSON.")
    args = parser.parse_args()

    text = json.dumps(profile_csv(args.input), indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
