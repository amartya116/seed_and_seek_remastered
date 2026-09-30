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


def profile_csv(path: str) -> dict[str, Any]:
    """Read a CSV with pandas and return stats and examples per column."""
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple per-column stats for a CSV file.")
    parser.add_argument("input", help="Path to the CSV file.")
    parser.add_argument("--output", "-o", help="Optional path to save the stats as JSON.")
    args = parser.parse_args()

    text = json.dumps(profile_csv(args.input), indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
