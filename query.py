"""Generate a natural-language description for every column from the simple stats file.

For each column Gemini says what the column is, which entity it describes
(the "owner") and how that entity relates to the others, e.g.
    population -> owned by the country/year record, not by capital_city.

Usage:
    Put GEMINI_API_KEY=... in a .env file next to this script (or set it in the environment).
    python query.py
    python query.py --stats output/simple_stats.json --output output/column_queries.json
"""

from __future__ import annotations

import argparse
import json
import os

from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a .env file without overriding the real environment."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

ENTITY_PROMPT = """Below are per-column statistics of a CSV file and groups of whole sample rows. Each group's rows share one value (shared_column = shared_value).

{stats}

Pass 1: identify the entities (things the rows are about) and how they nest. Return JSON:
{{"entities": [{{"name": str, "key_column": str, "parent": str|null}}], "row_is_about": str}}
- key_column is the column that identifies the entity.
- parent is the entity it belongs to (a hierarchy such as city -> country), or null.
- row_is_about is the finest-grained entity a single row describes.
Use the sample row groups: a column constant across a group's rows is a property of the group's shared entity, not of the finer entities."""

PROMPT = """You are given per-column statistics of a CSV file, groups of whole sample rows, and the entity hierarchy found in pass 1.

{stats}

Entities (pass 1):
{entities}

Pass 2: write one natural-language description for EVERY column, in original order. Per column return:
- "column": the column name
- "description": what the column holds, in plain English
- "owner": the single entity whose property this value is (use an entity name from pass 1)
- "not_owners": other related entities that do NOT own it, and why
- "relations": how the owning entity relates to the others (e.g. "city belongs to country")
- "query": a natural-language question about this column mentioning its owner and value

Decide ownership ONLY from the sample row groups, not from your prior knowledge of the world. Look at the rows inside each group: if a column has the SAME value in every row of a group, it is a property of that group's shared entity (the coarser one). If it DIFFERS between rows of the group, it is a property of a finer entity (the one the rows differ by). Example: rows sharing country=X with identical population => population belongs to the country, even though cities appear in the row. Return only a JSON array."""


def ask(client, prompt: str, model: str):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )
    return json.loads(response.text)


def describe_columns(stats: dict, model: str = MODEL) -> list[dict]:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    stats_text = json.dumps(stats, indent=2, ensure_ascii=False)
    entities = ask(client, ENTITY_PROMPT.format(stats=stats_text), model)
    return ask(client, PROMPT.format(stats=stats_text, entities=json.dumps(entities, indent=2)), model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural-language column descriptions via Gemini.")
    parser.add_argument("--stats", default="output/profiler_output.json", help="Stats JSON from profiler.py.")
    parser.add_argument("--output", "-o", default="output/column_queries.json", help="Where to save the result.")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    load_dotenv()
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("Put GEMINI_API_KEY in a .env file or set it in the environment.")

    with open(args.stats, encoding="utf-8") as handle:
        stats = json.load(handle)

    text = json.dumps(describe_columns(stats, args.model), indent=2, ensure_ascii=False)
    print(text)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
