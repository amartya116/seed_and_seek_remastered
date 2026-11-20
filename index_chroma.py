"""Describe every DuckDB table and number column as a text card, embed the cards,
and store them in ChromaDB.

    DuckDB tables  ->  cards (text)  ->  embeddings  ->  ChromaDB
                                                         table_cards   (one card per table)
                                                         column_cards  (one card per number column,
                                                                        or per block of year columns)

Every card keeps its table_id (and column_name) in its metadata, so the real values can
always be loaded from DuckDB afterwards. retrieve_chroma.py searches these collections.

Run from the folder that contains data/:
    python index_chroma.py
"""

import json
import math
import numbers
import re
from pathlib import Path

import duckdb
import pandas as pd

#Settings

DB_PATH = Path("data/seedandseek.duckdb")
CHROMA_DIR = Path("data/chroma_seed_seek")
DESCRIPTIONS_DIR = Path("output/tables")  # Groq descriptions from run_all_tables.py (optional)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLE_ROWS = 3      # example rows shown in each table card
SAMPLE_COLUMNS = 6   # columns shown in those example rows (keeps cards short)
SAMPLE_VALUES = 5    # example values shown in each column card

MULTIPLIER_WORDS = {1_000: "thousands", 1_000_000: "millions", 1_000_000_000: "billions"}
YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


#Reading from DuckDB

def load_registries():
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = con.execute("SELECT * FROM table_registry ORDER BY table_id").df()
        columns = con.execute("SELECT * FROM column_registry").df()
    return tables, columns


def load_table(db_table):
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(f'SELECT * FROM "{db_table}"').df()


# Small text helpers

def clean_text(value):
    """Text without extra spaces, or None for empty / missing values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    value = " ".join(str(value).split())
    return value or None


def format_value(value):
    """Numbers become readable text: 32383920.0 -> '32,383,920', 85.513 -> '85.51'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{round(float(value), 2):g}"
    return clean_text(value) or ""


def unit_label(column):
    """'USD, millions' from the unit and multiplier saved by clean.py."""
    parts = []
    unit = clean_text(column["unit"])
    if unit:
        parts.append(unit)
    multiplier = column["multiplier"]
    if pd.notna(multiplier) and int(multiplier) in MULTIPLIER_WORDS:
        parts.append(MULTIPLIER_WORDS[int(multiplier)])
    return ", ".join(parts)


def column_label(column):
    """'IMF (2026) (USD, millions)', or just the name if there is no unit."""
    unit = unit_label(column)
    return f"{column['column_name']} ({unit})" if unit else column["column_name"]


#Table cards

def table_card(table, table_columns, df):
    names = [n for n in table_columns["column_name"] if n in df.columns]
    keys = table_columns.loc[table_columns["role"] == "key", "column_name"].tolist()
    years = [int(y) for y in table_columns["year"].dropna()]

    lines = [f"Table: {clean_text(table['page_title'])}"]
    section = clean_text(table["section"])
    caption = clean_text(table["caption"])
    if section:
        lines.append(f"Section: {section}")
    if caption and caption != section:
        lines.append(f"Caption: {caption}")
    if keys:
        lines.append(f"Each row is about: {', '.join(keys)}")
    lines.append("Columns: " + " | ".join(column_label(c) for _, c in table_columns.iterrows()))

    shown = (keys + [n for n in names if n not in keys])[:SAMPLE_COLUMNS]
    lines.append("Example rows (" + " | ".join(shown) + "):")
    for _, row in df[shown].head(SAMPLE_ROWS).iterrows():
        lines.append(" | ".join(format_value(v) for v in row))

    metadata = {
        "card_type": "table",
        "table_id": table["table_id"],
        "db_table": table["db_table"],
        "page_title": clean_text(table["page_title"]) or "",
        "source_url": clean_text(table["source_url"]) or "",
        "key_columns": ", ".join(keys),
        "n_rows": int(table["n_rows"]),
        "n_columns": len(names),
    }
    if years:
        metadata["year_min"] = min(years)
        metadata["year_max"] = max(years)

    return {"id": table["table_id"], "text": "\n".join(lines), "metadata": metadata}


#Column cards

def load_descriptions(table_id):
    """Groq's description of each column, if run_all_tables.py has been run for this table."""
    path = DESCRIPTIONS_DIR / table_id / "column_queries_groq.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {
        r["column"]: clean_text(r.get("description"))
        for r in records
        if isinstance(r, dict) and "column" in r
    }


def column_groups(number_columns):
    """Columns that only differ by a year form one group: 'Historical data - <year>'."""
    groups = {}
    for _, column in number_columns.iterrows():
        name = column["column_name"]
        if YEAR.search(name):
            pattern = " ".join(YEAR.sub("<year>", name).split())
        else:
            pattern = name
        groups.setdefault(pattern, []).append(column)
    return list(groups.items())


def column_card(table, pattern, group, keys, df, descriptions):
    names = [c["column_name"] for c in group]
    years = sorted({int(y) for c in group for y in YEAR.findall(c["column_name"])})
    first = group[0]
    is_block = len(group) > 1

    lines = [f"Column: {pattern if is_block else names[0]}"]
    if is_block:
        lines.append(f"One column per year: {', '.join(names)}")
    description = descriptions.get(names[0])
    if description:
        lines.append(f"Measures: {description}")
    unit = unit_label(first)
    if unit:
        lines.append(f"Unit: {unit}")
    if years:
        lines.append(f"Years: {years[0]}" + (f" to {years[-1]}" if len(years) > 1 else ""))
    if keys:
        lines.append(f"Value for each: {', '.join(keys)}")
    lines.append(f"From table: {clean_text(table['page_title'])}")
    caption = clean_text(table["caption"])
    if caption:
        lines.append(f"Table caption: {caption}")

    key = keys[0] if keys and keys[0] in df.columns else None
    samples = []
    for _, row in df.head(SAMPLE_VALUES).iterrows():
        value = format_value(row[names[0]])
        samples.append(f"{row[key]}: {value}" if key else value)
    lines.append("Sample values: " + "; ".join(samples))

    metadata = {
        "card_type": "column",
        "table_id": table["table_id"],
        "db_table": table["db_table"],
        "column_name": names[0],
        "column_names": " | ".join(names),
        "unit": clean_text(first["unit"]) or "",
        "multiplier": float(first["multiplier"]) if pd.notna(first["multiplier"]) else 1.0,
        "key_columns": ", ".join(keys),
    }
    if years:
        metadata["year_min"] = years[0]
        metadata["year_max"] = years[-1]

    card_id = f"{table['table_id']}::{pattern if is_block else names[0]}"
    return {"id": card_id, "text": "\n".join(lines), "metadata": metadata}


def column_cards(table, table_columns, df):
    keys = table_columns.loc[table_columns["role"] == "key", "column_name"].tolist()
    number_columns = table_columns[table_columns["role"] == "number"]
    number_columns = number_columns[number_columns["column_name"].isin(df.columns)]
    descriptions = load_descriptions(table["table_id"])
    return [column_card(table, p, g, keys, df, descriptions) for p, g in column_groups(number_columns)]


#All cards

def build_cards():
    tables, columns = load_registries()
    all_table_cards, all_column_cards = [], []
    for _, table in tables.iterrows():
        table_columns = columns[columns["table_id"] == table["table_id"]]
        df = load_table(table["db_table"])
        all_table_cards.append(table_card(table, table_columns, df))
        all_column_cards.extend(column_cards(table, table_columns, df))
    return all_table_cards, all_column_cards


def save_cards(table_cards, column_cards_):
    """Readable copy of all cards, to check what was embedded."""
    path = CHROMA_DIR.parent / "cards.json"
    path.write_text(
        json.dumps({"table_cards": table_cards, "column_cards": column_cards_}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


#Embedding and storing in ChromaDB

def store(client, name, cards, model):
    """Rebuild one collection from scratch: embed the card texts and add them."""
    try:
        client.delete_collection(name)
    except Exception:
        pass  # the collection did not exist yet

    collection = client.create_collection(
        name=name,
        metadata={"embedding_model": EMBEDDING_MODEL, "hnsw:space": "cosine"},
    )
    texts = [card["text"] for card in cards]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
    collection.add(
        ids=[card["id"] for card in cards],
        documents=texts,
        embeddings=vectors,
        metadatas=[card["metadata"] for card in cards],
    )
    return collection


def test_search(collection, model, question, n_results=3):
    vector = model.encode([question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=n_results, include=["metadatas", "distances"])
    print(f'\n"{question}"')
    for card_id, distance in zip(result["ids"][0], result["distances"][0]):
        print(f"  {1 - distance:.2f}  {card_id}")


def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    table_cards, all_column_cards = build_cards()
    print(f"Built {len(table_cards)} table cards and {len(all_column_cards)} column cards")
    print(f"Saved a readable copy to {save_cards(table_cards, all_column_cards)}")

    print(f"\nLoading embedding model {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    tables = store(client, "table_cards", table_cards, model)
    columns = store(client, "column_cards", all_column_cards, model)
    print(f"\nStored {tables.count()} table cards and {columns.count()} column cards in {CHROMA_DIR}")

    if (tables.metadata or {}).get("hnsw:space") != "cosine":
        print("WARNING: table_cards is not marked as cosine; retrieve_chroma.py will refuse to run")

    test_search(tables, model, "What is the GDP of a country?")
    test_search(columns, model, "Life expectancy of women in a country")


if __name__ == "__main__":
    main()