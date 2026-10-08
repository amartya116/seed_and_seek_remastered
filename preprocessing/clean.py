import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Display settings: only change how tables look when printed, not the data
pd.set_option("display.max_columns", None)  # show all columns instead of hiding some with "..."
pd.set_option("display.width", 200)         # allow wider lines so tables don't wrap too early

TABLES_DIR = Path("data/tables")
CLEAN_DIR = Path("data/clean")

# Units and multipliers found in column names or captions (first match wins)
UNIT_PATTERNS = [
    (r"%|percent", "%"),
    (r"km2|km²|square kilomet", "km²"),
    (r"mi2|mi²|square mile", "mi²"),
    (r"us\$|usd|dollar", "USD"),
    (r"€|\beur\b|euro", "EUR"),
    (r"\byears?\b", "years"),
]

MULTIPLIER_PATTERNS = [
    (r"billion|\bbn\b", 1_000_000_000),
    (r"million|\bmn\b", 1_000_000),
    (r"thousand", 1_000),
]


# ---------- Loading ----------

def load_metadata():
    file_path = TABLES_DIR / "tables_metadata.json"
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_table(table_id):
    return pd.read_csv(TABLES_DIR / f"{table_id}.csv")


# ---------- Step 1 + 2: headers and cells ----------

def remove_footnotes(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"\[.*?\]", "", text)
    return " ".join(text.split())


def clean_headers(df):
    df = df.copy()
    df.columns = [remove_footnotes(col) for col in df.columns]

    keep = [
        col for col in df.columns
        if col and not col.startswith("Unnamed") and not df[col].isna().all()
    ]
    df = df[keep]

    seen = {}
    unique_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            unique_columns.append(f"{col} ({seen[col]})")
        else:
            seen[col] = 0
            unique_columns.append(col)
    df.columns = unique_columns

    return df


def clean_cells(df):
    df = df.copy()
    df = df.map(remove_footnotes)

    df = df.replace({"": None, "—": None, "–": None, "-": None, "N/A": None, "n/a": None})

    matches = pd.DataFrame({col: df[col].astype(str) == col for col in df.columns})
    header_like = matches.sum(axis=1)
    df = df[header_like < len(df.columns) / 2]

    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    return df


# ---------- Step 3: key columns ----------

def normalize_key(value):
    if pd.isna(value):
        return None
    text = re.sub(r"\(.*?\)", "", str(value))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = " ".join(text.split())
    return text or None


def looks_like_number(value):
    if isinstance(value, (int, float)):
        return True
    text = re.sub(r"\(.*?\)", "", str(value)).strip()
    return bool(re.fullmatch(r"[-+−]?[\d.,\s]+%?", text))


def find_key_columns(df, min_rows=5, min_text_share=0.8, min_unique_share=0.8, max_avg_length=50):
    if len(df) < min_rows:
        return []

    keys = []
    for col in df.columns:
        values = df[col].dropna()
        if len(values) < min_rows:
            continue

        text_share = 1 - values.map(looks_like_number).mean()
        unique_share = values.nunique() / len(values)
        avg_length = values.astype(str).str.len().mean()

        if text_share >= min_text_share and unique_share >= min_unique_share and avg_length <= max_avg_length:
            keys.append(col)

    return keys


# ---------- Step 4: numbers, duplicates, units and years ----------

def to_number(value):
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    text = re.sub(r"\(.*?\)", "", value)
    text = text.replace(",", "").replace("%", "").replace("−", "-").replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return None


def find_number_columns(df, min_share=0.8):
    number_columns = []
    for col in df.columns:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        if values.map(looks_like_number).mean() >= min_share:
            number_columns.append(col)
    return number_columns


def parse_numbers(df):
    df = df.copy()
    number_columns = find_number_columns(df)
    for col in number_columns:
        df[col] = df[col].map(to_number)
    return df, number_columns


def remove_duplicate_columns(df):
    kept = []
    removed = []
    for col in df.columns:
        if any(df[col].equals(df[other]) for other in kept):
            removed.append(col)
        else:
            kept.append(col)
    return df[kept], removed


def find_pattern(text, patterns):
    for pattern, result in patterns:
        if re.search(pattern, text):
            return result
    return None


def detect_unit(column_name, caption=None):
    column_text = column_name.lower()
    caption_text = (caption or "").lower()

    unit = find_pattern(column_text, UNIT_PATTERNS) or find_pattern(caption_text, UNIT_PATTERNS)
    multiplier = (
        find_pattern(column_text, MULTIPLIER_PATTERNS)
        or find_pattern(caption_text, MULTIPLIER_PATTERNS)
        or 1
    )
    return unit, multiplier


def detect_year(column_name):
    match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", column_name)
    return int(match.group(1)) if match else None


# ---------- Step 5: clean one table (works for any table) ----------

def clean_table(df, caption=None):
    df = clean_cells(clean_headers(df))

    key_columns = find_key_columns(df)
    if not key_columns:
        return None, None, []

    df, number_columns = parse_numbers(df)
    df, removed_columns = remove_duplicate_columns(df)
    key_columns = [k for k in key_columns if k in df.columns]
    number_columns = [n for n in number_columns if n in df.columns]

    for key in key_columns:
        df[f"{key}__norm"] = df[key].map(normalize_key)

    columns_info = []
    for col in df.columns:
        if col.endswith("__norm"):
            continue
        if col in key_columns:
            role = "key"
        elif col in number_columns:
            role = "number"
        else:
            role = "text"

        unit, multiplier = detect_unit(col, caption) if role == "number" else (None, 1)
        columns_info.append({
            "name": col,
            "role": role,
            "unit": unit,
            "multiplier": multiplier,
            "year": detect_year(col),
        })

    return df, columns_info, removed_columns


# ---------- Saving ----------

def save_clean_table(table_id, df):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DIR / f"{table_id}.csv", index=False, encoding="utf-8")


def main():
    clean_metadata = []

    for meta in load_metadata():
        df, columns_info, removed = clean_table(load_table(meta["table_id"]), meta["caption"])

        if df is None:
            print(f"Skipped {meta['table_id']} (no key column)")
            continue

        if removed:
            print(f"  Removed duplicate columns in {meta['table_id']}: {removed}")

        save_clean_table(meta["table_id"], df)
        clean_metadata.append({**meta, "n_rows": len(df), "columns": columns_info})
        print(f"Cleaned {meta['table_id']}: {len(df)} rows, {len(columns_info)} columns")

    file_path = CLEAN_DIR / "clean_metadata.json"
    file_path.write_text(json.dumps(clean_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata for {len(clean_metadata)} tables to {file_path}")


if __name__ == "__main__":
    main()