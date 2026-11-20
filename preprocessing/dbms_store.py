import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from clean import normalize_key

# Display settings: only change how tables look when printed, not the data
pd.set_option("display.max_columns", None)  # show all columns instead of hiding some with "..."
pd.set_option("display.width", 200)         # allow wider lines so tables don't wrap too early

INPUT_DIR = Path("data/imputed")
DB_PATH = Path("data/seedandseek.duckdb")


# ---------- Connection and setup ----------

def connect(read_only=False):
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def create_registries(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS table_registry (
            table_id     VARCHAR PRIMARY KEY,
            db_table     VARCHAR,
            page_title   VARCHAR,
            source_url   VARCHAR,
            revision_id  BIGINT,
            section      VARCHAR,
            caption      VARCHAR,
            n_rows       INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS column_registry (
            table_id     VARCHAR,
            column_name  VARCHAR,
            role         VARCHAR,
            unit         VARCHAR,
            multiplier   DOUBLE,
            year         INTEGER
        )
    """)


def db_table_name(table_id):
    return "t_" + re.sub(r"\W+", "_", table_id).strip("_").lower()


# ---------- Storing ----------

def store_table(con, meta, df):
    table_id = meta["table_id"]
    name = db_table_name(table_id)

    con.register("incoming", df)
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM incoming')
    con.unregister("incoming")

    con.execute("DELETE FROM table_registry WHERE table_id = ?", [table_id])
    con.execute("DELETE FROM column_registry WHERE table_id = ?", [table_id])

    con.execute(
        "INSERT INTO table_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [table_id, name, meta["page_title"], meta["source_url"], meta["revision_id"],
         meta["section"], meta["caption"], meta["n_rows"]],
    )

    for col in meta["columns"]:
        con.execute(
            "INSERT INTO column_registry VALUES (?, ?, ?, ?, ?, ?)",
            [table_id, col["name"], col["role"], col["unit"], col["multiplier"], col["year"]],
        )


# ---------- Loading (for the augmentation step and the UI) ----------

def list_tables():
    with connect(read_only=True) as con:
        return con.execute("SELECT * FROM table_registry ORDER BY table_id").df()


def get_columns(table_id):
    with connect(read_only=True) as con:
        return con.execute(
            "SELECT * FROM column_registry WHERE table_id = ?", [table_id]
        ).df()


def load_table(table_id):
    with connect(read_only=True) as con:
        result = con.execute(
            "SELECT db_table FROM table_registry WHERE table_id = ?", [table_id]
        ).fetchone()
        if result is None:
            raise KeyError(f"Unknown table_id: {table_id}")
        return con.execute(f'SELECT * FROM "{result[0]}"').df()


# ---------- Main ----------

def load_metadata():
    return json.loads((INPUT_DIR / "imputed_metadata.json").read_text(encoding="utf-8"))


def main():
    con = connect()
    create_registries(con)

    for meta in load_metadata():
        df = pd.read_csv(INPUT_DIR / f"{meta['table_id']}.csv")
        store_table(con, meta, df)
        print(f"Stored {meta['table_id']} ({len(df)} rows)")

    print()
    print(con.execute("SELECT table_id, db_table, n_rows FROM table_registry").df())
    con.close()


if __name__ == "__main__":
    main()

    # Test: join two tables from different pages on normalised keys (computed on the fly)
    population = load_table("List_of_countries_and_dependencies_by_population__t0")
    gdp = load_table("List_of_countries_by_GDP_(nominal)__t0")

    population["join_key"] = population["Location"].map(normalize_key)
    gdp["join_key"] = gdp["Country/Territory"].map(normalize_key)

    joined = population.merge(gdp, on="join_key")
    print()
    print(f"Population rows: {len(population)}, GDP rows: {len(gdp)}, joined: {len(joined)}")
    print(joined[["Location", "Population", "IMF (2026)"]].head())