"""Load the cleaned/imputed source tables into DuckDB (one table per source table, plus registries).

    python store_duckdb.py --input "C:/Users/amart/Final Data/imputed"

The input folder needs <table_id>.csv files and imputed_metadata.json (list of table records with
table_id, page_title, source_url, revision_id, section, caption, n_rows, columns[name, role, unit,
multiplier, year]). Each table is stored as t_<table_id> (lower-case, non-word characters -> _), and
table_registry / column_registry describe them. Re-running replaces tables with the same table_id.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path('data/seedandseek.duckdb')


def db_table_name(table_id: str) -> str:
    return 't_' + re.sub(r'\W+', '_', table_id).strip('_').lower()


def create_registries(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS table_registry (
            table_id VARCHAR PRIMARY KEY, db_table VARCHAR, page_title VARCHAR, source_url VARCHAR,
            revision_id BIGINT, section VARCHAR, caption VARCHAR, n_rows INTEGER)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS column_registry (
            table_id VARCHAR, column_name VARCHAR, role VARCHAR, unit VARCHAR, multiplier DOUBLE, year INTEGER)
    """)


def store_table(con, meta: dict, df: pd.DataFrame):
    table_id, name = meta['table_id'], db_table_name(meta['table_id'])
    con.register('incoming', df)
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM incoming')
    con.unregister('incoming')
    con.execute('DELETE FROM table_registry WHERE table_id = ?', [table_id])
    con.execute('DELETE FROM column_registry WHERE table_id = ?', [table_id])
    con.execute('INSERT INTO table_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                [table_id, name, meta.get('page_title'), meta.get('source_url'), meta.get('revision_id'),
                 meta.get('section'), meta.get('caption'), len(df)])
    for col in meta.get('columns', []):
        con.execute('INSERT INTO column_registry VALUES (?, ?, ?, ?, ?, ?)',
                    [table_id, col.get('name'), col.get('role'), col.get('unit'), col.get('multiplier'),
                     col.get('year')])


def load_table(table_id: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Read one stored source table back as a DataFrame."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        row = con.execute('SELECT db_table FROM table_registry WHERE table_id = ?', [table_id]).fetchone()
        if row is None:
            raise KeyError(f'Unknown table_id: {table_id}')
        return con.execute(f'SELECT * FROM "{row[0]}"').df()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', default='data/imputed', help='folder with <table_id>.csv and imputed_metadata.json')
    p.add_argument('--db', default=str(DB_PATH))
    args = p.parse_args()

    source = Path(args.input)
    metadata = json.loads((source / 'imputed_metadata.json').read_text(encoding='utf-8'))
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(args.db) as con:
        create_registries(con)
        for meta in metadata:
            csv_path = source / f"{meta['table_id']}.csv"
            if not csv_path.exists():
                print(f"skip {meta['table_id']}: {csv_path.name} not found")
                continue
            df = pd.read_csv(csv_path)
            store_table(con, meta, df)
            print(f"stored {meta['table_id']:<58} -> {db_table_name(meta['table_id'])} ({len(df)} rows, {df.shape[1]} cols)")
        print()
        print(con.execute('SELECT table_id, db_table, n_rows FROM table_registry ORDER BY table_id').df().to_string(index=False))


if __name__ == '__main__':
    main()
