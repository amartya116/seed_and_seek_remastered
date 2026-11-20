"""Run query.py (Gemini) for every table that already has profiler statistics.

Reads   output/tables/<table name>/profiler_output.json   (made by run_all_tables.py)
Writes  output/tables/<table name>/column_queries_gemini.json

Tables that already have a Gemini result are skipped, so you can stop and restart.

Usage:
    Put GEMINI_API_KEY=... in the .env file next to this script.
    python run_all_gemini.py
    python run_all_gemini.py --model gemini-2.5-flash
"""

import argparse
import json
import os
import time
from pathlib import Path

from query import MODEL, describe_columns, load_dotenv

RESULT_FILE = "column_queries_gemini.json"


def ask_gemini(stats, model, attempts=5):
    """describe_columns from query.py, retried when Gemini's free limit is reached."""
    for attempt in range(attempts):
        try:
            return describe_columns(stats, model)
        except Exception as error:
            busy = "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)
            if not busy or attempt == attempts - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"  Gemini limit reached, waiting {wait} seconds")
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini column descriptions for every profiled table.")
    parser.add_argument("--tables-dir", default="output/tables",
                        help="Folder with one sub-folder per table (default: output/tables)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--pause", type=float, default=10,
                        help="Seconds to wait between tables")
    parser.add_argument("--force", action="store_true", help="Redo tables that are already finished")
    args = parser.parse_args()

    load_dotenv()
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("Put GEMINI_API_KEY in the .env file or set it in the environment.")

    table_dirs = sorted(d for d in Path(args.tables_dir).iterdir() if (d / "profiler_output.json").exists())
    if not table_dirs:
        raise SystemExit(f"No profiled tables in {args.tables_dir}. Run run_all_tables.py first.")

    done, skipped, failed = [], [], []
    for number, table_dir in enumerate(table_dirs, start=1):
        print(f"[{number}/{len(table_dirs)}] {table_dir.name}")

        if (table_dir / RESULT_FILE).exists() and not args.force:
            print("  already finished, skipping")
            skipped.append(table_dir.name)
            continue

        try:
            stats = json.loads((table_dir / "profiler_output.json").read_text(encoding="utf-8"))
            columns = ask_gemini(stats, args.model)
            (table_dir / RESULT_FILE).write_text(
                json.dumps(columns, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"  done -> {table_dir / RESULT_FILE}")
            done.append(table_dir.name)
        except Exception as error:  # one broken table should not stop the others
            print(f"  FAILED: {error}")
            failed.append(table_dir.name)

        if number < len(table_dirs):
            time.sleep(args.pause)

    print(f"\nFinished: {len(done)} new, {len(skipped)} skipped, {len(failed)} failed")
    for table in failed:
        print(f"  failed: {table}")


if __name__ == "__main__":
    main()