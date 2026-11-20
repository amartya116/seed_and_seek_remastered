"""Run profiler.py and querygroq.py for EVERY table in a folder.

Each table's results go into their own folder:

    output/tables/<table name>/
        profiler_output.json                  (from profiler.py, full statistics)
        column_queries_groq.json              (from querygroq.py)
        column_queries_groq_non_derived.json  (from querygroq.py)

If a table is too big for Groq's free limit (8000 tokens per request), a shorter
version of the statistics is sent instead. Tables that are already finished are
skipped, so you can stop the script and start it again later.

Usage:
    Put GROQ_API_KEY=... in a .env file next to this script.
    python run_all_tables.py
"""

import argparse
import json
import os
import time
from pathlib import Path

from profiler import profile_csv
from querygroq import MODEL, describe_columns, load_dotenv, non_derived_column_queries

FINAL_FILE = "column_queries_groq_non_derived.json"  # written last, so it marks a finished table
MAX_LEVEL = 2  # how far the statistics may be shortened for big tables


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def shorten(value, max_length):
    """Cut long texts and round decimals, everywhere inside the statistics."""
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + "…"
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [shorten(item, max_length) for item in value]
    if isinstance(value, dict):
        return {key: shorten(item, max_length) for key, item in value.items()}
    return value


def shrink_stats(stats, level):
    """Level 0: full statistics. Level 1: shorter. Level 2: also without sample rows."""
    if level == 0:
        return stats
    small = shorten(json.loads(json.dumps(stats, default=str)), 30)
    for column in small["columns"].values():
        column["examples"] = column.get("examples", [])[:2]
        if "tfidf_top_terms" in column:
            column["tfidf_top_terms"] = dict(list(column["tfidf_top_terms"].items())[:3])
    small["sample_row_groups"] = [
        {**group, "rows": group["rows"][:2]} for group in small.get("sample_row_groups", [])[:1]
    ]
    if level >= 2:
        small["sample_row_groups"] = []
    return small


def is_too_large(error) -> bool:
    text = str(error)
    return "413" in text or "Request too large" in text


def process_table(csv_path: Path, out_dir: Path, model: str) -> None:
    """Profiler, then querygroq, for one table."""
    stats = profile_csv(str(csv_path))
    write_json(out_dir / "profiler_output.json", stats)

    for level in range(MAX_LEVEL + 1):
        try:
            columns = describe_columns(shrink_stats(stats, level), model)
            break
        except Exception as error:
            if is_too_large(error) and level < MAX_LEVEL:
                print(f"  too large for Groq, retrying with shorter statistics (level {level + 1})")
                time.sleep(60)  # let the per-minute token budget refill
                continue
            raise

    write_json(out_dir / "column_queries_groq.json", columns)
    write_json(out_dir / FINAL_FILE, non_derived_column_queries(columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run profiler and querygroq on every CSV table in a folder.")
    parser.add_argument("--input-dir", default="data/imputed",
                        help="Folder with the table CSVs (default: data/imputed)")
    parser.add_argument("--output-dir", default="output/tables",
                        help="One sub-folder per table is created here (default: output/tables)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--pause", type=float, default=10,
                        help="Seconds to wait between tables, to stay under the Groq rate limit")
    parser.add_argument("--force", action="store_true",
                        help="Redo tables that are already finished")
    args = parser.parse_args()

    load_dotenv()
    if "GROQ_API_KEY" not in os.environ:
        raise SystemExit("Set GROQ_API_KEY in .env or the environment.")

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    done, skipped, failed = [], [], []
    for number, csv_path in enumerate(csv_files, start=1):
        table = csv_path.stem
        out_dir = output_dir / table
        print(f"[{number}/{len(csv_files)}] {table}")

        if (out_dir / FINAL_FILE).exists() and not args.force:
            print("  already finished, skipping")
            skipped.append(table)
            continue

        try:
            process_table(csv_path, out_dir, args.model)
            print(f"  done -> {out_dir}")
            done.append(table)
        except Exception as error:  # one broken table should not stop the others
            print(f"  FAILED: {error}")
            failed.append({"table": table, "error": str(error)})

        if number < len(csv_files):
            time.sleep(args.pause)

    print(f"\nFinished: {len(done)} new, {len(skipped)} skipped, {len(failed)} failed")
    for item in failed:
        print(f"  {item['table']}: {item['error']}")
    write_json(output_dir / "run_summary.json", {"done": done, "skipped": skipped, "failed": failed})


if __name__ == "__main__":
    main()