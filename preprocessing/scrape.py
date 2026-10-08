import json
import re
import time
from io import StringIO
from pathlib import Path

import lxml.html
import pandas as pd
import requests

# Display settings: only change how tables look when printed in the run window
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 200)

URLS = [
    "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population",
    "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)",
    "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_area",
    "https://en.wikipedia.org/wiki/List_of_national_capitals",
    "https://en.wikipedia.org/wiki/List_of_countries_by_life_expectancy",
]

RAW_DIR = Path("data/raw")
TABLES_DIR = Path("data/tables")

HEADERS = {
    "User-Agent": "SeedandSeek/0.1 (your.email@example.com)"
}


def download_page(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def page_name(url):
    return url.split("/")[-1]

def save_html(name, html):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DIR / f"{name}.html"
    file_path.write_text(html, encoding="utf-8")
    return file_path

def get_page_info(html):
    title_match = re.search(r'"wgTitle":"(.*?)"', html)
    revision_match = re.search(r'"wgRevisionId":(\d+)', html)

    title = title_match.group(1) if title_match else None
    revision_id = int(revision_match.group(1)) if revision_match else None

    return title, revision_id

def find_wikitables(html):
    tree = lxml.html.fromstring(html)

    for element in tree.xpath("//style | //script"):
        element.drop_tree()

    return tree.xpath(
        "//table[contains(concat(' ', normalize-space(@class), ' '), ' wikitable ')]"
    )


def get_caption(table):
    caption = table.find("caption")
    if caption is None:
        return None
    return " ".join(caption.text_content().split())


def get_section(table):
    headings = table.xpath("preceding::*[self::h2 or self::h3 or self::h4][1]")
    if not headings:
        return None
    text = " ".join(headings[0].text_content().split())
    return text.replace("[edit]", "").strip()

def flatten_columns(df):
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    new_columns = []
    for levels in df.columns:
        parts = []
        for part in levels:
            part = str(part).strip()
            if part.startswith("Unnamed") or part in parts:
                continue
            parts.append(part)
        new_columns.append(" - ".join(parts))

    df.columns = new_columns
    return df


def table_to_dataframe(table):
    table_html = lxml.html.tostring(table, encoding="unicode")
    df = pd.read_html(StringIO(table_html), flavor="lxml")[0]
    return flatten_columns(df)


def extract_tables(html, name, url):
    title, revision_id = get_page_info(html)
    results = []

    for i, table in enumerate(find_wikitables(html)):
        df = table_to_dataframe(table)
        results.append({
            "table_id": f"{name}__t{i}",
            "page_title": title,
            "source_url": url,
            "revision_id": revision_id,
            "section": get_section(table),
            "caption": get_caption(table),
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "dataframe": df,
        })

    return results

def save_tables(tables):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for t in tables:
        file_path = TABLES_DIR / f"{t['table_id']}.csv"
        t["dataframe"].to_csv(file_path, index=False, encoding="utf-8")
        print(f"  Saved {t['table_id']} ({t['n_rows']} rows) | {t['caption']}")


def save_metadata(all_tables):
    metadata = [{k: v for k, v in t.items() if k != "dataframe"} for t in all_tables]
    file_path = TABLES_DIR / "tables_metadata.json"
    file_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata for {len(metadata)} tables to {file_path}")

def main():
    all_tables = []

    for url in URLS:
        name = page_name(url)
        print(f"Processing {name}")

        html = download_page(url)
        save_html(name, html)

        tables = extract_tables(html, name, url)
        save_tables(tables)
        all_tables.extend(tables)

        time.sleep(1)

    save_metadata(all_tables)
    print("Done")


if __name__ == "__main__":
    main()