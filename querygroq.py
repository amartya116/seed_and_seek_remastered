"""Describe every CSV column with Groq. Usage: python query_groq.py [--stats FILE] [--output FILE]"""

import argparse
import json
import os
import time

from pydantic import BaseModel
from groq import Groq

MODEL = "openai/gpt-oss-120b"


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Entity(BaseModel):
    name: str
    key_column: str
    parent: str | None


class Entities(BaseModel):
    entities: list[Entity]
    row_is_about: str


class Column(BaseModel):
    column: str
    description: str
    is_numeric: bool
    belongs_to: str | None
    subset_of: str | None
    is_derived: bool
    derived_from: list[str]
    derivation_formula: str | None
    query: str

class Columns(BaseModel):
    columns: list[Column]


ENTITY_PROMPT = """Below are per-column statistics of a CSV file and groups of whole sample rows. Each group's rows share one value (shared_column = shared_value).

{stats}

Pass 1: identify the entities (things the rows are about) and how they nest.
- key_column is the column that identifies the entity.
- parent is the entity it belongs to (e.g. city -> country), or null.
- row_is_about is the finest-grained entity a single row describes.
A column constant across a group's rows is a property of the group's shared entity, not of the finer entities."""

PROMPT = """You are given per-column statistics of a CSV file, groups of whole sample rows, and the entity hierarchy found in pass 1.

{stats}

Entities (pass 1):
{entities}

Pass 2: describe EVERY column, in original order. For each column return:
- column: the column name
- description: what the column holds, in plain English
- is_numeric: true if the values are numbers that count or measure something; false for names, categories, codes, IDs and dates
- belongs_to: the column that identifies the entity this value describes, i.e. whom the value is about. Applies to ANY kind of column, numeric or not. Null if the column is itself the name, key or identifier of an entity
- subset_of: the column representing the immediate broader population or scope that this column's measure covers. This is a semantic relationship between what the columns measure, not a claim that one column's numeric values are contained within another column's numeric values. Null if this column does not describe a strict subset of another column's scope.
- is_derived: true only if this column's values are calculated from one or more other columns in this table by a clear, explainable rule; otherwise false
- derived_from: the exact names of all input columns needed to calculate this column, in original table order. Use an empty list if is_derived is false. Do not include this column itself.
- derivation_formula: a concise formula for calculating this column from derived_from. Refer to each input as [exact column name] so names containing spaces or operator characters are unambiguous; null if is_derived is false
- query: a reusable natural-language question about the information in this column, phrased so it applies to any row and does not include a specific observed value

STRICT: belongs_to, subset_of, and every name in derived_from must be exact column names copied from the statistics above. Never invent names. belongs_to and subset_of must not refer to the column itself; derived_from must not contain the column itself.

Query-writing rules:
- Write `query` as a reusable question template, not a question about one particular sample row.
- Never copy a value from sample rows or statistics into the question. In particular, do not use actual country, person, company, product, place, date, ID, or other record values, even if one is prominent in the examples.
- Refer to the entity generically by its type when that type is clear from the entity hierarchy or column meaning (for example, "a customer" or "a country"). If its type is not clear, use a neutral placeholder such as "[entity]". Do not substitute a specific entity instance.
- Ask about the column's meaning and include its relevant subgroup, time, location, or other dimensions when needed to distinguish it from related columns. Keep the wording natural and generic.
- If `belongs_to` is null, formulate a generic question about the column itself; do not insert an observed value to make the question concrete.

Derivability rules:
- Mark a column derived only when the table's column names, descriptions, units, sample rows, or statistics provide clear evidence of a calculation from other columns. Similar names, correlation, or a plausible real-world relationship alone are not sufficient evidence.
- Include every input column needed for the calculation, and no unrelated columns. Use exact source column names. If an input cannot be identified confidently, do not guess: set is_derived to false, derived_from to [], and derivation_formula to null.
- Write derivation_formula as a concise, executable-style expression. Refer to each input as [exact column name] so names containing spaces or operator characters are unambiguous. Use parentheses to make operation order clear. Basic operators (+, -, *, /), percentages, and well-known functions such as AVG, SUM, MIN, and MAX may be used when supported by the column meaning. Include a constant only when the table's meaning or statistics support it.
- For a difference between two measures, express the formula as ([first source column] - [second source column]), preserving the direction indicated by the target column's meaning.
- For an average or other aggregate, identify the source column(s) and aggregation explicitly, for example AVG([source column]) or AVG([source column 1], [source column 2]) as appropriate. State whether it aggregates rows or combines columns only when the table context makes that clear; never imply an unsupported grouping or denominator.
- A descriptive statistic, index, ratio, difference, change, percentage, or total can be derived even if its inputs have different names. Conversely, a column is not derived merely because its values happen to correlate with another column.
- A source column may itself be derived; list the direct input columns needed by the target formula, not every upstream ancestor, unless the formula explicitly uses those ancestors.
- Ensure is_derived, derived_from, and derivation_formula agree: true requires at least one valid source column and a non-null formula; false requires an empty list and a null formula.

Judge belongs_to and subset_of independently. A column can have one, both, or neither. `belongs_to` says what entity a value describes; `subset_of` says whether the column describes a narrower population or scope than another column. Do not use `subset_of` to point to an entity identifier just because the column is about that entity.

For `subset_of`, compare the meaning of each measure, using its name, description, and sample data:
- First identify the measure and all of its dimensions, such as subject, age, sex, location, and time period.
- Two columns are counterparts when they measure the same thing and have the same dimensions, except that one covers a narrower population or scope.
- When a measure is for a subgroup and a corresponding measure exists for the whole population, set the subgroup column's `subset_of` to the exact name of the whole-population column. This applies when the column names or descriptions indicate a category, demographic, geographic, organizational, or other subset versus a total, overall, all-groups, or unfiltered value. Infer the relationship from the table's own column names, descriptions, and statistics; do not rely on any particular domain, naming convention, abbreviation, or example. Only match columns that measure the same underlying thing and have the same other dimensions, with the subgroup/whole-population scope as the sole difference.
- REQUIRED PAIRWISE AUDIT: Before deciding that `subset_of` is null for a measure, compare it with every other column. Look for columns with the same measure and same non-scope dimensions but a different population/scope, including columns whose names share a common stem and differ by a short suffix, category label, or total/all label. Interpret those labels using the descriptions, units, and sample values; do not assume a particular abbreviation has a fixed meaning. If one column clearly measures a subgroup and another clearly measures the matching whole, assign the subgroup's `subset_of` to the whole column. Do not leave it null merely because the relation is not stated in one exact phrase.
- Apply the audit symmetrically across the table: if there is a broader-total column and multiple matching subgroup columns, evaluate each subgroup independently and link each clearly matching subgroup to that same broader column. Do not create links among peer subgroups.
- If the table has separate subgroup columns and a total/overall counterpart for a repeated measure, treat this as strong evidence of a scope hierarchy: the subgroup columns point to the total counterpart, while the total has no parent for that dimension.
- The broader column does not point back to the subgroup; its `subset_of` is null for this relationship.
- Match every dimension other than the subgroup. Do not link measures at different ages, locations, or time periods, or different underlying metrics, just because one sounds broader.
- Prefer the most specific matching broader counterpart. If no corresponding broader column exists in this table, use null; do not invent a name.
- A difference, change, ratio, average, or other derived measure is not a subgroup of the measure it was calculated from merely because it uses that measure. Use null unless it independently represents a narrower population/scope and has a matching broader counterpart.

Illustrations (use only column names that really exist in this table):
- gender describes a person -> belongs_to the person's id column, subset_of null
- population describes a country -> belongs_to the country column, subset_of null
- city is part of a state -> belongs_to null, subset_of the state column
- airport_code is part of an airport -> belongs_to null, subset_of the airport column
- A measure for one region compared with the same measure for the whole country -> the regional measure's `subset_of` is the whole-country measure
- A measure for one category compared with the same measure across all categories -> the category measure's `subset_of` is the all-categories measure
- In both cases, the broader measure's `subset_of` is null; do not create a reciprocal link
- A target measure explicitly defined as the difference between two other measures -> `is_derived: true`, `derived_from` contains those two exact source column names, and `derivation_formula` subtracts them in the stated direction using bracketed column references
- A target measure explicitly defined as the average of values from one source column -> `is_derived: true`, `derived_from` contains that exact source column, and `derivation_formula` uses `AVG` on its bracketed column reference
- If the table does not clearly identify a calculation and its inputs -> `is_derived: false`, `derived_from: []`, and `derivation_formula: null`

How to decide:
- A value belongs to the entity it describes, not to whatever happens to sit next to it in the row. Population describes a country, not its capital city, even though both appear in the same row.
- Use the sample row groups. If a column has the SAME value in every row of a group, it belongs to the group's shared (coarser) entity. If it DIFFERS between rows, it belongs to the finer entity. Mixed cases are possible, so look at the whole row before deciding.
- Sanity-check with real-world knowledge: ask whether the value truly belongs to the entity, not just whether it is unique to it.
- Before finalizing the output, re-check all `subset_of: null` values against the full set of columns and correct any clear subgroup-to-total matches. Check the converse too: a total/overall column should not point to a narrower subgroup.
- For population/scope relationships, subset_of names the immediate broader matching column only; never skip an available direct counterpart.
- If no suitable column exists in this table, use null."""


def ask(client, prompt, schema, model):
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__.lower(), "schema": schema.model_json_schema()},
                },
                temperature=0,
                reasoning_effort="medium",
                max_completion_tokens=8192,
            )
            return schema.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            if not ("429" in str(e) or "json_validate_failed" in str(e)) or attempt == 5:
                raise
            time.sleep(3 * 2**attempt)


def describe_columns(stats, model=MODEL):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    stats_text = json.dumps(stats, indent=2, ensure_ascii=False)
    entities = ask(client, ENTITY_PROMPT.format(stats=stats_text), Entities, model)
    prompt = PROMPT.format(stats=stats_text, entities=entities.model_dump_json(indent=2))
    columns = ask(client, prompt, Columns, model).columns
    names = {c.column for c in columns}
    for c in columns:
        valid = names - {c.column}
        if c.belongs_to not in valid:
            c.belongs_to = None
        if c.subset_of not in valid:
            c.subset_of = None
        source_names = set(c.derived_from) & valid
        c.derived_from = [column.column for column in columns if column.column in source_names]
        if not c.is_derived or not c.derived_from or not c.derivation_formula:
            c.is_derived = False
            c.derived_from = []
            c.derivation_formula = None
    return [c.model_dump() for c in columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default="output/profiler_output.json")
    parser.add_argument("--output", "-o", default="output/column_queries_groq.json")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    load_dotenv()
    if "GROQ_API_KEY" not in os.environ:
        raise SystemExit("Set GROQ_API_KEY in .env or the environment.")

    with open(args.stats, encoding="utf-8") as f:
        stats = json.load(f)

    text = json.dumps(describe_columns(stats, args.model), indent=2, ensure_ascii=False)
    print(text)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()