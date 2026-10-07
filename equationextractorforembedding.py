"""Convert column ownership and subset links into embedding-friendly predicates."""

import argparse
import json
import os


RELATIONS = ("belongs_to", "subset_of")


def build_equations(columns):
    if not isinstance(columns, list):
        raise ValueError("Input JSON must contain a list of column descriptions.")

    column_names = []
    for index, column in enumerate(columns):
        if not isinstance(column, dict) or not isinstance(column.get("column"), str):
            raise ValueError(f"Column record at index {index} must have a string 'column' name.")
        column_names.append(column["column"])

    known_columns = set(column_names)
    predicates = []
    for column in columns:
        column_name = column["column"]
        for relation in RELATIONS:
            target = column.get(relation)
            if target is None:
                continue
            if not isinstance(target, str) or target not in known_columns:
                raise ValueError(
                    f"Invalid {relation} target for column {column_name!r}: {target!r}"
                )
            if target == column_name:
                raise ValueError(f"Column {column_name!r} cannot refer to itself via {relation}.")

            predicates.append(
                f"{relation}({json.dumps(column_name, ensure_ascii=False)}, "
                f"{json.dumps(target, ensure_ascii=False)})"
            )

    table_columns = list(dict.fromkeys(column_names))
    table_term = "Table(columns=[" + ", ".join(
        json.dumps(name, ensure_ascii=False) for name in table_columns
    ) + "])"
    equation = " AND ".join([table_term, *predicates])
    return {
        "table_columns": table_columns,
        "equation": equation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/column_queries_groq.json")
    parser.add_argument("--output", default="output/column_relationship_equations.json")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as file:
        columns = json.load(file)

    equations = build_equations(columns)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(equations, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Wrote one table-wide relationship equation to {args.output}")


if __name__ == "__main__":
    main()