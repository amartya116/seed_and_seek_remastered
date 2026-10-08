import json
from pathlib import Path

import numpy as np
import pandas as pd

# Display settings: only change how tables look when printed, not the data
pd.set_option("display.max_columns", None)  # show all columns instead of hiding some with "..."
pd.set_option("display.width", 200)         # allow wider lines so tables don't wrap too early

CLEAN_DIR = Path("data/clean")
IMPUTED_DIR = Path("data/imputed")

MIN_NUMBER_COLUMNS = 2   # TabPFN needs at least 2 number columns to learn from
MIN_ROWS = 5             # too few complete rows → no reliable outlier scores
N_PERMUTATIONS = 3       # more = more robust, but slower
TOP_OUTLIERS = 5         # how many unusual rows to report per table


# ---------- Loading ----------

def load_clean_metadata():
    return json.loads((CLEAN_DIR / "clean_metadata.json").read_text(encoding="utf-8"))


def load_clean_table(table_id):
    return pd.read_csv(CLEAN_DIR / f"{table_id}.csv")


def get_number_columns(meta):
    return [c["name"] for c in meta["columns"] if c["role"] == "number"]


def get_key_column(meta):
    keys = [c["name"] for c in meta["columns"] if c["role"] == "key"]
    return keys[0] if keys else None


def count_gaps(df, number_columns):
    return int(df[number_columns].isna().sum().sum())


# ---------- TabPFN helpers ----------

def build_model():
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    return TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(),
        tabpfn_reg=TabPFNRegressor(),
    )


def to_numpy(result):
    if hasattr(result, "detach"):
        result = result.detach().cpu()
    return np.asarray(result, dtype=float)


def to_model_scale(X):
    positive = np.array([
        not np.isnan(X[:, i]).all() and np.nanmin(X[:, i]) > 0
        for i in range(X.shape[1])
    ])
    X_model = X.copy()
    X_model[:, positive] = np.log(X[:, positive])
    return X_model, positive


# ---------- Imputation ----------

def impute_table(df, number_columns, n_permutations=N_PERMUTATIONS):
    X = df[number_columns].to_numpy(dtype=float)
    missing = np.isnan(X)
    if not missing.any():
        return df, 0

    has_any_value = ~missing.all(axis=1)
    to_fill = missing & has_any_value[:, None]

    X_model, positive = to_model_scale(X)

    model = build_model()
    model.fit(X_model)
    imputed = to_numpy(model.impute(X_model, n_permutations=n_permutations))
    imputed[:, positive] = np.exp(imputed[:, positive])

    df = df.copy()
    df[number_columns] = np.where(to_fill, imputed, X)
    return df, int(to_fill.sum())


# ---------- Outlier detection ----------

def find_outliers(df, number_columns, key_column, top_n=TOP_OUTLIERS, n_permutations=N_PERMUTATIONS):
    X = df[number_columns].to_numpy(dtype=float)
    complete = ~np.isnan(X).any(axis=1)
    if complete.sum() < MIN_ROWS:
        return []

    X = X[complete]
    keys = df.loc[complete, key_column].tolist()

    X_model, _ = to_model_scale(X)

    model = build_model()
    model.fit(X_model)
    scores = to_numpy(model.outliers(X_model, n_permutations=n_permutations))

    most_unusual = np.argsort(scores)[:top_n]
    return [
        {
            "key": str(keys[i]),
            "score": float(scores[i]),
            "values": {col: float(X[i, j]) for j, col in enumerate(number_columns)},
        }
        for i in most_unusual
    ]


# ---------- Main: all tables ----------

def main():
    IMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    outlier_report = {}
    imputed_metadata = []

    for meta in load_clean_metadata():
        table_id = meta["table_id"]
        df = load_clean_table(table_id)
        number_columns = get_number_columns(meta)
        key_column = get_key_column(meta)

        if len(number_columns) >= MIN_NUMBER_COLUMNS and key_column:
            print(f"Processing {table_id} ({len(number_columns)} number columns)")
            df, n_filled = impute_table(df, number_columns)
            outliers = find_outliers(df, number_columns, key_column)
            outlier_report[table_id] = outliers
            print(f"  filled {n_filled} cells, most unusual: {[o['key'] for o in outliers]}")
        else:
            print(f"Copied {table_id} unchanged (not enough number columns)")

        df.to_csv(IMPUTED_DIR / f"{table_id}.csv", index=False, encoding="utf-8")
        imputed_metadata.append(meta)

    (IMPUTED_DIR / "imputed_metadata.json").write_text(
        json.dumps(imputed_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (IMPUTED_DIR / "outliers.json").write_text(
        json.dumps(outlier_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {len(imputed_metadata)} tables and the outlier report to {IMPUTED_DIR}")


if __name__ == "__main__":
    main()