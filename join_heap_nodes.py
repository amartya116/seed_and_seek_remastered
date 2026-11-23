"""Join the tables of every heap node and write a join report per node.

    python join_heap_nodes.py --seed data/List_of_countries_by_life_expectancy__t4.csv

For each node of the heap (output/ranked_groq_table_matrices.json) the node's tables are read from
DuckDB (data/seedandseek.duckdb, opened READ-ONLY: nothing in the database is changed) and joined in
memory on their entity key with every join type: inner, left, right, outer (for 3+ tables the same type
is applied left to right in heap order).

Join keys come from the data only (DuckDB is the truth): every text column of a source table is
normalised (case, accents, punctuation, [footnotes], parenthetical notes; no hand-made aliases) and the
column containing the most seed entities is the table's key. A table where no column reaches
--min-key-overlap of the seed's entities is not joinable, and nodes containing it are not joined.
The seed's key is its entity column from the seed relations (column without belongs_to).

Which source column stands for which seed column: for each non-derived seed column, the node table and
column with the highest score in the matrix (column_scores).

Per node: key overlap between the node's tables and with the seed, and per join type: rows, distinct
keys, seed entities matched, duplicate keys (fan-out), NULL % per bound column, belongs_to violations
(an entity with two different values), and agreement with the seed's own values.
Output: output/join_report.json and output/join_report.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd

JOIN_TYPES = ('inner', 'left', 'right', 'outer')


def normalize_key(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = unicodedata.normalize('NFKD', str(value))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r'\[[^\]]*\]|\([^)]*\)', ' ', text)
    text = re.sub(r'[^0-9A-Za-z]+', ' ', text).strip().lower()
    return text or None


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table_key(records):
    return next((r['column'] for r in records if not r.get('belongs_to')), None)


def detect_key(df: pd.DataFrame, seed_keys: set):
    """Pick the text column whose normalised values contain the most seed entities.

    Returns (column or None, share of seed entities found, {column: share} for every text column)."""
    shares = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        values = set(df[col].map(normalize_key).dropna())
        shares[col] = len(values & seed_keys) / max(len(seed_keys), 1)
    if not shares:
        return None, 0.0, {}
    best = max(shares, key=lambda c: shares[c])
    return best, shares[best], {c: round(100 * s, 1) for c, s in sorted(shares.items(), key=lambda kv: -kv[1])}


def bind_columns(node_ids, seed_cols, entity, scores, keys):
    """Seed column -> (table, source column, score): best scoring column among the node's tables."""
    binding = {}
    for col in seed_cols:
        best = None
        for tid in node_ids:
            cands = scores[col][tid].items()
            if col == entity:  # the entity is the key found in the data, whatever its score
                cands = [(keys[tid], scores[col][tid].get(keys[tid], 0.0))]
            for c, s in cands:
                if best is None or s > best[2]:
                    best = (tid, c, s)
        if best:
            binding[col] = {'table': best[0], 'source_column': best[1], 'similarity': round(best[2], 2)}
    return binding


def values_agree(a: pd.Series, b: pd.Series):
    """Share of rows where seed and joined values agree (numeric: within 1% or 0.05)."""
    both = a.notna() & b.notna()
    if not both.any():
        return None
    x, y = a[both], b[both]
    xn, yn = pd.to_numeric(x, errors='coerce'), pd.to_numeric(y, errors='coerce')
    if xn.notna().all() and yn.notna().all():
        ok = (xn - yn).abs() <= (0.01 * xn.abs()).clip(lower=0.05)
    else:
        ok = x.map(normalize_key) == y.map(normalize_key)
    return round(100.0 * ok.mean(), 1)


def join_report(node, binding, frames, keys, seed_df, seed_key):
    tids = node
    seed_keys = set(seed_df['_key'].dropna())
    key_sets = {t: set(frames[t]['_key'].dropna()) for t in tids}

    overlap = {t: {'rows': len(frames[t]), 'distinct_keys': len(key_sets[t]),
                   'seed_keys_found_pct': round(100.0 * len(seed_keys & key_sets[t]) / max(len(seed_keys), 1), 1),
                   'unmatched_seed_keys_sample': sorted(seed_keys - key_sets[t])[:8]} for t in tids}
    pairs = []
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            inter = len(key_sets[a] & key_sets[b])
            pairs.append({'tables': [a, b], 'common_keys': inter,
                          f'pct_of_{a}': round(100.0 * inter / max(len(key_sets[a]), 1), 1),
                          f'pct_of_{b}': round(100.0 * inter / max(len(key_sets[b]), 1), 1)})

    # one frame per table with only its key and bound columns, named <table>.<column>
    parts = []
    for t in tids:
        cols = [v['source_column'] for v in binding.values()
                if v['table'] == t and v['source_column'] != keys[t] and v['source_column'] in frames[t]]
        cols = list(dict.fromkeys(cols))
        part = frames[t][['_key', keys[t], *cols]].rename(columns={c: f'{t}.{c}' for c in [keys[t], *cols]})
        parts.append(part)

    reused = {}
    for seed_col, v in binding.items():
        reused.setdefault((v['table'], v['source_column']), []).append(seed_col)
    reused = [{'source': f'{t}.{c}', 'seed_columns': cs} for (t, c), cs in reused.items() if len(cs) > 1]

    results = {}
    for how in JOIN_TYPES:
        joined = parts[0]
        for part in parts[1:]:
            joined = joined.merge(part, on='_key', how=how)
        counts = joined['_key'].value_counts()
        stats = {'rows': len(joined), 'distinct_keys': int(joined['_key'].nunique()),
                 'seed_entities_matched_pct': round(100.0 * len(seed_keys & set(joined['_key'].dropna()))
                                                    / max(len(seed_keys), 1), 1),
                 'duplicate_keys': int((counts > 1).sum()), 'columns': {}}
        with_seed = joined.merge(seed_df, on='_key', how='inner', suffixes=('', '__seed'))
        for seed_col, v in binding.items():
            name = f"{v['table']}.{v['source_column']}"
            if name not in joined:
                continue
            per_key = joined.dropna(subset=['_key']).groupby('_key')[name].nunique()
            stats['columns'][seed_col] = {
                'source': name,
                'null_pct': round(100.0 * joined[name].isna().mean(), 1) if len(joined) else None,
                'belongs_to_violations': int((per_key > 1).sum()),
                'agrees_with_seed_pct': values_agree(with_seed[seed_col], with_seed[name])
                if seed_col in with_seed and seed_col != seed_key else None,
            }
        stats['belongs_to_holds'] = all(c['belongs_to_violations'] == 0 for c in stats['columns'].values())
        results[how] = stats
    return {'key_overlap_with_seed': overlap, 'key_overlap_between_tables': pairs,
            'source_columns_used_twice': reused, 'joins': results}


def render_text(report):
    out = [f"JOIN REPORT  (seed: {report['seed']}, {report['seed_rows']} rows, key {report['seed_key']!r})",
           'DuckDB opened read-only; joins are done in memory.', '',
           f"Join keys found in the data (column holding the most seed entities; "
           f"joinable from {report['min_key_overlap_pct']}%):"]
    for t, k in sorted(report['tables'].items(), key=lambda kv: int(kv[0][1:])):
        flag = 'joinable' if k['joinable'] else 'NOT joinable'
        out.append(f"    {t:<4} {str(k['key'])!r:<30} {k['seed_entities_found_pct']:>5}% of seed entities  {flag}")
    out.append('')
    for node in report['nodes']:
        out.append(f"[#{node['rank']}] {' + '.join(node['tables'])}   heap avg {node['heap_average_score']}")
        if not node['joinable']:
            for t in node['not_joinable']:
                k = report['tables'][t]
                best = ', '.join(f"{c!r} {s}%" for c, s in list(k['candidates'].items())[:3]) or 'no text columns'
                out.append(f"    not joined: {t} has no column holding enough seed entities (best: {best})")
            out.append('')
            continue
        for t, o in node['key_overlap_with_seed'].items():
            out.append(f"    {t}: key {node['keys'][t]!r}, {o['rows']} rows, "
                       f"{o['seed_keys_found_pct']}% of seed entities found")
            if o['unmatched_seed_keys_sample'] and o['seed_keys_found_pct'] < 100:
                out.append(f"        not found e.g.: {', '.join(o['unmatched_seed_keys_sample'][:5])}")
        for p in node['key_overlap_between_tables']:
            a, b = p['tables']
            out.append(f"    {a} x {b}: {p['common_keys']} common keys "
                       f"({p[f'pct_of_{a}']}% of {a}, {p[f'pct_of_{b}']}% of {b})")
        for r in node['source_columns_used_twice']:
            out.append(f"    ! {r['source']} stands for {len(r['seed_columns'])} seed columns: "
                       f"{', '.join(r['seed_columns'])}")
        out.append(f"    {'join':<6} {'rows':>5} {'keys':>5} {'seed%':>6} {'dup':>4} {'avgNULL%':>8} "
                   f"{'belongs_to':>10} {'agree%':>7}")
        for how, s in node['joins'].items():
            cols = list(s['columns'].values())
            nulls = [c['null_pct'] for c in cols if c['null_pct'] is not None]
            agree = [c['agrees_with_seed_pct'] for c in cols if c['agrees_with_seed_pct'] is not None]
            out.append(f"    {how:<6} {s['rows']:>5} {s['distinct_keys']:>5} {s['seed_entities_matched_pct']:>6} "
                       f"{s['duplicate_keys']:>4} {(sum(nulls) / len(nulls) if nulls else 0):>8.1f} "
                       f"{'holds' if s['belongs_to_holds'] else 'BROKEN':>10} "
                       f"{(f'{sum(agree) / len(agree):.1f}' if agree else '-'):>7}")
        out.append('')
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', default='data/List_of_countries_by_life_expectancy__t4.csv')
    p.add_argument('--seed-queries', default='output/column_queries_groq.json')
    p.add_argument('--matrix', default='output/groq_query_score_matrix.json')
    p.add_argument('--heap', default='output/ranked_groq_table_matrices.json')
    p.add_argument('--queries-dir', default='data/queries/queries/groq')
    p.add_argument('--db', default='data/seedandseek.duckdb')
    p.add_argument('--top', type=int, default=None, help='only the first N heap nodes')
    p.add_argument('--min-key-overlap', type=float, default=0.3,
                   help='a table is joinable only if its best key column holds this share of seed entities')
    p.add_argument('--output', default='output/join_report.json')
    p.add_argument('--text', default='output/join_report.txt')
    args = p.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    seed_records = load_json(args.seed_queries)
    seed_cols = [r['column'] for r in seed_records if not r.get('is_derived')]
    seed_key = table_key([r for r in seed_records if not r.get('is_derived')])
    matrix, heap = load_json(args.matrix), load_json(args.heap)
    scores = {q['seed_column']: row for q, row in zip(matrix['queries'], matrix['column_scores'])}
    seed_cols = [c for c in seed_cols if c in scores]
    files = {t['id']: t['source_file'] for t in matrix['tables']}

    seed_df = pd.read_csv(args.seed)
    if seed_key not in seed_df:
        raise SystemExit(f'Seed entity column {seed_key!r} not found in {args.seed}')
    seed_df['_key'] = seed_df[seed_key].map(normalize_key)
    seed_keys = set(seed_df['_key'].dropna())

    nodes = heap['ranked'][:args.top] if args.top else heap['ranked']
    needed = {t['id'] for n in nodes for t in n['tables']}
    frames, keys, key_info = {}, {}, {}
    with duckdb.connect(args.db, read_only=True) as con:
        registry = dict(con.execute('SELECT table_id, db_table FROM table_registry').fetchall())
        for tid in sorted(needed):
            table_id = Path(files[tid]).stem
            if table_id not in registry:
                raise SystemExit(f'{table_id} is not in {args.db}; run store_duckdb.py')
            df = con.execute(f'SELECT * FROM "{registry[table_id]}"').df()
            key, share, shares = detect_key(df, seed_keys)
            joinable = key is not None and share >= args.min_key_overlap
            key_info[tid] = {'key': key, 'seed_entities_found_pct': round(100 * share, 1),
                             'joinable': joinable, 'candidates': shares}
            if joinable:
                keys[tid] = key
                df['_key'] = df[key].map(normalize_key)
            frames[tid] = df

    report = {'seed': args.seed, 'seed_rows': len(seed_df), 'seed_key': seed_key,
              'min_key_overlap_pct': round(100 * args.min_key_overlap, 1), 'tables': key_info, 'nodes': []}
    for node in nodes:
        ids = [t['id'] for t in node['tables']]
        entry = {'rank': node['rank'], 'tables': ids, 'heap_average_score': node['average_score'],
                 'keys': {t: key_info[t]['key'] for t in ids}}
        blocked = [t for t in ids if not key_info[t]['joinable']]
        if blocked:
            entry.update(joinable=False, not_joinable=blocked)
        else:
            binding = bind_columns(ids, seed_cols, seed_key, scores, keys)
            entry.update(joinable=True, binding=binding)
            entry.update(join_report(ids, binding, frames, keys, seed_df, seed_key))
        report['nodes'].append(entry)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n',
                                 encoding='utf-8')
    text = render_text(report)
    Path(args.text).write_text(text, encoding='utf-8')
    print(text)
    print(f'Written {args.output} and {args.text}')


if __name__ == '__main__':
    main()
