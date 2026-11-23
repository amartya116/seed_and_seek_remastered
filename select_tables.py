"""Find minimal table combinations that fulfil every query and rank them in a max-heap.

    python select_tables.py --matrix output/groq_query_score_matrix.json --threshold 60 \
        --output output/ranked_groq_table_matrices.json --heap-txt output/groq_table_set_max_heap.txt

Scores are cosine similarity x 100. Two coverage modes:

  queries   (default) candidates: a table fulfils a query when its best score is >= threshold. Every
            query must be fulfilled and no table may be redundant. The candidates are then joined and
            checked on real rows by join_heap_nodes.py.

  equation  A table set is valid when every non-derived seed column is bound to its own source column
            (score >= threshold, one-to-one), each subset_of group of the seed (e.g. {All, M, F}) is bound
            inside ONE table with the same subset_of shape, every bound column belongs_to its table's key,
            and every table's key matches the seed entity. Relations come from --seed-queries and the
            source query JSONs in --queries-dir. Derived seed columns are computed from bound inputs.

Only minimal sets are kept: dropping any table makes the set invalid. Results are ranked by the mean
score per seed column, highest first, and also written as a max-heap tree (--heap-txt).
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
from pathlib import Path

from table_equations import (SeedModel, SourceTable, best_binding, build_options, column_scores,
                             computable_derived, load_json)


def candidates(scores, threshold, per_query=None):
    """Return all threshold-qualified tables per query, optionally capped by rank."""
    cand = []
    for row in scores:
        ranked = sorted(range(len(row)), key=lambda t: -row[t])
        qualified = [t for t in ranked if row[t] >= threshold]
        cand.append(qualified[:per_query] if per_query else qualified)
    return cand


def minimal_covers(cand, max_tables, limit):
    """Enumerate all minimal table covers; fail rather than return a partial result."""
    n_q = len(cand)
    covers = {}
    for q, ts in enumerate(cand):
        for t in ts:
            covers.setdefault(t, set()).add(q)
    found, visits = set(), 0

    def search(chosen, uncovered):
        nonlocal visits
        visits += 1
        if visits > limit:
            raise RuntimeError(
                f"Search exceeded {limit} states; refusing to return an incomplete combination list"
            )
        if not uncovered:
            everything = set(range(n_q))
            if all(set().union(set(), *(covers[u] for u in chosen if u != t)) != everything for t in chosen):
                found.add(frozenset(chosen))
            return
        if len(chosen) >= max_tables:
            return
        q = min(uncovered, key=lambda query: sum(t not in chosen for t in cand[query]))
        if not cand[q]:
            return
        for t in cand[q]:
            if t not in chosen:
                search(chosen + [t], uncovered - covers[t])

    search([], set(range(n_q)))
    return found


def ranked_matrices(scores, threshold, per_query=None, max_tables=None, limit=200000):
    """Rank all minimal-cover matrices with a max-priority heap."""
    n_q = len(scores)
    cand = candidates(scores, threshold, per_query)
    uncovered = [q for q, tables in enumerate(cand) if not tables]
    if uncovered:
        raise ValueError(
            f"No table reaches threshold {threshold:g} for query row(s): "
            + ", ".join(f"q{q + 1}" for q in uncovered)
        )

    max_tables = max_tables if max_tables is not None else len(scores[0])
    found = minimal_covers(cand, max_tables, limit)
    heap = []
    for tables in found:
        cols = sorted(tables)
        sub = [[scores[q][t] for t in cols] for q in range(n_q)]
        avg = sum(max(row) for row in sub) / n_q
        result = {'tables': cols, 'matrix': sub, 'average_score': avg}
        heapq.heappush(heap, (-avg, len(cols), cols, result))
    results = [heapq.heappop(heap)[3] for _ in range(len(heap))]
    for r in results:
        assert all(set(ts) & set(r['tables']) for ts in cand), 'combination leaves a query unfulfilled'
    return results


def equation_covers(seed, tables, scores, threshold, max_tables=None, top_k=3, limit=200000):
    """All minimal table sets that satisfy the seed equation, best average first."""
    options, usable = build_options(seed, tables, scores, threshold, top_k)
    missing = [seed.groups[g] for g, by_table in enumerate(options) if not by_table]
    if missing:
        raise ValueError(f"No joinable table binds seed group(s) at threshold {threshold:g}: "
                         + "; ".join("[" + ", ".join(g) + "]" for g in missing))
    candidates = [tid for tid in usable if any(tid in by_table for by_table in options)]
    max_tables = max_tables or len(candidates)
    found, checked = [], 0
    for size in range(1, max_tables + 1):
        for combo in itertools.combinations(candidates, size):
            chosen = set(combo)
            if any(f['set'] <= chosen for f in found):
                continue
            checked += 1
            if checked > limit:
                raise RuntimeError(f"Checked more than {limit} table sets; refusing to return a partial list")
            result = best_binding(combo, options)
            if result:
                total, parts = result
                binding = {c: {'table': tid, 'source_column': sc,
                               'similarity': round(scores[c][tid][sc], 2)}
                           for _, tid, b in parts for c, sc in b.items()}
                found.append({'set': chosen, 'tables': list(combo), 'binding': binding,
                              'average_score': total / len(seed.columns)})
    heap = [(-f['average_score'], len(f['tables']), f['tables'], i) for i, f in enumerate(found)]
    heapq.heapify(heap)
    return [found[heapq.heappop(heap)[3]] for _ in range(len(heap))]


def heap_tree(ranked, labels, threshold, priority):
    """Render ranked results as an array max-heap: node i has children 2i and 2i+1."""
    lines = ["MAX-HEAP TREE OF MINIMAL TABLE SETS",
             f"Threshold: {threshold:g}/100 | Total minimal covers: {len(ranked)}",
             f"Priority: {priority}", ""]

    def name(i):
        r = ranked[i - 1]
        return f"[#{i}] {' + '.join(labels[i - 1])} | avg {r['average_score']:.2f}"

    def walk(i, prefix):
        kids = [k for k in (2 * i, 2 * i + 1) if k <= len(ranked)]
        for n, k in enumerate(kids):
            last = n == len(kids) - 1
            lines.append(f"{prefix}{'`-- ' if last else '|-- '}{name(k)}")
            walk(k, prefix + ('    ' if last else '|   '))

    if ranked:
        lines.append(name(1))
        walk(1, '')
    return "\n".join(lines) + "\n"


def run_equation_mode(args, data, tables, table_ids):
    """Equation-aware covers, returned in the same shape as ranked_matrices() plus the binding."""
    seed = SeedModel(load_json(args.seed_queries))
    rows = [q['seed_column'] for q in data['queries']]
    if sorted(rows) != sorted(seed.columns):
        raise ValueError('Matrix rows must be exactly the non-derived seed columns of --seed-queries; '
                         f'only in matrix: {sorted(set(rows) - set(seed.columns))}, '
                         f'only in seed: {sorted(set(seed.columns) - set(rows))}')
    sources = [SourceTable(tid, t['source_file'], load_json(Path(args.queries_dir) / t['source_file']))
               for tid, t in zip(table_ids, tables)]
    covers = equation_covers(seed, sources, column_scores(data), args.threshold, args.max_tables, args.top_k)
    index = {tid: i for i, tid in enumerate(table_ids)}
    sims = data['similarities']
    ranked = []
    for c in covers:
        cols = sorted(index[t] for t in c['tables'])
        computed, missing = computable_derived(seed, c['binding'])
        ranked.append({'tables': cols, 'average_score': c['average_score'],
                       'matrix': [[sims[q][t] for t in cols] for q in range(len(rows))],
                       'binding': c['binding'], 'computed': computed, 'not_computable': missing})
    return ranked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', default='output/groq_query_score_matrix.json')
    p.add_argument('--threshold', type=float, default=50,
                   help='minimum score (0-100) for a table to fulfil a query')
    p.add_argument('--top', type=int, default=None,
                   help='maximum ranked matrices to write (default: all minimal covers)')
    p.add_argument('--per-query', type=int, default=None,
                   help='optional cap on qualifying tables per query (default: consider all)')
    p.add_argument('--max-tables', type=int, default=None,
                   help='largest combination size (default: number of source tables)')
    p.add_argument('--output', default='output/ranked_groq_table_matrices.json')
    p.add_argument('--heap-txt', default='output/groq_table_set_max_heap.txt', help='max-heap tree as text')
    p.add_argument('--mode', choices=('queries', 'equation', 'auto'), default='queries',
                   help='queries: score-based candidates (default); equation: only equation-valid sets')
    p.add_argument('--seed-queries', default='output/column_queries_groq.json',
                   help='seed column records with belongs_to / subset_of / is_derived (equation mode)')
    p.add_argument('--queries-dir', default='data/queries/queries/groq',
                   help='one query JSON per source table (equation mode)')
    p.add_argument('--top-k', type=int, default=3,
                   help='candidate source columns per seed column and table (equation mode)')
    args = p.parse_args()
    if (args.top is not None and args.top < 1) or (args.per_query is not None and args.per_query < 1):
        p.error('--top and --per-query must be positive')
    data = json.loads(Path(args.matrix).read_text(encoding='utf-8'))
    tables, queries = data['tables'], data['queries']
    if 'similarities' in data:
        scores = data['similarities']
    elif 'scores' in data:
        scores = [[100 * s for s in row] for row in data['scores']]
    else:
        raise ValueError("Matrix JSON must contain either 'similarities' or 'scores'")
    if not scores or not tables:
        raise ValueError('Score matrix is empty')
    table_ids = [table['id'] if isinstance(table, dict) else table for table in tables]
    mode = args.mode
    if mode == 'auto':
        mode = 'equation' if 'column_scores' in data else 'queries'
    if mode == 'equation':
        ranked = run_equation_mode(args, data, tables, table_ids)
    else:
        ranked = ranked_matrices(scores, args.threshold, args.per_query, args.max_tables)
    extra = ('binding', 'computed', 'not_computable') if mode == 'equation' else ()
    out = {'matrix_source': args.matrix, 'mode': mode, 'threshold': args.threshold, 'score_scale': '0-100',
           'total_valid_combinations': len(ranked), 'search_truncated': False,
           'weak_queries': [], 'queries': [{k: queries[q][k] for k in ('id', 'seed_column', 'query')}
                                            for q in range(len(queries))],
           'ranked': [{'rank': i, 'tables': [tables[t] for t in r['tables']],
                       'average_score': round(r['average_score'], 2),
                       'matrix': [[round(s, 2) for s in row] for row in r['matrix']],
                       **{k: r[k] for k in extra}}
                      for i, r in enumerate(ranked[:args.top], 1)]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"[{mode} mode] {len(ranked)} valid table combinations; "
          f"saved {len(out['ranked'])} ranked results to {target}")
    if args.heap_txt:
        labels = [[Path(tables[t]['source_file']).stem if isinstance(tables[t], dict) else str(tables[t])
                   for t in r['tables']] for r in ranked[:args.top]]
        priority = ('mean score per non-derived seed column, equation-valid sets only' if mode == 'equation'
                    else "average of each query's best similarity score (higher is better)")
        Path(args.heap_txt).write_text(heap_tree(out['ranked'], labels, args.threshold, priority),
                                       encoding='utf-8')
        print(f"Heap tree written to {args.heap_txt}")
    qids = [q['id'] for q in out['queries']]
    for r in out['ranked']:
        labels = [table_ids[t] for t in ranked[r['rank'] - 1]['tables']]
        print(f"\n#{r['rank']}  avg {r['average_score']:.2f}  tables: {', '.join(labels)}")
        print('      ' + ''.join(f'{t[:12]:>14}' for t in labels))
        for qid, row in zip(qids, r['matrix']):
            print(f'{qid:>5} ' + ''.join(f'{s:>14.2f}' for s in row))


if __name__ == '__main__':
    main()
