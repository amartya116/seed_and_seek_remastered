"""Find minimal table combinations that fulfil every query and rank their score matrices.

    python select_tables.py --matrix output/groq_query_score_matrix.json --threshold 60

Scores are cosine similarity x 100. A table fulfils a query when its score is >= --threshold.
Each result is a queries x tables sub-matrix. Only minimal covers are kept: every query meets
the threshold, and dropping any table would leave at least one query unfulfilled. Results are
ranked by mean per-query best score, highest first.
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', default='output/score_matrix.json')
    p.add_argument('--threshold', type=float, default=50,
                   help='minimum score (0-100) for a table to fulfil a query')
    p.add_argument('--top', type=int, default=None,
                   help='maximum ranked matrices to write (default: all minimal covers)')
    p.add_argument('--per-query', type=int, default=None,
                   help='optional cap on qualifying tables per query (default: consider all)')
    p.add_argument('--max-tables', type=int, default=None,
                   help='largest combination size (default: number of source tables)')
    p.add_argument('--output', default='output/ranked_table_matrices.json')
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
    ranked = ranked_matrices(scores, args.threshold, args.per_query, args.max_tables)
    table_ids = [table['id'] if isinstance(table, dict) else table for table in tables]
    out = {'matrix_source': args.matrix, 'threshold': args.threshold, 'score_scale': '0-100',
           'total_valid_combinations': len(ranked), 'search_truncated': False,
           'weak_queries': [], 'queries': [{k: queries[q][k] for k in ('id', 'seed_column', 'query')}
                                            for q in range(len(queries))],
           'ranked': [{'rank': i, 'tables': [tables[t] for t in r['tables']],
                       'average_score': round(r['average_score'], 2),
                       'matrix': [[round(s, 2) for s in row] for row in r['matrix']]}
                      for i, r in enumerate(ranked[:args.top], 1)]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"{len(ranked)} valid table combinations; saved {len(out['ranked'])} ranked results to {target}")
    qids = [q['id'] for q in out['queries']]
    for r in out['ranked']:
        labels = [table_ids[t] for t in ranked[r['rank'] - 1]['tables']]
        print(f"\n#{r['rank']}  avg {r['average_score']:.2f}  tables: {', '.join(labels)}")
        print('      ' + ''.join(f'{t[:12]:>14}' for t in labels))
        for qid, row in zip(qids, r['matrix']):
            print(f'{qid:>5} ' + ''.join(f'{s:>14.2f}' for s in row))


if __name__ == '__main__':
    main()
