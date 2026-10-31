"""List every table combination that fulfils all queries, as score matrices ranked best-first.

    python select_tables.py --matrix output/score_matrix.json --threshold 50 --top 10

Scores are cosine similarity x 100. A table fulfils a query when its score is >= --threshold.
Each result is a queries x tables sub-matrix whose tables together fulfil every query, with no
table that could be dropped (a minimal cover). Results are sorted by average score, highest
first: each query counts its best score within the sub-matrix, and those are averaged. A query that no table reaches the threshold for falls back to
its best-scoring table and is flagged weak, so every query is always covered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def candidates(scores, threshold, per_query):
    """Per query, the tables allowed to fulfil it: top scorers at/above threshold, else its single best table."""
    cand = []
    for row in scores:
        ranked = sorted(range(len(row)), key=lambda t: -row[t])
        ok = [t for t in ranked if row[t] >= threshold][:per_query]
        cand.append(ok or ranked[:1])
    return cand


def minimal_covers(cand, max_tables, limit):
    """All table sets (frozensets) covering every query where no table is redundant, up to max_tables each."""
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
            return
        if not uncovered:
            everything = set(range(n_q))
            if all(set().union(set(), *(covers[u] for u in chosen if u != t)) != everything for t in chosen):
                found.add(frozenset(chosen))
            return
        if len(chosen) >= max_tables:
            return
        q = min(uncovered)
        for t in cand[q]:
            if t not in chosen:
                search(chosen + [t], uncovered - covers[t])

    search([], set(range(n_q)))
    if not found:
        # --max-tables too small or the search limit hit first: never return an incomplete or empty answer
        chosen, uncovered = [], set(range(n_q))
        while uncovered:
            t = max(covers, key=lambda t: len(covers[t] & uncovered))
            chosen.append(t)
            uncovered -= covers[t]
        found.add(frozenset(chosen))
        return found, True
    return found, visits > limit


def ranked_matrices(scores, threshold, per_query=5, max_tables=None, limit=200000):
    """Sub-matrices of every minimal cover, sorted by mean of per-query best score desc (then fewer tables)."""
    n_q = len(scores)
    cand = candidates(scores, threshold, per_query)
    found, truncated = minimal_covers(cand, max_tables or n_q, limit)
    results = []
    for tables in found:
        cols = sorted(tables)
        sub = [[scores[q][t] for t in cols] for q in range(n_q)]
        avg = sum(max(row) for row in sub) / n_q
        results.append({'tables': cols, 'matrix': sub, 'average_score': avg})
    results.sort(key=lambda r: (-r['average_score'], len(r['tables']), r['tables']))
    for r in results:
        assert all(set(ts) & set(r['tables']) for ts in cand), 'combination leaves a query unfulfilled'
    return results, truncated


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', default='output/score_matrix.json')
    p.add_argument('--threshold', type=float, default=50,
                   help='minimum score (0-100) for a table to fulfil a query')
    p.add_argument('--top', type=int, default=10, help='how many ranked matrices to keep')
    p.add_argument('--per-query', type=int, default=5,
                   help='consider only the N best tables above threshold for each query')
    p.add_argument('--max-tables', type=int, default=None, help='largest combination size (default: number of queries)')
    p.add_argument('--output', default='output/ranked_table_matrices.json')
    args = p.parse_args()
    if args.top < 1 or args.per_query < 1:
        p.error('--top and --per-query must be positive')
    data = json.loads(Path(args.matrix).read_text(encoding='utf-8'))
    tables, queries = data['tables'], data['queries']
    scores = [[100 * s for s in row] for row in data['scores']]
    if not scores or not tables:
        raise ValueError('Score matrix is empty')
    ranked, truncated = ranked_matrices(scores, args.threshold, args.per_query, args.max_tables)
    weak = [queries[q]['id'] for q, row in enumerate(scores) if max(row) < args.threshold]
    out = {'matrix_source': args.matrix, 'threshold': args.threshold, 'score_scale': '0-100',
           'total_valid_combinations': len(ranked), 'search_truncated': truncated,
           'weak_queries': weak, 'queries': [{k: queries[q][k] for k in ('id', 'seed_column', 'query')}
                                              for q in range(len(queries))],
           'ranked': [{'rank': i, 'tables': [tables[t] for t in r['tables']],
                       'average_score': round(r['average_score'], 2),
                       'matrix': [[round(s, 2) for s in row] for row in r['matrix']]}
                      for i, r in enumerate(ranked[:args.top], 1)]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"{len(ranked)} valid table combinations; saved top {len(out['ranked'])} to {target}")
    if truncated:
        print('Search was cut short (size limit or --max-tables too small), so the list may be partial; '
              'every combination shown still fulfils all queries.')
    if weak:
        print(f"Weak coverage (no table reaches {args.threshold:g}): {', '.join(weak)}")
    qids = [q['id'] for q in out['queries']]
    for r in out['ranked']:
        print(f"\n#{r['rank']}  avg {r['average_score']:.2f}  tables: {', '.join(r['tables'])}")
        print('      ' + ''.join(f'{t[:12]:>14}' for t in r['tables']))
        for qid, row in zip(qids, r['matrix']):
            print(f'{qid:>5} ' + ''.join(f'{s:>14.2f}' for s in row))


if __name__ == '__main__':
    main()
