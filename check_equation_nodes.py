"""Walk the max-heap of table sets and explain, per node, whether its joined tables satisfy the seed equation.

    python check_equation_nodes.py
    python check_equation_nodes.py --threshold 60 --stop-at-first

Inputs (produced by earlier steps):
  * seed queries + relations  output/column_queries_groq.json          (belongs_to / subset_of / is_derived)
  * score matrix              output/groq_query_score_matrix.json      (needs column_scores, from match_groq_queries.py)
  * heap of table sets        output/ranked_groq_table_matrices.json   (rank 1 = top of the max-heap)
  * per-table relations       data/queries/queries/groq/<source_file>

The rules are the same ones select_tables.py uses in equation mode (see table_equations.py): every
non-derived seed column is bound to its own source column (score >= threshold), each seed subset_of group
is bound inside one table with the same subset_of shape, bound columns belong_to their table's key, and
every table's key matches the seed entity. Derived seed columns must be computable from bound columns.

Equation-level only (no row data). The verdict is the same for inner/left/right/outer joins; the join type
only changes which bound columns can be NULL, which is reported per node.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from table_equations import (SeedModel, SourceTable, best_binding, column_scores, computable_derived,
                             group_options, joinable, load_json)


def check_node(node_ids, tables, seed, scores, threshold, top_k):
    members = [tables[t] for t in node_ids]
    failures = []

    keys = {}
    for t in members:
        s = scores[seed.entity][t.id].get(t.root, 0.0) if t.root else 0.0
        ok = joinable(t, seed, scores, threshold)
        keys[t.id] = {'key': t.root, 'entity_similarity': round(s, 2), 'joinable': ok}
        if not ok:
            failures.append(f"{t.id}: key {t.root!r} scores {s:.1f} against seed entity {seed.entity!r}; "
                            f"cannot join on the entity")
    joinable_members = [t for t in members if keys[t.id]['joinable']]
    join_plan = [{'left': members[0].id, 'right': t.id, 'on': f"{members[0].root} = {t.root}"}
                 for t in members[1:]]

    options = [{t.id: group_options(g, t, seed, scores, threshold, top_k) for t in joinable_members}
               for g in seed.groups]
    options = [{tid: o for tid, o in by_table.items() if o} for by_table in options]
    for g, by_table in zip(seed.groups, options):
        if not by_table:
            failures.append(f"[{', '.join(g)}] cannot be bound inside any one table of this node")

    binding = {}
    if not failures:
        result = best_binding([t.id for t in joinable_members], options)
        if result is None:
            claims = {}
            for g, by_table in zip(seed.groups, options):
                best = max(((s, tid, b) for tid, opts in by_table.items() for s, b in opts[:1]),
                           key=lambda o: o[0])
                for c, sc in best[2].items():
                    claims.setdefault((best[1], sc), []).append(c)
            failures.append('no one-to-one binding (a source column cannot serve two seed columns): ' + '; '.join(
                f"{tid}.{sc} <- {' & '.join(cs)}" for (tid, sc), cs in claims.items() if len(cs) > 1))
        else:
            for _, tid, b in result[1]:
                for c, sc in b.items():
                    binding[c] = {'table': tid, 'source_column': sc, 'similarity': round(scores[c][tid][sc], 2)}

    computed, missing = computable_derived(seed, binding) if binding else ({}, dict(seed.derived))
    if binding:
        for c, d in missing.items():
            failures.append(f"derived {c}: cannot be computed from bound columns ({', '.join(d['from'])})")

    unused = sorted({t.id for t in members} - {v['table'] for v in binding.values()}) if binding else []
    driver = binding.get(seed.entity, {}).get('table', members[0].id)
    used = {v['table'] for v in binding.values()}
    nullable = {
        'inner': [],
        'left': sorted(c for c, v in binding.items() if v['table'] != driver),
        'right': sorted(c for c, v in binding.items() if v['table'] == driver) if len(used) > 1 else [],
        'outer': sorted(binding) if len(members) > 1 else [],
    }
    return {
        'tables': [t.id for t in members],
        'satisfied': not failures,
        'failures': failures,
        'join_plan': join_plan,
        'keys': keys,
        'binding': binding,
        'computed': computed,
        'unused_tables': unused,
        'avg_bound_similarity': round(sum(v['similarity'] for v in binding.values()) / len(binding), 2)
        if binding else None,
        'nullable_columns_by_join_type': nullable,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed-queries', default='output/column_queries_groq.json')
    p.add_argument('--matrix', default='output/groq_query_score_matrix.json')
    p.add_argument('--heap', default='output/ranked_groq_table_matrices.json')
    p.add_argument('--queries-dir', default='data/queries/queries/groq')
    p.add_argument('--threshold', type=float, default=None, help='default: the heap file threshold')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--stop-at-first', action='store_true', help='stop at the first satisfying node')
    p.add_argument('--output', default='output/equation_node_check.json')
    args = p.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    seed = SeedModel(load_json(args.seed_queries))
    matrix, heap = load_json(args.matrix), load_json(args.heap)
    threshold = args.threshold if args.threshold is not None else heap['threshold']
    scores = column_scores(matrix)
    missing_rows = sorted(set(seed.columns) - set(scores))
    if missing_rows:
        raise SystemExit(f'Score matrix lacks non-derived seed columns {missing_rows}; rerun match_groq_queries.py')
    tables = {t['id']: SourceTable(t['id'], t['source_file'], load_json(Path(args.queries_dir) / t['source_file']))
              for t in matrix['tables']}

    results = []
    for node in heap['ranked']:
        res = check_node([t['id'] for t in node['tables']], tables, seed, scores, threshold, args.top_k)
        res.update(rank=node['rank'], heap_average_score=node['average_score'])
        results.append(res)
        if args.stop_at_first and res['satisfied']:
            break

    out = {'entity_column': seed.entity, 'threshold': threshold, 'seed_groups': seed.groups,
           'derived_seed_columns': seed.derived, 'nodes_checked': len(results),
           'satisfying_ranks': [r['rank'] for r in results if r['satisfied']], 'nodes': results}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f"Seed entity: {seed.entity} | {len(seed.columns)} non-derived columns in {len(seed.groups)} groups, "
          f"{len(seed.derived)} derived | threshold {threshold:g}")
    for r in results:
        flag = 'SATISFIED' if r['satisfied'] else 'not satisfied'
        print(f"\n[#{r['rank']}] {' + '.join(r['tables'])} | heap avg {r['heap_average_score']} -> {flag}")
        for j in r['join_plan']:
            print(f"    join {j['left']} x {j['right']} on {j['on']}")
        if r['satisfied']:
            per_table = {}
            for c, v in r['binding'].items():
                per_table.setdefault(v['table'], []).append(c)
            for tid, cols in per_table.items():
                print(f"    {tid} supplies {len(cols)}/{len(seed.columns)} non-derived columns")
            print(f"    {len(r['computed'])}/{len(seed.derived)} derived columns computable")
            n = r['nullable_columns_by_join_type']
            print(f"    NULL-prone under: left {len(n['left'])}, right {len(n['right'])}, "
                  f"outer {len(n['outer'])}, inner 0 columns")
        for f in r['failures']:
            print(f"    x {f}")
    print(f"\nSatisfying heap ranks: {out['satisfying_ranks'] or 'none'}  (details: {target})")


if __name__ == '__main__':
    main()
