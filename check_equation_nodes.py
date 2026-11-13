"""Walk the max-heap of table sets and test which node's joined tables satisfy the seed equation.

    python check_equation_nodes.py
    python check_equation_nodes.py --threshold 60 --stop-at-first

Inputs (all produced by earlier steps):
  * seed equation          output/column_relationship_equations.json   (Table(...) AND belongs_to/subset_of)
  * seed queries           output/column_queries_groq.json
  * score matrix           output/groq_query_score_matrix.json         (query x table, nearest source column)
  * heap of table sets     output/ranked_groq_table_matrices.json      (rank 1 = top of the max-heap)
  * per-table equations    built from data/queries/queries/groq/<source_file>

For every heap node (a set of tables) the tables are joined on their entity key, giving one combined
table whose equation is the union of the member equations. The seed equation is satisfied by the node when
every seed column can be bound to a source column (score >= threshold) such that

  * the entity column binds to a join-key column of a table in the node, and every table in the node can
    be joined to it (its own key matches the seed entity query),
  * belongs_to(c, entity): the bound source column belongs_to its table's key in that table's equation,
  * subset_of(a, b):       the bound columns are in the same table and that table's equation has
                           subset_of(sa, sb) (directly or through a chain).

Equation-level only: it needs no row data. The result is the same for inner/left/right/outer joins; the join
type only changes which bound columns can be NULL (reported per node).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from equationextractorforembedding import build_equations

QUOTED = r'("(?:[^"\\]|\\.)*")'
PREDICATE = re.compile(rf'(belongs_to|subset_of)\({QUOTED}, {QUOTED}\)')
TABLE_TERM = re.compile(r'^Table\(columns=\[(.*?)\]\)', re.S)


def parse_equation(equation):
    """Split an equation string into its column list and (relation, column, target) predicates."""
    head = TABLE_TERM.match(equation)
    if not head:
        raise ValueError('Equation must start with Table(columns=[...])')
    columns = [json.loads(s) for s in re.findall(QUOTED, head.group(1))]
    preds = [(r, json.loads(a), json.loads(b)) for r, a, b in PREDICATE.findall(equation)]
    return columns, preds


def closure(preds, relation):
    """Transitive closure of one relation as {column: {reachable targets}}."""
    direct = {}
    for r, a, b in preds:
        if r == relation:
            direct.setdefault(a, set()).add(b)
    out = {}
    for start in direct:
        seen, stack = set(), list(direct[start])
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                stack.extend(direct.get(c, ()))
        out[start] = seen
    return out


class SourceTable:
    """One source table: its equation, entity key and ownership/subset relations."""

    def __init__(self, tid, source_file, records):
        self.id, self.source_file = tid, source_file
        eq = build_equations(records)
        self.equation = eq['equation']
        self.columns, self.preds = parse_equation(self.equation)
        self.owner = {a: b for r, a, b in self.preds if r == 'belongs_to'}
        self.subset = closure(self.preds, 'subset_of')
        self.root = next((c for c in self.columns if c not in self.owner), None)
        self.derived = {r['column'] for r in records if r.get('is_derived')}

    def has_subset(self, a, b):
        return b in self.subset.get(a, ())


def bind_component(component, table, ctx):
    """Bind every seed column of one subset_of-connected component to `table`; return (ok, reasons, binding)."""
    binding, reasons = {}, []
    for q in component:
        if q in ctx['derived']:  # computed from its parents after binding, not bound to a source column
            continue
        cell = ctx['cells'][q][table.id]
        if cell['similarity'] < ctx['threshold']:
            reasons.append(f"{ctx['seed_cols'][q]}: {table.id} scores {cell['similarity']:.1f} < {ctx['threshold']:g}")
            continue
        sc = cell['source_column']
        # a join-key column (entity or its normalised alias such as Countries__norm) identifies itself
        if sc != table.root and table.owner.get(sc) != table.root:
            reasons.append(f"{ctx['seed_cols'][q]}: {table.id}.{sc} does not belong_to {table.root}")
            continue
        if sc in table.derived:
            reasons.append(f"{ctx['seed_cols'][q]}: is not derived but {table.id}.{sc} is")
            continue
        binding[q] = {'table': table.id, 'source_column': sc, 'similarity': round(cell['similarity'], 2)}
    for r, a, b in ctx['seed_preds']:
        if r != 'subset_of' or ctx['col_q'][a] not in component:
            continue
        qa, qb = ctx['col_q'][a], ctx['col_q'][b]
        if qa in binding and qb in binding and not table.has_subset(binding[qa]['source_column'],
                                                                     binding[qb]['source_column']):
            reasons.append(f"subset_of({a}, {b}): {table.id} has no subset_of("
                           f"{binding[qa]['source_column']}, {binding[qb]['source_column']})")
    return not reasons, reasons, binding


def components(n_q, edges):
    parent = list(range(n_q))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for q in range(n_q):
        groups.setdefault(find(q), []).append(q)
    return list(groups.values())


def check_node(node_tables, tables, ctx):
    """Return the verdict for one heap node (a list of table ids)."""
    members = [tables[t] for t in node_tables]
    failures = []

    # 1. join graph: every table's key must match the seed entity query, so they all join on the entity.
    keys = {}
    for t in members:
        cell = ctx['cells'][ctx['entity_q']][t.id]
        ok = cell['similarity'] >= ctx['threshold'] and cell['source_column'] == t.root
        keys[t.id] = {'key': t.root, 'entity_similarity': round(cell['similarity'], 2), 'joinable': ok}
        if not ok:
            failures.append(f"{t.id}: key {t.root!r} does not match the seed entity ({cell['source_column']!r}, "
                            f"{cell['similarity']:.1f}); cannot join on the entity")
    join_plan = [{'left': members[0].id, 'right': t.id,
                  'on': f"{members[0].root} = {t.root}"} for t in members[1:]]

    # 2. bind seed columns: each subset_of-connected component must live in one table of the node.
    comps, all_options = components(len(ctx['seed_cols']), ctx['subset_edges']), []
    for comp in comps:
        options = []
        for t in members:
            ok, reasons, b = bind_component(comp, t, ctx)
            options.append((ok, sum(x['similarity'] for x in b.values()), t.id, reasons, b))
        good = sorted((o for o in options if o[0]), key=lambda o: -o[1])
        all_options.append(good)
        if not good:
            names = ', '.join(ctx['seed_cols'][q] for q in comp)
            failures.append(f"[{names}] cannot be bound in any single table: "
                            + ' | '.join(f"{o[2]}: {'; '.join(o[3])}" for o in options))

    # distinct seed columns need distinct source columns; pick the best collision-free choice per component
    binding = {}
    if not failures:
        best = {'score': -1, 'choice': None}

        def search(i, used, score, choice):
            if i == len(all_options):
                if score > best['score']:
                    best.update(score=score, choice=list(choice))
                return
            for o in all_options[i]:
                cols = {(v['table'], v['source_column']) for v in o[4].values()}
                if len(cols) == len(o[4]) and not cols & used:
                    search(i + 1, used | cols, score + o[1], choice + [o[4]])

        search(0, set(), 0.0, [])
        if best['choice'] is None:
            clash = {}
            for opts in all_options:
                for q, v in opts[0][4].items():
                    clash.setdefault((v['table'], v['source_column']), []).append(ctx['seed_cols'][q])
            failures.append('no one-to-one binding (a source column cannot serve two seed columns): ' + '; '.join(
                f"{t}.{c} <- {' & '.join(names)}" for (t, c), names in clash.items() if len(names) > 1))
        else:
            for b in best['choice']:
                binding.update(b)

    # derived seed columns are computed from their parents once those are bound (or computed)
    computed, pending = {}, set(ctx['derived'])
    while pending and not failures:
        ready = [q for q in pending if all(p in binding or p in computed for p in ctx['derived'][q]['parents'])]
        if not ready:
            break
        for q in ready:
            computed[q] = ctx['derived'][q]
            pending.discard(q)
    for q in sorted(pending) if not failures else []:
        missing = [ctx['seed_cols'][p] for p in ctx['derived'][q]['parents'] if p not in binding and p not in computed]
        failures.append(f"{ctx['seed_cols'][q]}: cannot be computed, missing {', '.join(missing)}")
    satisfied = not failures and len(binding) + len(computed) == len(ctx['seed_cols'])
    seed_cols = ctx['seed_cols']
    bound = {seed_cols[q]: v for q, v in sorted(binding.items())}
    computed_out = {seed_cols[q]: {'formula': d['formula'], 'from': [seed_cols[p] for p in d['parents']]}
                    for q, d in sorted(computed.items())}

    # 3. join types: same equation; only NULL-ability of bound columns changes.
    driver = binding.get(ctx['entity_q'], {}).get('table', members[0].id)
    used = {v['table'] for v in bound.values()}
    nullable = {
        'inner': [],
        'left': sorted(c for c, v in bound.items() if v['table'] != driver),
        'right': sorted(c for c, v in bound.items() if v['table'] == driver) if len(used) > 1 else [],
        'outer': sorted(bound) if len(members) > 1 else [],
    }
    combined = {
        'columns': [f'{t.id}.{c}' for t in members for c in t.columns],
        'predicates': [f'{r}({t.id}.{a}, {t.id}.{b})' for t in members for r, a, b in t.preds],
        'join_keys': ' = '.join(f'{t.id}.{t.root}' for t in members),
    }
    return {
        'tables': [t.id for t in members],
        'satisfied': satisfied,
        'failures': failures,
        'join_plan': join_plan,
        'keys': keys,
        'binding': bound,
        'computed': computed_out,
        'avg_bound_similarity': round(sum(v['similarity'] for v in bound.values()) / len(bound), 2) if bound else None,
        'nullable_columns_by_join_type': nullable,
        'combined_equation': combined,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed-equation', default='output/column_relationship_equations.json')
    p.add_argument('--seed-queries', default='output/column_queries_groq.json')
    p.add_argument('--matrix', default='output/groq_query_score_matrix.json')
    p.add_argument('--heap', default='output/ranked_groq_table_matrices.json')
    p.add_argument('--queries-dir', default='data/queries/queries/groq')
    p.add_argument('--threshold', type=float, default=None, help='default: the heap file threshold')
    p.add_argument('--stop-at-first', action='store_true', help='stop at the first satisfying node')
    p.add_argument('--output', default='output/equation_node_check.json')
    args = p.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    load = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    seed_eq = load(args.seed_equation)
    seed_cols, seed_preds = parse_equation(seed_eq['equation'])
    seed_records = load(args.seed_queries)
    matrix, heap = load(args.matrix), load(args.heap)
    threshold = args.threshold if args.threshold is not None else heap['threshold']

    mq = [q['seed_column'] for q in matrix['queries']]
    if mq != seed_cols:
        raise SystemExit('Score matrix rows do not match the seed equation columns; rerun match_groq_queries.py')
    col_q = {c: i for i, c in enumerate(seed_cols)}

    tables = {}
    for t in matrix['tables']:
        records = load(Path(args.queries_dir) / t['source_file'])
        tables[t['id']] = SourceTable(t['id'], t['source_file'], records)

    cells = [{t['id']: {**matrix['best_matches'][q][ti], 'similarity': matrix['similarities'][q][ti]}
              for ti, t in enumerate(matrix['tables'])} for q in range(len(seed_cols))]
    seed_root_set = {a for r, a, b in seed_preds if r == 'belongs_to'}
    entity_q = col_q[next(c for c in seed_cols if c not in seed_root_set)]
    derived = {col_q[r['column']]: {'parents': [col_q[c] for c in r.get('derived_from') or []],
                                    'formula': r.get('derivation_formula')}
               for r in seed_records if r.get('is_derived')}
    ctx = {'derived': derived, 'cells': cells, 'seed_cols': seed_cols, 'seed_preds': seed_preds, 'col_q': col_q,
           'entity_q': entity_q, 'threshold': threshold,
           'subset_edges': [(col_q[a], col_q[b]) for r, a, b in seed_preds if r == 'subset_of']}

    results = []
    for node in heap['ranked']:
        res = check_node([t['id'] for t in node['tables']], tables, ctx)
        res.update(rank=node['rank'], heap_average_score=node['average_score'])
        results.append(res)
        if args.stop_at_first and res['satisfied']:
            break

    out = {'seed_equation': seed_eq['equation'], 'entity_column': seed_cols[entity_q], 'threshold': threshold,
           'nodes_checked': len(results), 'satisfying_ranks': [r['rank'] for r in results if r['satisfied']],
           'nodes': results}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f"Seed entity: {seed_cols[entity_q]} | {len(seed_cols)} columns, {len(seed_preds)} predicates | "
          f"threshold {threshold:g}")
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
                print(f"    {tid} supplies {len(cols)}/{len(seed_cols)} columns")
            print(f"    {len(r['computed'])} columns computed from bound ones (derived)")
            print(f"    NULL-prone under: left {len(r['nullable_columns_by_join_type']['left'])}, "
                  f"right {len(r['nullable_columns_by_join_type']['right'])}, "
                  f"outer {len(r['nullable_columns_by_join_type']['outer'])}, inner 0 columns")
        for f in r['failures']:
            print(f"    x {f}")
    print(f"\nSatisfying heap ranks: {out['satisfying_ranks'] or 'none'}  (details: {target})")


if __name__ == '__main__':
    main()
