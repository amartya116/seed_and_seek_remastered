"""Equation structure of the seed and the source tables, and equation-aware coverage.

Used by select_tables.py to decide which table sets satisfy the seed equation (the heap).

Seed side: only non-derived seed columns are searched for. They are grouped by subset_of
(e.g. {All, M, F}); a group must be found inside ONE source table with the same subset_of shape.
Derived seed columns are computed afterwards from their inputs.

Source side: every column is a candidate, derived or not. A source column can stand for a seed column
only if it belongs_to its table's key, and that key matches the seed entity column.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _closure(direct):
    out = {}
    for start in direct:
        seen, stack = set(), [direct[start]]
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                if c in direct:
                    stack.append(direct[c])
        out[start] = seen
    return out


class SourceTable:
    """One source table, read from its query JSON (column, belongs_to, subset_of, is_derived)."""

    def __init__(self, tid, source_file, records):
        self.id, self.source_file = tid, source_file
        self.columns = [r['column'] for r in records]
        self.owner = {r['column']: r.get('belongs_to') for r in records}
        self.parent = {r['column']: r['subset_of'] for r in records if r.get('subset_of')}
        self.derived = {r['column'] for r in records if r.get('is_derived')}
        roots = [c for c in self.columns if not self.owner.get(c)]
        self.root = roots[0] if roots else None
        self._subset = _closure(self.parent)

    def has_subset(self, a, b):
        """True if subset_of(a, b) holds directly or through a chain."""
        return b in self._subset.get(a, ())


class SeedModel:
    """Non-derived seed columns, their entity column, subset_of groups and derived columns."""

    def __init__(self, records):
        derived = [r for r in records if r.get('is_derived')]
        plain = [r for r in records if not r.get('is_derived')]
        self.columns = [r['column'] for r in plain]
        self.derived = {r['column']: {'from': list(r.get('derived_from') or []),
                                      'formula': r.get('derivation_formula')} for r in derived}
        entities = [r['column'] for r in plain if not r.get('belongs_to')]
        if len(entities) != 1:
            raise ValueError(f'Expected exactly one seed entity column (no belongs_to), found {entities}')
        self.entity = entities[0]
        plain_set = set(self.columns)
        self.parent = {r['column']: r['subset_of'] for r in plain
                       if r.get('subset_of') in plain_set}
        groups = {}
        for c in self.columns:
            root = c
            while root in self.parent:
                root = self.parent[root]
            groups.setdefault(root, []).append(c)
        self.groups = list(groups.values())


def column_scores(matrix):
    """{seed_column: {table_id: {source_column: similarity}}} from a match_groq_queries.py matrix."""
    if 'column_scores' not in matrix:
        raise ValueError('Matrix has no column_scores; rerun match_groq_queries.py')
    return {q['seed_column']: row for q, row in zip(matrix['queries'], matrix['column_scores'])}


def joinable(table, seed, scores, threshold):
    """The table's key column must match the seed entity column."""
    return table.root is not None and scores[seed.entity][table.id].get(table.root, 0.0) >= threshold


def group_options(group, table, seed, scores, threshold, top_k=3):
    """Every valid binding of one seed group inside one table, best first.

    Each option is (total similarity, {seed_column: source_column}). A binding is valid when each member
    scores >= threshold, belongs_to the table's key (the entity binds to the key itself), no source column
    is used twice, and every seed subset_of(a, b) inside the group also holds in the table.
    """
    candidates = []
    for c in group:
        ranked = sorted(scores[c][table.id].items(), key=lambda kv: -kv[1])
        if c == seed.entity:
            cols = [sc for sc, s in ranked if sc == table.root and s >= threshold]
        else:
            cols = [sc for sc, s in ranked
                    if s >= threshold and sc != table.root and table.owner.get(sc) == table.root][:top_k]
        if not cols:
            return []
        candidates.append(cols)
    options = []
    for combo in itertools.product(*candidates):
        if len(set(combo)) < len(combo):
            continue
        binding = dict(zip(group, combo))
        if all(table.has_subset(binding[a], binding[b])
               for a, b in seed.parent.items() if a in binding and b in binding):
            options.append((sum(scores[c][table.id][binding[c]] for c in group), binding))
    options.sort(key=lambda o: -o[0])
    return options


def best_binding(table_ids, options):
    """Best one-to-one binding of every group using only `table_ids`, or None.

    options[g][tid] = group_options(...) for group g in table tid. Returns (total score,
    [(group index, tid, binding), ...]).
    """
    per_group = []
    for g, by_table in enumerate(options):
        opts = [(s, tid, b) for tid in table_ids for s, b in by_table.get(tid, [])]
        if not opts:
            return None
        per_group.append((g, sorted(opts, key=lambda o: -o[0])))
    per_group.sort(key=lambda x: len(x[1]))
    bound_max = [max(o[0] for o in opts) for _, opts in per_group]
    suffix = [0.0] * (len(per_group) + 1)
    for i in range(len(per_group) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + bound_max[i]
    best = [None, -1.0]

    def search(i, used, score, chosen):
        if score + suffix[i] <= best[1]:
            return
        if i == len(per_group):
            best[0], best[1] = list(chosen), score
            return
        g, opts = per_group[i]
        for s, tid, b in opts:
            cols = {(tid, sc) for sc in b.values()}
            if not cols & used:
                chosen.append((g, tid, b))
                search(i + 1, used | cols, score + s, chosen)
                chosen.pop()

    search(0, frozenset(), 0.0, [])
    return None if best[0] is None else (best[1], sorted(best[0]))


def build_options(seed, tables, scores, threshold, top_k=3):
    """Per group, per joinable table: the valid bindings. Also returns the joinable table ids."""
    usable = [t for t in tables if joinable(t, seed, scores, threshold)]
    options = [{t.id: group_options(group, t, seed, scores, threshold, top_k) for t in usable}
               for group in seed.groups]
    options = [{tid: o for tid, o in by_table.items() if o} for by_table in options]
    return options, [t.id for t in usable]


def computable_derived(seed, bound_columns):
    """Derived seed columns whose inputs are all bound (or computable), and those that are not."""
    have, pending = set(bound_columns), dict(seed.derived)
    computed = {}
    while True:
        ready = [c for c, d in pending.items() if all(x in have for x in d['from'])]
        if not ready:
            break
        for c in ready:
            computed[c] = pending.pop(c)
            have.add(c)
    return computed, pending
