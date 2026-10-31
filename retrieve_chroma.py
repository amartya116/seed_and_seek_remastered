"""Score every query from query.py/querygroq.py JSON against every indexed table card.

    python retrieve_chroma.py --queries output/column_queries.json
    python retrieve_chroma.py --queries output/column_queries_groq.json

Outputs a queries x tables matrix (rows q1..qN, columns T1..TM) whose entries are cosine
similarity scores (1 - cosine distance, higher is better), as JSON and CSV.
No matching or augmentation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_queries(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('Expected JSON array of column records from query.py or querygroq.py')
    queries = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f'Column item {i} is not an object')
        column = item.get('column')
        texts = item.get('queries', item.get('query'))
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(column, str) or not isinstance(texts, list):
            raise ValueError(f'Column item {i} missing column or query/queries')
        for q in texts:
            if isinstance(q, str) and q.strip():
                queries.append({'seed_column': column, 'query': q.strip()})
    if not queries:
        raise ValueError('No usable natural-language queries found')
    return queries


def build_matrix(tables, vectors):
    """Score every query vector against every table card: returns (table labels, rows of 1 - cosine distance)."""
    ids = tables.get(include=[])['ids']
    labels = {}
    scores = []
    for vec in vectors:
        result = tables.query(query_embeddings=[vec], n_results=len(ids), include=['metadatas', 'distances'])
        by_id = {rid: 1.0 - float(d) for rid, d in zip(result['ids'][0], result['distances'][0])}
        for rid, m in zip(result['ids'][0], result['metadatas'][0]):
            labels.setdefault(rid, (m or {}).get('table_id') or rid)
        scores.append(by_id)
    order = sorted(labels, key=lambda rid: labels[rid])
    return [labels[rid] for rid in order], [[row[rid] for rid in order] for row in scores]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', default='output/column_queries.json')
    p.add_argument('--chroma', default='data/chroma_seed_seek')
    p.add_argument('--output', default='output/score_matrix.json')
    p.add_argument('--csv', default='output/score_matrix.csv')
    args = p.parse_args()
    input_path = Path(args.queries)
    q_items = load_queries(input_path)
    if not Path(args.chroma).is_dir():
        raise FileNotFoundError('Chroma index absent; run index_chroma.py first')
    import chromadb
    from sentence_transformers import SentenceTransformer
    client = chromadb.PersistentClient(path=str(Path(args.chroma)))
    tables = client.get_collection(name='table_cards')
    if tables.count() == 0:
        raise RuntimeError('No indexed table cards found')
    meta = tables.metadata or {}
    model_name = meta.get('embedding_model')
    if not model_name:
        raise RuntimeError('table_cards embedding model metadata missing')
    if meta.get('hnsw:space') != 'cosine':
        raise RuntimeError('table_cards is not configured for cosine distance')
    model = SentenceTransformer(model_name)
    vectors = model.encode([x['query'] for x in q_items], normalize_embeddings=True).tolist()
    table_labels, scores = build_matrix(tables, vectors)

    output = {'embedding_model': model_name, 'metric': 'cosine similarity (1 - cosine distance), higher is better',
              'query_source': str(input_path),
              'queries': [{'id': f'q{i}', **x} for i, x in enumerate(q_items, 1)],
              'tables': table_labels, 'scores': scores}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    import pandas as pd
    df = pd.DataFrame(scores, columns=table_labels, index=[q['id'] for q in output['queries']])
    df.insert(0, 'query', [q['query'] for q in output['queries']])
    csv_target = Path(args.csv)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_target, index_label='id', encoding='utf-8')
    print(f'Saved {len(q_items)} x {len(table_labels)} score matrix to {target} and {csv_target}')
    print(df.drop(columns='query').round(3).to_string())


if __name__ == '__main__':
    main()
