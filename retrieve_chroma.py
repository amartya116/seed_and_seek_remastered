"""Search indexed Wikipedia descriptions using unchanged query.py/querygroq.py JSON.

    python retrieve_chroma.py --queries output/column_queries.json
    python retrieve_chroma.py --queries output/column_queries_groq.json

Outputs transparent search hits only. No matching or augmentation is performed.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', default='output/column_queries.json')
    p.add_argument('--chroma', default='data/chroma_seed_seek')
    p.add_argument('--output', default='output/retrieval_results.json')
    p.add_argument('--top-k', type=int, default=10)
    args = p.parse_args()
    if args.top_k < 1:
        p.error('--top-k must be positive')
    input_path = Path(args.queries)
    q_items = load_queries(input_path)
    if not Path(args.chroma).is_dir():
        raise FileNotFoundError('Chroma index absent; run index_chroma.py first')
    import chromadb
    from sentence_transformers import SentenceTransformer
    client = chromadb.PersistentClient(path=str(Path(args.chroma)))
    collections = {}
    for name in ('table_cards', 'column_cards'):
        c = client.get_collection(name=name)
        if c.count() > 0:
            collections[name] = c
    if not collections:
        raise RuntimeError('No indexed cards found')
    model_names = {(c.metadata or {}).get('embedding_model') for c in collections.values()}
    if len(model_names) != 1 or None in model_names:
        raise RuntimeError('Collection embedding model metadata missing or inconsistent')
    model_name = model_names.pop()
    for name, c in collections.items():
        if (c.metadata or {}).get('hnsw:space') != 'cosine':
            raise RuntimeError(f'{name} is not configured for cosine distance')
    model = SentenceTransformer(model_name)
    vectors = model.encode([x['query'] for x in q_items], normalize_embeddings=True).tolist()
    output = {'embedding_model': model_name, 'metric': 'cosine', 'query_source': str(input_path),
              'note': 'Cosine distance: lower is better; hits are candidates, not verified join matches.', 'searches': []}
    for item, vec in zip(q_items, vectors):
        hits = []
        for name, c in collections.items():
            result = c.query(query_embeddings=[vec], n_results=min(args.top_k, c.count()),
                             include=['metadatas','documents','distances'])
            for rid, doc, meta, distance in zip(result['ids'][0], result['documents'][0],
                                                 result['metadatas'][0], result['distances'][0]):
                hits.append({'collection': name, 'card_id': rid, 'table_id': meta.get('table_id'),
                             'column_name': meta.get('column_name'), 'description': doc,
                             'metadata': meta, 'cosine_distance': float(distance)})
        hits.sort(key=lambda x: x['cosine_distance'])
        output['searches'].append({**item, 'hits': hits})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Saved {len(output['searches'])} searches to {target}")
    for s in output['searches']:
        print(f"\n{s['seed_column']}: {s['query']}")
        for h in s['hits'][:5]:
            print(f"  {h['cosine_distance']:.3f}  {h['collection']}  {h['table_id']}  {h.get('column_name') or ''}")


if __name__ == '__main__':
    main()
