| method | dense index | section+region filter | n | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| bm25 | - | off | 27 | 0.926 | 1.000 | 1.000 | 0.963 |
| bm25 | - | on | 27 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | in-memory TF-IDF | off | 27 | 0.926 | 1.000 | 1.000 | 0.963 |
| dense | in-memory TF-IDF | on | 27 | 0.963 | 1.000 | 1.000 | 0.981 |
| dense | vector DB (local) | off | 27 | 0.926 | 1.000 | 1.000 | 0.963 |
| dense | vector DB (local) | on | 27 | 0.963 | 1.000 | 1.000 | 0.981 |
| dense | vector DB (ChromaDB) | off | 27 | 0.926 | 1.000 | 1.000 | 0.963 |
| dense | vector DB (ChromaDB) | on | 27 | 0.963 | 1.000 | 1.000 | 0.981 |
| hybrid | in-memory TF-IDF | off | 27 | 0.889 | 1.000 | 1.000 | 0.944 |
| hybrid | in-memory TF-IDF | on | 27 | 0.963 | 1.000 | 1.000 | 0.981 |
| hybrid | vector DB (local) | off | 27 | 0.889 | 1.000 | 1.000 | 0.944 |
| hybrid | vector DB (local) | on | 27 | 0.963 | 1.000 | 1.000 | 0.981 |
| hybrid | vector DB (ChromaDB) | off | 27 | 0.889 | 1.000 | 1.000 | 0.944 |
| hybrid | vector DB (ChromaDB) | on | 27 | 0.963 | 1.000 | 1.000 | 0.981 |
