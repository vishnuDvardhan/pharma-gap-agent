"""Retrieval evaluation: does the evidence file for a requirement appear in the top-k chunks?

Gold: `evidence:` block in data/gold/gaps_gold.yaml (file that holds the evidence).
Grid: method {bm25, dense, hybrid} x dense index {in-memory TF-IDF, local vector DB, ChromaDB}
      x section/region filter {on, off}. Metrics: recall@1/3/5, MRR.
The ChromaDB rows run only when chromadb is installed (pip install -e ".[vectordb]").
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile

import yaml

from gapagent.config import ROOT, load_config
from gapagent.loader import load_dossier, load_requirements
from gapagent.retrieval import Retriever, candidate_indices, requirement_query
from gapagent.vectorstore import DOSSIER_COLLECTION, index_dossier, make_vector_store


def main() -> int:
    gold = yaml.safe_load((ROOT / "data" / "gold" / "gaps_gold.yaml").read_text())["evidence"]
    reqs = {r.id: r for rs in load_requirements(ROOT / "data").values() for r in rs}
    stores = ["memory", "local"] + (["chroma"] if importlib.util.find_spec("chromadb") else [])
    tmp = tempfile.mkdtemp(prefix="gapagent-eval-")
    rows = []
    for dossier_id, ev in gold.items():
        d = load_dossier(ROOT / "data" / "dossiers" / dossier_id)
        for method in ("bm25", "dense", "hybrid"):
            for store_kind in (["-"] if method == "bm25" else stores):
                store = None
                if store_kind in ("local", "chroma"):
                    cfg = load_config(vector_store=store_kind, vector_path=f"{tmp}/{store_kind}", embedder="hash")
                    store = make_vector_store(cfg, DOSSIER_COLLECTION)
                    index_dossier(store, d)
                ret = Retriever(d.chunks, method, "tfidf", vector_store=store)
                for filt in (False, True):
                    hits_at = {1: 0, 3: 0, 5: 0}
                    rr = 0.0
                    for req_id, file in ev.items():
                        idx, _ = candidate_indices(d.chunks, reqs[req_id], section_filter=filt)
                        hits = ret.search(requirement_query(reqs[req_id]), 10, idx)
                        files = [h.chunk.file for h in hits]
                        rank = files.index(file) + 1 if file in files else None
                        for k in hits_at:
                            hits_at[k] += bool(rank and rank <= k)
                        rr += 1 / rank if rank else 0
                    n = len(ev)
                    rows.append({"dossier": dossier_id, "method": method, "index": store_kind, "filter": filt, "n": n,
                                 **{f"recall@{k}": v / n for k, v in hits_at.items()}, "mrr": rr / n})

    # pool the two dossiers (weighted by n) into one row per setting
    pooled: dict[tuple, dict] = {}
    for r in rows:
        key = (r["method"], r["index"], r["filter"])
        p = pooled.setdefault(key, {"method": r["method"], "index": r["index"], "filter": r["filter"], "n": 0,
                                    "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "mrr": 0.0})
        p["n"] += r["n"]
        for m in ("recall@1", "recall@3", "recall@5", "mrr"):
            p[m] += r[m] * r["n"]
    table = []
    for p in pooled.values():
        for m in ("recall@1", "recall@3", "recall@5", "mrr"):
            p[m] = round(p[m] / p["n"], 3)
        table.append(p)

    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_eval.json").write_text(json.dumps({"per_dossier": rows, "pooled": table}, indent=2))
    label = {"-": "-", "memory": "in-memory TF-IDF", "local": "vector DB (local)", "chroma": "vector DB (ChromaDB)"}
    lines = ["| method | dense index | section+region filter | n | recall@1 | recall@3 | recall@5 | MRR |",
             "|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['method']} | {label[r['index']]} | {'on' if r['filter'] else 'off'} | {r['n']} | "
                     f"{r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['mrr']:.3f} |")
    (out / "retrieval_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
