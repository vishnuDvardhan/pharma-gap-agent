"""Retrieval evaluation: does the evidence file for a requirement appear in the top-k chunks?

Gold: `evidence:` block in data/gold/gaps_gold.yaml (file that holds the evidence).
Grid: method {bm25, dense, hybrid} x section/region filter {on, off}. Metrics: recall@1/3/5, MRR.
"""
from __future__ import annotations

import json
import sys

import yaml

from gapagent.config import ROOT
from gapagent.loader import load_dossier, load_requirements
from gapagent.retrieval import Retriever, candidate_indices, requirement_query


def main() -> int:
    gold = yaml.safe_load((ROOT / "data" / "gold" / "gaps_gold.yaml").read_text())["evidence"]
    reqs = {r.id: r for rs in load_requirements(ROOT / "data").values() for r in rs}
    rows = []
    for dossier_id, ev in gold.items():
        d = load_dossier(ROOT / "data" / "dossiers" / dossier_id)
        for method in ("bm25", "dense", "hybrid"):
            ret = Retriever(d.chunks, method)
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
                rows.append({"dossier": dossier_id, "method": method, "filter": filt, "n": n,
                             **{f"recall@{k}": round(v / n, 3) for k, v in hits_at.items()}, "mrr": round(rr / n, 3)})

    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_eval.json").write_text(json.dumps(rows, indent=2))
    lines = ["| method | section+region filter | n | recall@1 | recall@3 | recall@5 | MRR |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {'on' if r['filter'] else 'off'} | {r['n']} | {r['recall@1']:.3f} | "
                     f"{r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['mrr']:.3f} |")
    (out / "retrieval_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
