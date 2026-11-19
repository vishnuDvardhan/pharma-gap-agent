"""Gap-detection evaluation against hand-written gold labels.

Metrics (per backend, over both dossiers):
  gap_recall     of gold gaps (Missing/Partial), share flagged as Missing, Partial or Needs expert review.
                 This is the safety metric: a missed gap is the expensive failure.
  gap_precision  of predicted Missing/Partial, share that are gold gaps.
  exact_acc      exact verdict match.
  quote_verified share of Met/Partial findings whose every quote was found verbatim in the dossier.
Also reports judge calls in diff vs full mode (cost of re-checking the ICH core per market).

Usage:
  python eval/eval_gaps.py                                   # offline baselines
  python eval/eval_gaps.py --backends anthropic --model claude-sonnet-5-5
  python eval/eval_gaps.py --min-gap-recall 0.9              # CI gate (exit 1 if below)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from gapagent.config import ROOT, load_config
from gapagent.pipeline import run_gap_check
from gapagent.schema import Verdict

GAP = {"Missing", "Partial"}
FLAGGED = GAP | {"Needs expert review"}
MARKETS = ["US", "IN", "JP"]


def evaluate(backend: str, model: str | None, gold: dict, mode: str = "diff", **retrieval) -> dict:
    cfg = load_config(backend=backend, model=model, **retrieval)
    pairs, misses, stats, quotes = [], [], Counter(), Counter()
    for dossier, by_market in gold.items():
        run = run_gap_check(cfg, dossier, MARKETS, mode)
        stats["judge_calls"] += run.stats.judge_calls
        stats["llm_calls"] += run.stats.llm_calls
        stats["input_tokens"] += run.stats.input_tokens
        stats["output_tokens"] += run.stats.output_tokens
        stats["duration_ms"] += run.stats.duration_ms
        for f in run.findings:
            gm = "ICH" if f.market in ("ICH",) or f.req_id.startswith("ICH-") else f.market
            g = by_market[gm][f.req_id]
            p = f.verdict.value
            pairs.append((g, p))
            if g != p:
                misses.append({"dossier": dossier, "market": f.market, "req_id": f.req_id, "gold": g, "pred": p,
                               "reasoning": f.reasoning[:160]})
            if f.verdict in (Verdict.MET, Verdict.PARTIAL) and f.evidence:
                quotes["total"] += 1
                quotes["verified"] += all(e.quote_verified for e in f.evidence)
    gold_gaps = [(g, p) for g, p in pairs if g in GAP]
    pred_gaps = [(g, p) for g, p in pairs if p in GAP]
    labels = ["Met", "Partial", "Missing", "Needs expert review", "Not applicable"]
    return {
        "backend": backend, "model": model or "-", "mode": mode, "retrieval": retrieval or "config",
        "n": len(pairs),
        "gap_recall": round(sum(p in FLAGGED for _, p in gold_gaps) / max(len(gold_gaps), 1), 3),
        "gap_precision": round(sum(g in GAP for g, _ in pred_gaps) / max(len(pred_gaps), 1), 3),
        "exact_acc": round(sum(g == p for g, p in pairs) / len(pairs), 3),
        "quote_verified": round(quotes["verified"] / max(quotes["total"], 1), 3),
        "confusion": {g: {p: sum(1 for x, y in pairs if x == g and y == p) for p in labels} for g in labels},
        "stats": dict(stats), "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="offline-keyword,offline-heuristic")
    ap.add_argument("--model")
    ap.add_argument("--min-gap-recall", type=float)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    gold = yaml.safe_load((ROOT / "data" / "gold" / "gaps_gold.yaml").read_text())["verdicts"]
    results = []
    for b in args.backends.split(","):
        results.append(evaluate(b, args.model, gold, "diff"))
        results.append(evaluate(b, args.model, gold, "full"))
    # retrieval ablation on the best offline judge: does the section/region filter matter?
    if "offline-heuristic" in args.backends:
        results.append(evaluate("offline-heuristic", None, gold, "diff", section_filter=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap_eval.json").write_text(json.dumps(results, indent=2))

    lines = ["| backend | mode | retrieval | n | gap recall | gap precision | exact acc | quotes verified | judge calls | LLM calls |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        rv = "no filter" if isinstance(r["retrieval"], dict) else "section+region filter"
        lines.append(f"| {r['backend']} | {r['mode']} | {rv} | {r['n']} | {r['gap_recall']:.3f} | {r['gap_precision']:.3f} | "
                     f"{r['exact_acc']:.3f} | {r['quote_verified']:.3f} | {r['stats']['judge_calls']} | {r['stats']['llm_calls']} |")
    table = "\n".join(lines)
    (out / "gap_eval.md").write_text(table + "\n")
    print(table)
    for r in results:
        if r["mode"] == "diff" and not isinstance(r["retrieval"], dict):
            print(f"\nMisses for {r['backend']} ({len(r['misses'])}):")
            for m in r["misses"]:
                print(f"  {m['dossier']:16} {m['req_id']:16} gold={m['gold']:<20} pred={m['pred']:<20} {m['reasoning'][:70]}")

    if args.min_gap_recall is not None:
        worst = min(r["gap_recall"] for r in results if r["backend"] == args.backends.split(",")[-1]
                    and not isinstance(r["retrieval"], dict))
        if worst < args.min_gap_recall:
            print(f"\nFAIL: gap recall {worst} < {args.min_gap_recall}")
            return 1
        print(f"\nPASS: gap recall {worst} ≥ {args.min_gap_recall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
