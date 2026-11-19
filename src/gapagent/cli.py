"""Command line: `gapagent check`, `gapagent ask`, `gapagent serve`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import run_agent
from .audit import AuditStore
from .config import load_config
from .pipeline import market_view, run_gap_check
from .report import market_counts, render_html
from .schema import GAP_VERDICTS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gapagent")
    p.add_argument("--config")
    p.add_argument("--backend", help="offline-heuristic | offline-keyword | anthropic | openai | bedrock | ollama")
    p.add_argument("--model")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="run a gap check and write an HTML report")
    c.add_argument("dossier")
    c.add_argument("--markets", default="US,IN,JP")
    c.add_argument("--mode", choices=["diff", "full"])
    c.add_argument("--out", default="reports")

    a = sub.add_parser("ask", help="ask the launch-readiness agent")
    a.add_argument("question")
    a.add_argument("--dossier", default="zelvorin_global")
    a.add_argument("--trace", action="store_true")

    s = sub.add_parser("serve", help="start the REST API")
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    cfg = load_config(args.config, backend=args.backend, model=args.model)

    if args.cmd == "check":
        run = run_gap_check(cfg, args.dossier, args.markets.split(","), args.mode)
        AuditStore(cfg["db_path"]).save(run)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"gap_{run.dossier_id}_{run.run_id}.html"
        path.write_text(render_html(run), encoding="utf-8")
        print(f"run {run.run_id} · backend {run.backend} · {run.stats.requirements_checked} checks, "
              f"{run.stats.llm_calls} LLM calls, {run.stats.duration_ms} ms")
        for m in run.markets:
            cnt = market_counts(run, m)
            print(f"\n{m}: {run.pathways[m].name}\n  {cnt['gaps']} gaps ({cnt['critical']} critical), "
                  f"{cnt['expert']} expert review, {cnt['met']} met")
            for f in market_view(run, m):
                if f.verdict in GAP_VERDICTS:
                    print(f"  - {f.verdict.value:8} [{f.severity}] {f.req_id}: {f.reasoning[:90]}")
        print(f"\nreport: {path}")
    elif args.cmd == "ask":
        res = run_agent(args.question, cfg, args.dossier, AuditStore(cfg["db_path"]))
        if args.trace:
            for t in res.trace:
                print(f"> {t['tool']}({t['args']})")
            print()
        print(res.answer)
        if res.flags:
            print(f"\nflags: {res.flags}")
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("gapagent.api:app", host="0.0.0.0", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
