"""Command line: `gapagent check | ask | index | search | serve`."""
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

    i = sub.add_parser("index", help="build / refresh the vector DB (incremental)")
    i.add_argument("dossiers", nargs="*", help="dossier ids (default: all)")

    q = sub.add_parser("search", help="semantic search over a dossier or the requirement library")
    q.add_argument("query")
    q.add_argument("--dossier", help="search this dossier's chunks (default: the requirement library)")
    q.add_argument("--module", choices=["m1", "m2", "m3", "m4", "m5"])
    q.add_argument("--region", choices=["eu", "us", "in", "jp"])
    q.add_argument("--market", choices=["ICH", "US", "IN", "JP"], help="requirement library only")
    q.add_argument("-k", type=int, default=5)

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
    elif args.cmd in ("index", "search"):
        return _vector_cmd(args, cfg)
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("gapagent.api:app", host="0.0.0.0", port=args.port)
    return 0


def _vector_cmd(args, cfg: dict) -> int:
    from .loader import load_dossier, load_requirements
    from .vectorstore import (
        DOSSIER_COLLECTION,
        REQUIREMENT_COLLECTION,
        index_dossier,
        index_requirements,
        make_vector_store,
    )
    chunks = make_vector_store(cfg, DOSSIER_COLLECTION)
    if chunks is None:
        print("retrieval.vector_store is 'none'; set it to 'local' or 'chroma' in config.yaml")
        return 1
    data = Path(cfg["data_dir"])
    if args.cmd == "index":
        ids = args.dossiers or sorted(p.name for p in (data / "dossiers").iterdir() if (p / "manifest.yaml").exists())
        for d in ids:
            print(f"{d:20} {index_dossier(chunks, load_dossier(data / 'dossiers' / d))}")
        reqs = make_vector_store(cfg, REQUIREMENT_COLLECTION)
        print(f"{'requirements':20} {index_requirements(reqs, [r for rs in load_requirements(data).values() for r in rs])}")
        print(f"backend {chunks.backend} · embedder {chunks.embedder.id} · {chunks.count()} chunks · "
              f"{cfg['retrieval'].get('vector_path')}")
        return 0
    if args.dossier:
        index_dossier(chunks, load_dossier(data / "dossiers" / args.dossier))
        where = {"dossier_id": args.dossier, **({"module": args.module} if args.module else {}),
                 **({"region": args.region} if args.region else {})}
        for h in chunks.query(args.query, args.k, where):
            m = h.metadata
            print(f"{h.score:.3f}  {m['path']}:{m['line_start']}-{m['line_end']}  [{m['section']}] {m['heading']}")
    else:
        reqs = make_vector_store(cfg, REQUIREMENT_COLLECTION)
        index_requirements(reqs, [r for rs in load_requirements(data).values() for r in rs])
        for h in reqs.query(args.query, args.k, {"market": args.market} if args.market else None):
            print(f"{h.score:.3f}  {h.id:18} {h.text[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
