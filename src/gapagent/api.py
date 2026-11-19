"""REST API (FastAPI). OpenAPI spec at /openapi.json can be imported as a custom connector
(e.g. Copilot Studio / Power Automate HTTP action) or behind an API gateway."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import run_agent
from .audit import AuditStore
from .config import load_config
from .llm import DataClassificationError
from .loader import load_markets
from .pipeline import market_view, run_gap_check
from .report import market_counts, render_html
from .schema import GAP_VERDICTS

log = logging.getLogger("gapagent")
logging.basicConfig(level=logging.INFO, format="%(message)s")

CFG = load_config()
STORE = AuditStore(CFG["db_path"])
METRICS: dict[str, float] = defaultdict(float)

app = FastAPI(title="Pharma Regulatory Gap Agent", version=__version__,
              description="Multi-market regulatory gap analysis (US / India / Japan) for CTD dossiers. "
                          "Prototype on synthetic data.")


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:8])
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        ms = (time.perf_counter() - t0) * 1000
        METRICS["http_requests_total"] += 1
        METRICS["http_request_ms_sum"] += ms
        if status >= 500:
            METRICS["http_errors_total"] += 1
        log.info(json.dumps({"rid": rid, "method": request.method, "path": request.url.path,
                             "status": status, "ms": round(ms, 1)}))


class GapCheckRequest(BaseModel):
    dossier_id: str = Field(examples=["zelvorin_global"])
    markets: list[Literal["US", "IN", "JP"]] = Field(examples=[["US", "IN", "JP"]])
    mode: Literal["diff", "full"] = "diff"


class AskRequest(BaseModel):
    question: str = Field(examples=["What blocks a launch in India and Japan?"])
    dossier_id: str = "zelvorin_global"


def _dossier_exists(dossier_id: str):
    if not (Path(CFG["data_dir"]) / "dossiers" / dossier_id / "manifest.yaml").exists():
        raise HTTPException(404, f"unknown dossier {dossier_id}")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "backend": CFG["backend"], "model": CFG["model"]}


@app.get("/markets")
def markets():
    return load_markets(Path(CFG["data_dir"]))


@app.get("/dossiers")
def dossiers():
    root = Path(CFG["data_dir"]) / "dossiers"
    return sorted(p.name for p in root.iterdir() if (p / "manifest.yaml").exists())


@app.post("/gap-checks")
def create_gap_check(req: GapCheckRequest):
    _dossier_exists(req.dossier_id)
    try:
        run = run_gap_check(CFG, req.dossier_id, list(req.markets), req.mode)
    except DataClassificationError as e:
        raise HTTPException(403, str(e)) from e
    STORE.save(run)
    METRICS["gap_checks_total"] += 1
    METRICS["llm_calls_total"] += run.stats.llm_calls
    return {
        "run_id": run.run_id,
        "report_url": f"/gap-checks/{run.run_id}/report",
        "stats": run.stats.model_dump(),
        "markets": {m: {"pathway": run.pathways[m].name, **market_counts(run, m),
                        "gaps": [{"req_id": f.req_id, "verdict": f.verdict.value, "severity": f.severity}
                                 for f in market_view(run, m) if f.verdict in GAP_VERDICTS]}
                    for m in run.markets},
    }


@app.get("/gap-checks")
def list_gap_checks(limit: int = 20):
    return STORE.list_runs(limit)


@app.get("/gap-checks/{run_id}")
def get_gap_check(run_id: str):
    run = STORE.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run.model_dump(mode="json")


@app.get("/gap-checks/{run_id}/summary")
def gap_summary(run_id: str):
    return STORE.gap_summary(run_id)


@app.get("/gap-checks/{run_id}/report", response_class=HTMLResponse)
def gap_report(run_id: str):
    run = STORE.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return render_html(run)


@app.post("/agent/ask")
def ask(req: AskRequest):
    _dossier_exists(req.dossier_id)
    try:
        res = run_agent(req.question, CFG, req.dossier_id, STORE)
    except DataClassificationError as e:
        raise HTTPException(403, str(e)) from e
    METRICS["agent_questions_total"] += 1
    METRICS["agent_guardrail_flags_total"] += len(res.flags)
    return {"answer": res.answer, "run_id": res.run_id, "flags": res.flags, "trace": res.trace,
            "llm_calls": res.llm_calls}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return "".join(f"gapagent_{k} {v:g}\n" for k, v in sorted(METRICS.items()))


@app.get("/", response_class=HTMLResponse)
def home():
    runs = STORE.list_runs(10)
    rows = "".join(f'<li><a href="/gap-checks/{r["run_id"]}/report">{r["run_id"]}</a> · {r["dossier_id"]} → '
                   f'{r["markets"]} · {r["backend"]} · {r["created_at"]}</li>' for r in runs) or "<li>No runs yet.</li>"
    return f"""<!doctype html><meta charset="utf-8"><title>Gap Agent</title>
<body style="font:15px system-ui;max-width:760px;margin:40px auto">
<h1>Pharma Regulatory Gap Agent</h1><p>Backend: <b>{CFG['backend']}</b>. API docs: <a href="/docs">/docs</a>.</p>
<h3>Run a gap check</h3>
<form onsubmit="event.preventDefault();fetch('/gap-checks',{{method:'POST',headers:{{'content-type':'application/json'}},
 body:JSON.stringify({{dossier_id:d.value,markets:[...document.querySelectorAll('input[name=m]:checked')].map(x=>x.value)}})}})
 .then(r=>r.json()).then(j=>location.href=j.report_url)">
<select id="d"><option>zelvorin_global</option><option>zelvorin_ref</option></select>
<label><input type="checkbox" name="m" value="US" checked>US</label>
<label><input type="checkbox" name="m" value="IN" checked>IN</label>
<label><input type="checkbox" name="m" value="JP" checked>JP</label>
<button>Run</button></form>
<h3>Recent runs</h3><ul>{rows}</ul></body>"""
