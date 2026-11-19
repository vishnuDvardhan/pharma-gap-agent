"""End-to-end: pipeline regression vs gold, audit immutability, REST API."""
from __future__ import annotations

import importlib
import sqlite3

import pytest
import yaml

from gapagent.audit import AuditStore
from gapagent.config import ROOT
from gapagent.pipeline import market_view, run_gap_check
from gapagent.schema import Verdict

GOLD = yaml.safe_load((ROOT / "data" / "gold" / "gaps_gold.yaml").read_text())["verdicts"]


@pytest.mark.parametrize("dossier", ["zelvorin_ref", "zelvorin_global"])
def test_regression_no_missed_gaps(cfg, dossier):
    """Safety regression: the default checker must not report a gold gap as Met."""
    run = run_gap_check(cfg, dossier, ["US", "IN", "JP"])
    for f in run.findings:
        gold = GOLD[dossier]["ICH" if f.req_id.startswith("ICH-") else f.market][f.req_id]
        if gold in ("Missing", "Partial"):
            assert f.verdict != Verdict.MET, f"{f.req_id}: gold {gold}, predicted Met"


def test_diff_mode_saves_judge_calls_with_same_verdicts(cfg):
    d = run_gap_check(cfg, "zelvorin_global", ["US", "IN", "JP"], "diff")
    f = run_gap_check(cfg, "zelvorin_global", ["US", "IN", "JP"], "full")
    assert d.stats.judge_calls < f.stats.judge_calls
    for m in ["US", "IN", "JP"]:
        dv = {x.req_id: x.verdict for x in market_view(d, m)}
        fv = {x.req_id: x.verdict for x in market_view(f, m)}
        assert dv == fv


def test_override_replaces_core_requirement_for_india_only(cfg):
    run = run_gap_check(cfg, "zelvorin_ref", ["US", "IN"])
    ids_in = {f.req_id for f in market_view(run, "IN")}
    ids_us = {f.req_id for f in market_view(run, "US")}
    assert "ICH-Q-STAB-04" not in ids_in and "IN-Q-STAB-IVB" in ids_in
    assert "ICH-Q-STAB-04" in ids_us


def test_not_applicable_by_pathway(cfg):
    run = run_gap_check(cfg, "zelvorin_global", ["IN"])
    f = next(x for x in run.findings if x.req_id == "IN-C-LOCALCT")
    assert f.verdict == Verdict.NA and "pathway" in f.reasoning


def test_regional_module1_is_not_mixed(cfg):
    """EU Module 1 must never be evidence for a US Module 1 requirement."""
    run = run_gap_check(cfg, "zelvorin_ref", ["US"])
    for f in run.findings:
        if f.req_id.startswith("US-M1"):
            assert all("1.2_application_form" != e.file for e in f.evidence)


def test_audit_store_is_append_only(cfg):
    store = AuditStore(cfg["db_path"])
    run = run_gap_check(cfg, "zelvorin_ref", ["JP"])
    store.save(run)
    assert store.get(run.run_id).run_id == run.run_id
    con = sqlite3.connect(cfg["db_path"])
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        con.execute("UPDATE findings SET verdict='Met' WHERE run_id=?", (run.run_id,))
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        con.execute("DELETE FROM runs WHERE run_id=?", (run.run_id,))


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("GAPAGENT_DB_PATH", str(tmp_path / "api.sqlite"))
    monkeypatch.setenv("GAPAGENT_VECTOR_PATH", str(tmp_path / "vectordb"))
    import gapagent.api as api
    importlib.reload(api)
    return TestClient(api.app)


def test_api_end_to_end(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "zelvorin_global" in client.get("/dossiers").json()
    r = client.post("/gap-checks", json={"dossier_id": "zelvorin_global", "markets": ["IN", "JP"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {g["req_id"] for g in body["markets"]["IN"]["gaps"]} >= {"IN-Q-STAB-IVB", "IN-LBL-SHELF", "IN-PV-PSUR"}
    html = client.get(body["report_url"]).text
    assert "Gap matrix" in html and "IN-LBL-SHELF" in html and "UNVERIFIED" not in html
    assert client.get(f"/gap-checks/{body['run_id']}/summary").json()
    a = client.post("/agent/ask", json={"question": "What blocks Japan?"}).json()
    assert "JP-M1-LANG" in a["answer"] and a["trace"]
    assert "gapagent_http_requests_total" in client.get("/metrics").text
    assert client.post("/gap-checks", json={"dossier_id": "nope", "markets": ["US"]}).status_code == 404
    assert client.post("/gap-checks", json={"dossier_id": "zelvorin_ref", "markets": ["BR"]}).status_code == 422


def test_api_vector_index_and_search(client):
    idx = client.post("/vectors/index").json()
    assert idx["chunks"] > 0 and idx["stats"]["zelvorin_global"]["added"] > 0
    assert client.post("/vectors/index").json()["stats"]["zelvorin_global"]["added"] == 0   # incremental
    hits = client.get("/vectors/search", params={"q": "Form 356h", "dossier_id": "zelvorin_ref",
                                                 "module": "m1", "region": "us"}).json()
    assert hits == []                                     # the EU-only dossier has no US Module 1
    hits = client.get("/vectors/search", params={"q": "Zone IVb stability", "market": "IN", "k": 2}).json()
    assert hits[0]["id"] == "IN-Q-STAB-IVB"
    assert client.get("/vectors/search", params={"q": "x", "k": 99}).status_code == 422
