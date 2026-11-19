"""Append-only audit store (SQLite). Supports traceability: who/what/when/which model/which corpus.

There is deliberately no UPDATE or DELETE path: a run, once written, is never changed.
A content hash per run makes later tampering detectable.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .schema import GapRun

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    dossier_id TEXT NOT NULL,
    markets TEXT NOT NULL,
    mode TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    corpus_hash TEXT NOT NULL,
    llm_calls INTEGER, input_tokens INTEGER, output_tokens INTEGER,
    requirements_checked INTEGER, requirements_inherited INTEGER, duration_ms INTEGER,
    content_sha256 TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    req_id TEXT NOT NULL,
    market TEXT NOT NULL,
    verdict TEXT NOT NULL,
    severity TEXT NOT NULL,
    check_type TEXT NOT NULL,
    inherited INTEGER NOT NULL,
    quote_verified INTEGER NOT NULL,
    flags TEXT NOT NULL,
    PRIMARY KEY (run_id, req_id, market)
);
CREATE TRIGGER IF NOT EXISTS runs_no_update BEFORE UPDATE ON runs
BEGIN SELECT RAISE(ABORT, 'audit records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'audit records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS findings_no_update BEFORE UPDATE ON findings
BEGIN SELECT RAISE(ABORT, 'audit records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS findings_no_delete BEFORE DELETE ON findings
BEGIN SELECT RAISE(ABORT, 'audit records are immutable'); END;
"""


class AuditStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, run: GapRun) -> str:
        payload = run.model_dump_json()
        sha = hashlib.sha256(payload.encode()).hexdigest()
        s = run.stats
        with self._conn() as c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                run.run_id, run.created_at, run.dossier_id, ",".join(run.markets), run.mode, run.backend,
                run.model, run.config_hash, run.corpus_hash, s.llm_calls, s.input_tokens, s.output_tokens,
                s.requirements_checked, s.requirements_inherited, s.duration_ms, sha, payload))
            c.executemany("INSERT INTO findings VALUES (?,?,?,?,?,?,?,?,?)", [
                (run.run_id, f.req_id, f.market, f.verdict.value, f.severity, f.check_type, int(f.inherited),
                 int(all(e.quote_verified for e in f.evidence)) if f.evidence else 0, json.dumps(f.flags))
                for f in run.findings])
        return sha

    def get(self, run_id: str) -> GapRun | None:
        with self._conn() as c:
            row = c.execute("SELECT payload, content_sha256 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row["payload"].encode()).hexdigest() != row["content_sha256"]:
            raise RuntimeError(f"run {run_id}: content hash mismatch (record was altered)")
        return GapRun.model_validate_json(row["payload"])

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id, created_at, dossier_id, markets, mode, backend, model, llm_calls, "
                "duration_ms FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def gap_summary(self, run_id: str) -> list[dict]:
        """Plain SQL view used by the API/dashboard: gap counts per market and severity."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT market, severity, verdict, COUNT(*) AS n FROM findings WHERE run_id=? "
                "GROUP BY market, severity, verdict ORDER BY market, severity, verdict", (run_id,)).fetchall()
        return [dict(r) for r in rows]
