"""Gap-check executor. Deterministic loop: requirements are the outer loop, so nothing is skipped.

diff mode: ICH core requirements are checked once and inherited by every market (unless a
           market overrides one). Only the regional delta is checked per market.
full mode: every requirement is checked again for every market (baseline for the cost ablation).
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import config_hash
from .guardrails import detect_injection
from .judge import JudgeResult, make_judge
from .llm import make_llm
from .loader import load_dossier, load_markets, load_product, load_requirements, tree_hash
from .pathway import decide_pathway, is_active
from .retrieval import Retriever, candidate_indices, requirement_query
from .rules import run_rule
from .schema import Dossier, Evidence, Finding, GapRun, Pathway, Requirement, RunStats, Verdict
from .vectorstore import DOSSIER_COLLECTION, index_dossier, make_vector_store


class GapChecker:
    def __init__(self, cfg: dict, dossier_id: str, llm=None):
        self.cfg = cfg
        self.data_dir = Path(cfg["data_dir"])
        self.dossier: Dossier = load_dossier(self.data_dir / "dossiers" / dossier_id)
        self.product = load_product(self.data_dir, self.dossier.product)
        self.market_cfg = load_markets(self.data_dir)
        rc = cfg["retrieval"]
        self.top_k, self.section_filter = rc["top_k"], rc["section_filter"]
        self.vector_store = make_vector_store(cfg, DOSSIER_COLLECTION)
        self.index_stats = index_dossier(self.vector_store, self.dossier) if self.vector_store else None
        self.retriever = Retriever(self.dossier.chunks, rc["method"], rc.get("embedder", "tfidf"),
                                   vector_store=self.vector_store)
        self.llm = llm if llm is not None else make_llm(
            cfg["backend"], cfg["model"], self.dossier.data_classification,
            cfg["governance"]["allowed_external_classifications"],
            temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 800))
        self.judge = make_judge(cfg["backend"], self.llm)

    # ---------------------------------------------------------------- one requirement
    def retrieve(self, req: Requirement):
        idx, fallback = candidate_indices(self.dossier.chunks, req, self.section_filter)
        return self.retriever.search(requirement_query(req), self.top_k, idx), fallback

    def check(self, req: Requirement, row_market: str, pathway: Pathway | None, inherited: bool) -> tuple[Finding, JudgeResult | None]:
        base = dict(req_id=req.id, market=row_market, title=req.title, severity=req.severity,
                    check_type=req.check_type, source=req.source, inherited=inherited,
                    overrides=req.overrides)
        active, why = is_active(req, pathway)
        if not active:
            return Finding(verdict=Verdict.NA, reasoning=why, **base), None

        if req.check_type == "rule":
            r = run_rule(req, self.dossier)
            return Finding(verdict=r.verdict, reasoning=r.reasoning, evidence=r.evidence, **base), None

        hits, fallback = self.retrieve(req)
        flags = ["searched_outside_expected_section"] if fallback and hits else []
        if req.check_type == "expert":
            ev = []
            for h in hits[:2]:
                first = h.chunk.text.strip().splitlines()[0]
                ev.append(Evidence(chunk_id=h.chunk.chunk_id, file=h.chunk.file, quote=first, quote_verified=True))
            return Finding(verdict=Verdict.EXPERT, flags=flags, evidence=ev,
                           reasoning="Judgement call reserved for a qualified reviewer; evidence attached.", **base), None

        jr = self.judge.judge(req, hits)
        if any(detect_injection(h.chunk.text) for h in hits) and "injection_detected" not in jr.flags:
            jr.flags.append("injection_detected")
        return Finding(verdict=jr.verdict, reasoning=jr.reasoning, evidence=jr.evidence,
                       flags=flags + jr.flags, **base), jr

    # ---------------------------------------------------------------- full run
    def run(self, markets: list[str], mode: str | None = None) -> GapRun:
        mode = mode or self.cfg.get("mode", "diff")
        t0 = time.perf_counter()
        reqs = load_requirements(self.data_dir, markets)
        pathways = {m: decide_pathway(m, self.product, self.market_cfg[m]) for m in markets}
        stats = RunStats()
        findings: list[Finding] = []

        def add(f: Finding, jr: JudgeResult | None, inherited_count: int = 0):
            findings.append(f)
            stats.requirements_checked += 1
            stats.requirements_inherited += inherited_count
            if jr:
                stats.judge_calls += 1
                stats.llm_calls += jr.llm_calls
                stats.input_tokens += jr.input_tokens
                stats.output_tokens += jr.output_tokens

        overridden = {m: {r.overrides for r in reqs[m] if r.overrides} for m in markets}
        if mode == "diff":
            for r in reqs["ICH"]:
                users = [m for m in markets if r.id not in overridden[m]]
                if users:
                    f, jr = self.check(r, "ICH", None, inherited=True)
                    add(f, jr, inherited_count=len(users) - 1)
        for m in markets:
            if mode == "full":
                for r in reqs["ICH"]:
                    if r.id not in overridden[m]:
                        f, jr = self.check(r, m, pathways[m], inherited=False)
                        add(f, jr)
            for r in reqs[m]:
                f, jr = self.check(r, m, pathways[m], inherited=False)
                add(f, jr)

        stats.duration_ms = int((time.perf_counter() - t0) * 1000)
        return GapRun(
            run_id=uuid.uuid4().hex[:12],
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            dossier_id=self.dossier.dossier_id, markets=markets, mode=mode,
            backend=self.cfg["backend"], model=self.cfg["model"] if self.llm else self.cfg["backend"],
            config_hash=config_hash(self.cfg),
            corpus_hash=tree_hash(self.data_dir / "requirements", Path(self.dossier.root)),
            pathways=pathways, findings=findings, stats=stats,
        )


def market_view(run: GapRun, market: str) -> list[Finding]:
    """Effective findings for one market: inherited core rows (minus overrides) + market rows."""
    own = [f for f in run.findings if f.market == market]
    overridden = {f.overrides for f in own if f.overrides}
    core = [f for f in run.findings if f.market == "ICH" and f.req_id not in overridden]
    return core + own


def run_gap_check(cfg: dict, dossier_id: str, markets: list[str], mode: str | None = None, llm=None) -> GapRun:
    return GapChecker(cfg, dossier_id, llm=llm).run([m.upper() for m in markets], mode)
