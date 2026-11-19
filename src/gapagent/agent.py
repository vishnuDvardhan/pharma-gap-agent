"""Launch-readiness agent: answers questions like "What blocks a launch of Zelvorin in India and Japan?"

Design choice: the agent PLANS and EXPLAINS; it never decides verdicts. Verdicts come only from
the deterministic gap-check executor (pipeline.py), which the agent calls as a tool.

Guardrails: tool allowlist + argument validation, max steps, every requirement ID in the
final answer must come from a tool result (else the answer is flagged).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .audit import AuditStore
from .guardrails import ungrounded_ids
from .llm import LLM, make_llm
from .loader import load_markets, load_product, load_requirements
from .pathway import decide_pathway
from .pipeline import market_view, run_gap_check
from .retrieval import BM25
from .schema import GAP_VERDICTS, GapRun, Verdict

TOOLS = [
    {"name": "get_product_profile", "description": "Product facts: active substance, applicant, sites, existing approvals.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "decide_pathway", "description": "Rule-based regulatory pathway for one target market (US, IN or JP).",
     "parameters": {"type": "object", "properties": {"market": {"type": "string", "enum": ["US", "IN", "JP"]}},
                    "required": ["market"]}},
    {"name": "run_gap_check",
     "description": "Run the deterministic gap check of the dossier against the target markets. Returns run_id and the gaps per market.",
     "parameters": {"type": "object", "properties": {
         "markets": {"type": "array", "items": {"type": "string", "enum": ["US", "IN", "JP"]}}},
         "required": ["markets"]}},
    {"name": "get_finding", "description": "Full detail (reasoning, evidence quotes, source clause) for one requirement in a run.",
     "parameters": {"type": "object", "properties": {"run_id": {"type": "string"}, "req_id": {"type": "string"}},
                    "required": ["run_id", "req_id"]}},
    {"name": "search_requirements",
     "description": "Search the regulatory requirement library (ICH core + market overlays) by free text.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                     "market": {"type": "string", "enum": ["ICH", "US", "IN", "JP"]}},
                    "required": ["query"]}},
]

AGENT_SYSTEM = """You are a regulatory launch-readiness assistant for a pharmaceutical company.
You help a regulatory affairs team see what is missing before a product can be filed in new markets.

Rules:
- Use the tools. Never invent verdicts, requirements or regulation clauses.
- Gap verdicts come ONLY from run_gap_check / get_finding. Do not change them.
- Cite requirement IDs (e.g. IN-Q-STAB-IVB) and their source document for every gap you mention.
- Put critical gaps first. Mention items that need expert review.
- Say clearly that this is a prototype on synthetic data and a qualified reviewer must confirm.
- Be short and concrete."""

MARKET_WORDS = {  # (case-insensitive names, case-sensitive codes)
    "US": (r"\b(usa|united states|america|fda)\b|u\.s\.", r"\bUS\b"),
    "IN": (r"\b(india|indian|cdsco)\b", r"\bIN\b"),
    "JP": (r"\b(japan|japanese|pmda|mhlw)\b", r"\bJP\b"),
}


@dataclass
class AgentResult:
    answer: str
    run_id: str | None
    trace: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    llm_calls: int = 0


class ToolBox:
    def __init__(self, cfg: dict, dossier_id: str, store: AuditStore | None = None, llm: LLM | None = None):
        self.cfg, self.dossier_id, self.store, self.llm = cfg, dossier_id, store, llm
        self.data_dir = Path(cfg["data_dir"])
        self.market_cfg = load_markets(self.data_dir)
        from .loader import load_dossier
        self.dossier_product = load_dossier(self.data_dir / "dossiers" / dossier_id).product
        self.product = load_product(self.data_dir, self.dossier_product)
        self.runs: dict[str, GapRun] = {}           # short-term memory for this conversation
        self.seen_ids: set[str] = set()             # requirement IDs returned by tools (grounding set)
        reqs = load_requirements(self.data_dir)
        self.req_list = [r for rs in reqs.values() for r in rs]
        self.req_index = BM25([f"{r.id} {r.title} {r.text} {r.source.doc} {r.source.clause}" for r in self.req_list])

    def call(self, name: str, args: dict) -> dict:
        if name not in {t["name"] for t in TOOLS}:
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, f"_t_{name}")(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except ValueError as e:
            return {"error": str(e)}

    def _check_market(self, m: str) -> str:
        m = m.upper()
        if m not in self.market_cfg:
            raise ValueError(f"unknown market {m}; supported: {sorted(self.market_cfg)}")
        return m

    def _t_get_product_profile(self) -> dict:
        return self.product.model_dump()

    def _t_decide_pathway(self, market: str) -> dict:
        m = self._check_market(market)
        return decide_pathway(m, self.product, self.market_cfg[m]).model_dump()

    def _t_run_gap_check(self, markets: list[str]) -> dict:
        ms = [self._check_market(m) for m in markets]
        run = run_gap_check(self.cfg, self.dossier_id, ms, llm=self.llm)
        self.runs[run.run_id] = run
        if self.store:
            self.store.save(run)
        out = {"run_id": run.run_id, "markets": {}}
        for m in ms:
            view = market_view(run, m)
            gaps = [{"req_id": f.req_id, "title": f.title, "verdict": f.verdict.value, "severity": f.severity,
                     "source": f"{f.source.doc}, {f.source.clause}"}
                    for f in view if f.verdict in GAP_VERDICTS]
            gaps.sort(key=lambda g: ({"critical": 0, "major": 1, "minor": 2}[g["severity"]], g["req_id"]))
            expert = [f.req_id for f in view if f.verdict == Verdict.EXPERT]
            self.seen_ids.update(g["req_id"] for g in gaps)
            self.seen_ids.update(expert)
            out["markets"][m] = {"pathway": run.pathways[m].name, "gaps": gaps, "needs_expert_review": expert,
                                 "met": sum(1 for f in view if f.verdict == Verdict.MET)}
        return out

    def _t_get_finding(self, run_id: str, req_id: str) -> dict:
        run = self.runs.get(run_id) or (self.store.get(run_id) if self.store else None)
        if run is None:
            raise ValueError(f"unknown run_id {run_id}")
        f = next((x for x in run.findings if x.req_id == req_id), None)
        if f is None:
            raise ValueError(f"{req_id} not in run {run_id}")
        self.seen_ids.add(req_id)
        return f.model_dump(mode="json")

    def _t_search_requirements(self, query: str, market: str | None = None, k: int = 5) -> dict:
        scores = self.req_index.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits = [self.req_list[i] for i in order if scores[i] > 0 and (market is None or self.req_list[i].market == market)][:k]
        self.seen_ids.update(r.id for r in hits)
        return {"results": [{"req_id": r.id, "market": r.market, "title": r.title, "text": r.text,
                             "source": f"{r.source.doc}, {r.source.clause}", "verified": r.verified} for r in hits]}


def _short(obj: dict, n: int = 600) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "…"


def run_agent(question: str, cfg: dict, dossier_id: str, store: AuditStore | None = None,
              llm: LLM | None = None) -> AgentResult:
    if llm is None and not cfg["backend"].startswith("offline"):
        from .loader import load_dossier
        d = load_dossier(Path(cfg["data_dir"]) / "dossiers" / dossier_id)
        llm = make_llm(cfg["backend"], cfg["model"], d.data_classification,
                       cfg["governance"]["allowed_external_classifications"],
                       temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 800))
    tb = ToolBox(cfg, dossier_id, store, llm)
    if llm is None:
        return _offline_agent(question, tb)

    messages: list[dict] = [{"role": "user", "content": question}]
    trace: list[dict] = []
    calls = 0
    for _ in range(cfg.get("agent", {}).get("max_steps", 8)):
        r = llm.chat(AGENT_SYSTEM, messages, TOOLS)
        calls += 1
        if not r.tool_calls:
            return _finish(r.text, tb, trace, calls)
        messages.append({"role": "assistant", "content": r.text, "tool_calls": r.tool_calls})
        for tc in r.tool_calls:
            result = tb.call(tc.name, tc.arguments)
            trace.append({"tool": tc.name, "args": tc.arguments, "result": _short(result)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                             "content": json.dumps(result, default=str)})
    return AgentResult("Stopped: step limit reached before a final answer.", _last_run(tb), trace,
                       ["max_steps_reached"], calls)


def _last_run(tb: ToolBox) -> str | None:
    return next(reversed(tb.runs), None) if tb.runs else None


def _finish(answer: str, tb: ToolBox, trace: list[dict], calls: int) -> AgentResult:
    flags = []
    bad = ungrounded_ids(answer, tb.seen_ids)
    if bad:
        flags.append("ungrounded_requirement_ids")
        answer += f"\n\n⚠ Guardrail: these IDs were not returned by any tool and may be invented: {', '.join(bad)}."
    return AgentResult(answer, _last_run(tb), trace, flags, calls)


def _offline_agent(question: str, tb: ToolBox) -> AgentResult:
    """Deterministic planner used without an LLM (CI, demos with no API key). Same tools, same trace."""
    markets = [m for m, (names, code) in MARKET_WORDS.items()
               if re.search(names, question, re.IGNORECASE) or re.search(code, question)] or list(tb.market_cfg)
    trace: list[dict] = []

    def step(name: str, args: dict) -> dict:
        res = tb.call(name, args)
        trace.append({"tool": name, "args": args, "result": _short(res)})
        return res

    step("get_product_profile", {})
    for m in markets:
        step("decide_pathway", {"market": m})
    res = step("run_gap_check", {"markets": markets})
    lines = [f"Launch readiness for {tb.product.name} (run {res['run_id']}):"]
    for m in markets:
        info = res["markets"][m]
        crit = [g for g in info["gaps"] if g["severity"] == "critical"]
        lines.append(f"\n{m} — pathway: {info['pathway']}. {len(info['gaps'])} gaps ({len(crit)} critical), "
                     f"{info['met']} met.")
        for g in info["gaps"]:
            lines.append(f"  - [{g['severity']}] {g['req_id']} {g['title']}: {g['verdict']} (source: {g['source']})")
        if info["needs_expert_review"]:
            lines.append(f"  Needs expert review: {', '.join(info['needs_expert_review'])}")
    lines.append("\nPrototype on synthetic data; a qualified regulatory reviewer must confirm every finding.")
    return _finish("\n".join(lines), tb, trace, 0)
