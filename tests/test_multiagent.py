"""Multi-agent crew: orchestration, least privilege, parallel specialists, output validation, review loop."""
from __future__ import annotations

import json
import threading
import time

import yaml

from gapagent.audit import AuditStore
from gapagent.config import ROOT
from gapagent.llm import LLMResponse, ToolCall
from gapagent.multiagent import ROLE_TOOLS, SPECIALIST_SYSTEM, SUPERVISOR_SYSTEM, LaunchReadinessCrew, run_multi_agent
from gapagent.pipeline import market_view

from .conftest import FakeLLM

GOLD = yaml.safe_load((ROOT / "data" / "gold" / "gaps_gold.yaml").read_text())["verdicts"]


def _agents(res):
    return [t["agent"] for t in res.trace]


def test_offline_crew_end_to_end(cfg):
    res = run_multi_agent("What blocks a launch in India and Japan?", cfg, "zelvorin_global")
    assert res.plan["markets"] == ["IN", "JP"]
    assert set(res.briefs) == {"IN", "JP"}
    # every role took part, in the right order
    order = list(dict.fromkeys(_agents(res)))
    assert order[:4] == ["supervisor", "regulatory_intel", "gap_check_executor", order[3]]
    assert {"specialist:IN", "specialist:JP", "reviewer"} <= set(order)
    assert res.review.approved and res.revisions == 0 and not res.flags
    # every gold gap for these markets is in the answer, with an owner and an action
    for m in ("IN", "JP"):
        gold_gaps = {r for r, v in GOLD["zelvorin_global"][m].items() if v in ("Missing", "Partial")}
        brief_ids = {a.req_id for a in res.briefs[m].actions}
        assert gold_gaps <= brief_ids
        assert all(i in res.answer for i in brief_ids)
        assert all(a.owner and a.action and a.location for a in res.briefs[m].actions)
    assert "qualified regulatory reviewer" in res.answer
    pr = [a.priority for a in res.briefs["IN"].actions]
    assert pr == sorted(pr)                      # P1 before P2 before P3


def test_specialists_only_get_their_tools(cfg):
    crew = LaunchReadinessCrew(cfg, "zelvorin_global")
    assert "run_gap_check" not in ROLE_TOOLS["specialist"]
    out = crew._tool("specialist:IN", "specialist", "run_gap_check", {"markets": ["IN"]})
    assert "not permitted" in out["error"]
    out = crew._tool("regulatory_intel", "regulatory_intel", "get_finding", {"run_id": "x", "req_id": "y"})
    assert "not permitted" in out["error"]


def test_specialists_run_in_parallel(cfg, monkeypatch):
    seen, active, peak = [], [0], [0]
    lock = threading.Lock()
    orig = LaunchReadinessCrew.specialist

    def slow(self, market, *a):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        try:
            return orig(self, market, *a)
        finally:
            with lock:
                active[0] -= 1
                seen.append(market)

    monkeypatch.setattr(LaunchReadinessCrew, "specialist", slow)
    res = run_multi_agent("Launch in the US, India and Japan?", cfg, "zelvorin_global")
    assert sorted(seen) == ["IN", "JP", "US"] and peak[0] == 3
    assert list(res.briefs) == ["US", "IN", "JP"]           # results keep plan order


def _scripted_llm(specialist_reply, synth_replies, review_replies, plan=None):
    """Route each call to the right scripted reply by the agent's system prompt."""
    synth, review = list(synth_replies), list(review_replies)

    def fn(system, messages, tools):
        if system == SUPERVISOR_SYSTEM:
            return LLMResponse(json.dumps(plan or {"markets": ["IN"], "focus": "India readiness"}), input_tokens=50)
        if system.startswith("You are the regulatory-intelligence"):
            if len(messages) == 1:
                return LLMResponse("", [ToolCall("p1", "decide_pathway", {"market": "IN"})])
            return LLMResponse("IN: waiver pathway (Rule 101).")
        if "market specialist" in system:
            if len(messages) == 1:     # first turn: use a tool, including one it is NOT allowed to use
                return LLMResponse("", [ToolCall("s1", "search_dossier", {"query": "PSUR", "region": "in"}),
                                        ToolCall("s2", "run_gap_check", {"markets": ["IN"]})])
            return LLMResponse(specialist_reply(messages), input_tokens=300, output_tokens=80)
        if system.startswith("You are the synthesizer"):
            reply = synth.pop(0)
            return LLMResponse(reply(messages) if callable(reply) else reply)
        if system.startswith("You are the compliance reviewer"):
            return LLMResponse(review.pop(0))
        # gap-check judge calls
        return LLMResponse('{"verdict":"Missing","chunk_id":null,"quote":null,"reasoning":"-"}')

    return FakeLLM(fn=fn)


def test_llm_specialist_output_is_validated_in_code(cfg):
    cfg["backend"] = "anthropic"

    def specialist(messages):
        task = json.loads(messages[0]["content"])
        ids = [g["req_id"] for g in task["gaps"]]
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert "not permitted" in tool_msgs[1]["content"]          # run_gap_check was refused
        return json.dumps({"summary": "India needs work.", "actions": [
            {"req_id": ids[0], "owner": "Labeling", "action": "Do the thing", "location": "m1/in"},
            {"req_id": "IN-M1-MADE-UP", "owner": "CMC / Quality", "action": "x", "location": "y"},
        ]})

    llm = _scripted_llm(specialist, [_answer_from_briefs], ['{"approved": true, "issues": []}'])
    store = AuditStore(cfg["db_path"])
    res = run_multi_agent("India?", cfg, "zelvorin_global", store=store, llm=llm)
    b = res.briefs["IN"]
    ids = {a.req_id for a in b.actions}
    assert "IN-M1-MADE-UP" not in ids                                 # invented ID dropped
    assert any(f.startswith("dropped_invented_ids") for f in b.flags)
    assert any(f.startswith("filled_missed_gaps") for f in b.flags)   # missed gaps added back
    executor = {f.req_id: f.verdict.value for f in market_view(store.get(res.run_id), "IN")}
    for a in b.actions:                                               # verdicts come from the executor
        assert a.verdict == executor[a.req_id]
    assert next(a for a in b.actions if a.owner == "Labeling").action == "Do the thing"   # LLM text kept
    assert res.plan["planner"] == "llm" and res.review.approved and res.revisions == 0
    assert res.usage["specialist:IN"].llm_calls == 2 and res.usage["specialist:IN"].input_tokens == 300


def _answer_from_briefs(messages):
    briefs = json.loads(messages[0]["content"].split("\n\nYour previous draft")[0])["briefs"]
    ids = [a["req_id"] for b in briefs.values() for a in b["actions"]]
    return f"Gaps: {', '.join(ids)}. A qualified regulatory reviewer must confirm."


def test_reviewer_rejects_then_synthesizer_revises(cfg):
    cfg["backend"] = "anthropic"

    def specialist(messages):
        return json.dumps({"summary": "s", "actions": []})

    llm = _scripted_llm(specialist,
                        ["India looks fine, also see IN-FAKE-01.", _answer_from_briefs],
                        ['{"approved": true, "issues": []}', '{"approved": true, "issues": []}'])
    res = run_multi_agent("India?", cfg, "zelvorin_global", llm=llm)
    assert res.revisions == 1
    assert res.review.approved
    first_verdict = next(t for t in res.trace if t.get("event") == "verdict")
    assert any("IN-FAKE-01" in i for i in first_verdict["issues"])       # caught by the grounding check
    assert any("missing from the answer" in i for i in first_verdict["issues"])
    revise_call = [c for c in llm.calls if c["system"].startswith("You are the synthesizer")][1]
    assert "reviewer rejected it" in revise_call["messages"][0]["content"]


def test_unfixable_draft_falls_back_to_grounded_answer(cfg):
    cfg["backend"] = "anthropic"
    llm = _scripted_llm(lambda m: "not json", ["bad draft", "still bad"],
                        ['{"approved": false, "issues": ["x"]}'] * 2)
    res = run_multi_agent("India?", cfg, "zelvorin_global", llm=llm)
    assert "review_failed:used_rule_based_answer" in res.flags
    assert "specialist:IN:specialist_output_unparseable:used_rule_based_plan" in res.flags
    assert "IN-Q-STAB-IVB" in res.answer and "qualified regulatory reviewer" in res.answer


def test_bad_plan_falls_back_to_rules(cfg):
    cfg["backend"] = "anthropic"
    llm = _scripted_llm(lambda m: "{}", ["a"], ['{"approved": true}'], plan={"markets": ["BR"]})
    crew = LaunchReadinessCrew(cfg, "zelvorin_global", llm=llm)
    p = crew.plan("What about Japan?")
    assert p == {"markets": ["JP"], "focus": "What about Japan?", "planner": "rules"}
    assert "plan_fallback" in crew.flags


def test_specialist_prompt_forbids_changing_verdicts():
    s = SPECIALIST_SYSTEM.format(market="IN", owners="x")
    assert "MUST NOT change any" in s and "untrusted" in s
