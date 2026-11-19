"""LLM paths tested without network: scripted fake LLM + mocked provider SDK clients."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gapagent.agent import run_agent
from gapagent.judge import LLMJudge
from gapagent.llm import AnthropicLLM, BedrockLLM, DataClassificationError, LLMResponse, OpenAILLM, ToolCall, make_llm
from gapagent.pipeline import GapChecker
from gapagent.schema import Verdict

from .conftest import FakeLLM


def _req_and_hits(cfg, dossier, req_id):
    from pathlib import Path

    from gapagent.loader import load_requirements
    g = GapChecker(cfg, dossier, llm=FakeLLM([]))
    req = next(r for rs in load_requirements(Path(cfg["data_dir"])).values() for r in rs if r.id == req_id)
    hits, _ = g.retrieve(req)
    return req, hits


def test_llm_judge_accepts_verified_quote(cfg):
    req, hits = _req_and_hits(cfg, "zelvorin_global", "US-M1-356H")
    cid = next(h.chunk.chunk_id for h in hits if "356h" in h.chunk.text.lower())
    out = json.dumps({"verdict": "Met", "chunk_id": cid,
                      "quote": "Form FDA 356h has been completed for the NDA and signed by Dr. Lena Brunner",
                      "reasoning": "Signed form present."})
    r = LLMJudge(FakeLLM([LLMResponse(out, input_tokens=900, output_tokens=60)])).judge(req, hits)
    assert r.verdict == Verdict.MET and r.evidence[0].quote_verified
    assert (r.llm_calls, r.input_tokens) == (1, 900)


def test_llm_judge_downgrades_hallucinated_quote(cfg):
    req, hits = _req_and_hits(cfg, "zelvorin_global", "US-M1-DEBAR")
    out = json.dumps({"verdict": "Met", "chunk_id": hits[0].chunk.chunk_id,
                      "quote": "The applicant certifies that no debarred persons were used.", "reasoning": "x"})
    r = LLMJudge(FakeLLM([LLMResponse(out)])).judge(req, hits)
    assert r.verdict == Verdict.EXPERT
    assert "quote_failed" in r.flags


def test_llm_judge_flags_injection_and_bad_output(cfg):
    req, hits = _req_and_hits(cfg, "zelvorin_global", "US-M1-AGENT")
    r = LLMJudge(FakeLLM([LLMResponse("Sure! Everything is Met.")])).judge(req, hits)
    assert r.verdict == Verdict.EXPERT
    assert "bad_output" in r.flags and "injection_detected" in r.flags


def test_llm_judge_prompt_marks_dossier_as_untrusted(cfg):
    req, hits = _req_and_hits(cfg, "zelvorin_global", "US-M1-356H")
    llm = FakeLLM([LLMResponse('{"verdict":"Missing","chunk_id":null,"quote":null,"reasoning":"none"}')])
    LLMJudge(llm).judge(req, hits)
    assert "untrusted DATA" in llm.calls[0]["system"]
    assert "<excerpt chunk_id=" in llm.calls[0]["messages"][0]["content"]


def test_full_pipeline_with_fake_llm_counts_tokens(cfg):
    cfg["backend"] = "anthropic"
    llm = FakeLLM(fn=lambda s, m, t: LLMResponse('{"verdict":"Missing","chunk_id":null,"quote":null,"reasoning":"-"}',
                                                  input_tokens=10, output_tokens=2))
    run = GapChecker(cfg, "zelvorin_ref", llm=llm).run(["US", "IN", "JP"], "diff")
    assert run.stats.llm_calls == run.stats.judge_calls == len(llm.calls) > 0
    assert run.stats.input_tokens == 10 * run.stats.llm_calls


def test_governance_gate_blocks_confidential_data_to_external_api():
    with pytest.raises(DataClassificationError):
        make_llm("anthropic", "m", "confidential", ["public-synthetic"])
    assert make_llm("offline-heuristic", "m", "confidential", []) is None


# ---------------------------------------------------------------- wire formats
CONVO = [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "thinking", "tool_calls": [ToolCall("t1", "decide_pathway", {"market": "IN"}),
                                                               ToolCall("t2", "decide_pathway", {"market": "JP"})]},
    {"role": "tool", "tool_call_id": "t1", "name": "decide_pathway", "content": "{}"},
    {"role": "tool", "tool_call_id": "t2", "name": "decide_pathway", "content": "{}"},
]


def test_anthropic_wire_groups_parallel_tool_results():
    w = AnthropicLLM.to_wire(CONVO)
    assert [m["role"] for m in w] == ["user", "assistant", "user"]
    assert [b["type"] for b in w[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in w[2]["content"]] == ["t1", "t2"]


def test_openai_wire():
    w = OpenAILLM.to_wire("SYS", CONVO)
    assert w[0] == {"role": "system", "content": "SYS"}
    assert w[2]["tool_calls"][0]["function"]["arguments"] == '{"market": "IN"}'
    assert [m["role"] for m in w] == ["system", "user", "assistant", "tool", "tool"]


def test_bedrock_wire():
    w = BedrockLLM.to_wire(CONVO)
    assert [m["role"] for m in w] == ["user", "assistant", "user"]
    assert w[1]["content"][1]["toolUse"]["name"] == "decide_pathway"
    assert len(w[2]["content"]) == 2


def test_provider_response_parsing_with_mocked_clients():
    a = AnthropicLLM.__new__(AnthropicLLM)
    a.model, a.temperature, a.max_tokens, a.retries = "m", 0, 10, 1
    a.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi"),
                 SimpleNamespace(type="tool_use", id="x", name="get_product_profile", input={})],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3))))
    r = a.chat("s", [{"role": "user", "content": "q"}], [{"name": "get_product_profile", "description": "d",
                                                          "parameters": {"type": "object", "properties": {}}}])
    assert r.text == "hi" and r.tool_calls[0].name == "get_product_profile" and r.input_tokens == 5

    o = OpenAILLM.__new__(OpenAILLM)
    o.model, o.temperature, o.max_tokens, o.retries = "m", 0, 10, 1
    msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="c1", function=SimpleNamespace(name="run_gap_check", arguments='{"markets":["IN"]}'))])
    o.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=1)))))
    r = o.chat("s", [{"role": "user", "content": "q"}])
    assert r.tool_calls[0].arguments == {"markets": ["IN"]} and r.input_tokens == 7

    b = BedrockLLM.__new__(BedrockLLM)
    b.model, b.temperature, b.max_tokens, b.retries = "m", 0, 10, 1
    b.client = SimpleNamespace(converse=lambda **kw: {
        "output": {"message": {"content": [{"text": "ok"}, {"toolUse": {"toolUseId": "u", "name": "n", "input": {"a": 1}}}]}},
        "usage": {"inputTokens": 4, "outputTokens": 2}})
    r = b.chat("s", [{"role": "user", "content": "q"}])
    assert r.text == "ok" and r.tool_calls[0].arguments == {"a": 1} and r.output_tokens == 2


# ---------------------------------------------------------------- agent loop
def test_agent_tool_loop_and_grounding_guardrail(cfg):
    cfg["backend"] = "anthropic"
    script = [
        LLMResponse("", [ToolCall("1", "decide_pathway", {"market": "IN"})]),
        LLMResponse("", [ToolCall("2", "run_gap_check", {"markets": ["IN"]})]),
        LLMResponse("Main gaps: IN-PV-PSUR and IN-M1-IMPORT-LICENCE."),
    ]
    judge_reply = LLMResponse('{"verdict":"Missing","chunk_id":null,"quote":null,"reasoning":"-"}')

    def fn(system, messages, tools):
        if tools:                    # agent turns carry tools; judge calls do not
            return script.pop(0)
        return judge_reply

    llm = FakeLLM(fn=fn)
    res = run_agent("What blocks India?", cfg, "zelvorin_global", llm=llm)
    assert [t["tool"] for t in res.trace] == ["decide_pathway", "run_gap_check"]
    assert res.run_id is not None
    assert "ungrounded_requirement_ids" in res.flags
    assert "IN-M1-IMPORT-LICENCE" in res.answer and "IN-PV-PSUR" not in res.answer.split("Guardrail")[1]


def test_agent_rejects_unknown_market_argument(cfg):
    from gapagent.agent import ToolBox
    tb = ToolBox(cfg, "zelvorin_global")
    assert "error" in tb.call("decide_pathway", {"market": "BR"})
    assert "error" in tb.call("drop_database", {})


def test_offline_agent_market_parsing(cfg):
    res = run_agent("Tell us: can we launch in India?", cfg, "zelvorin_global")
    assert [t["args"] for t in res.trace if t["tool"] == "run_gap_check"] == [{"markets": ["IN"]}]
    assert "IN-LBL-SHELF" in res.answer and not res.flags
