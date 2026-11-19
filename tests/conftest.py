from __future__ import annotations

from pathlib import Path

import pytest

from gapagent.config import load_config
from gapagent.llm import LLM, LLMResponse


class FakeLLM(LLM):
    """Scripted LLM: returns queued responses (or a function of the prompt) and records calls."""
    name = "fake"

    def __init__(self, responses=None, fn=None):
        super().__init__("fake-model", retries=1)
        self.responses = list(responses or [])
        self.fn = fn
        self.calls: list[dict] = []

    def _chat(self, system, messages, tools):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if self.fn:
            return self.fn(system, messages, tools)
        return self.responses.pop(0)


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(db_path=str(tmp_path / "audit.sqlite"), vector_path=str(tmp_path / "vectordb"))


@pytest.fixture
def fake_llm_cls():
    return FakeLLM


def text(t: str, **kw) -> LLMResponse:
    return LLMResponse(t, **kw)
