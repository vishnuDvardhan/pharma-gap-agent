"""Provider-neutral LLM layer: Claude (Anthropic API), OpenAI, AWS Bedrock (Converse API), Ollama.

One neutral message format is used everywhere:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Each backend converts to its own wire format. Tool specs are JSON Schema:
    {"name": str, "description": str, "parameters": {...json schema...}}
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

EXTERNAL_BACKENDS = {"anthropic", "openai", "bedrock"}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class DataClassificationError(RuntimeError):
    """Raised when data of a given classification may not leave the machine."""


class LLM:
    name = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024, retries: int = 3):
        self.model, self.temperature, self.max_tokens, self.retries = model, temperature, max_tokens, retries

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._chat(system, messages, tools or [])
            except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{self.name} call failed after {self.retries} tries: {last}") from last

    def complete(self, system: str, user: str) -> LLMResponse:
        return self.chat(system, [{"role": "user", "content": user}])

    def _chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        raise NotImplementedError


# --------------------------------------------------------------------------- Claude
class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, model: str, **kw):
        super().__init__(model, **kw)
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)       # group parallel tool results
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def _chat(self, system, messages, tools):
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                                      system=system, messages=self.to_wire(messages))
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                               for t in tools]
        r = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        return LLMResponse(text, calls, r.usage.input_tokens, r.usage.output_tokens)


# --------------------------------------------------------------------------- OpenAI / Ollama
class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **kw):
        super().__init__(model, **kw)
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @staticmethod
    def to_wire(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                          "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                         for tc in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    def _chat(self, system, messages, tools):
        kwargs: dict[str, Any] = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                                      messages=self.to_wire(system, messages))
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        r = self.client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}"))
                 for tc in (msg.tool_calls or [])]
        usage = r.usage
        return LLMResponse(msg.content or "", calls,
                           getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)


class OllamaLLM(OpenAILLM):
    """Local models through Ollama's OpenAI-compatible endpoint. Data never leaves the machine."""
    name = "ollama"

    def __init__(self, model: str, base_url: str | None = None, **kw):
        super().__init__(model, base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                         api_key="ollama", **kw)


# --------------------------------------------------------------------------- AWS Bedrock
class BedrockLLM(LLM):
    """AWS Bedrock Converse API (works for Claude and other Bedrock models)."""
    name = "bedrock"

    def __init__(self, model: str, region: str | None = None, **kw):
        super().__init__(model, **kw)
        import boto3
        self.client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                content: list[dict] = []
                if m.get("content"):
                    content.append({"text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append({"toolUse": {"toolUseId": tc.id, "name": tc.name, "input": tc.arguments}})
                out.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                block = {"toolResult": {"toolUseId": m["tool_call_id"], "content": [{"text": m["content"]}]}}
                if out and out[-1]["role"] == "user" and "toolResult" in out[-1]["content"][0]:
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def _chat(self, system, messages, tools):
        kwargs: dict[str, Any] = dict(
            modelId=self.model, system=[{"text": system}], messages=self.to_wire(messages),
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )
        if tools:
            kwargs["toolConfig"] = {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                                            "inputSchema": {"json": t["parameters"]}}} for t in tools]}
        r = self.client.converse(**kwargs)
        content = r["output"]["message"]["content"]
        text = "".join(c.get("text", "") for c in content)
        calls = [ToolCall(c["toolUse"]["toolUseId"], c["toolUse"]["name"], c["toolUse"]["input"])
                 for c in content if "toolUse" in c]
        u = r.get("usage", {})
        return LLMResponse(text, calls, u.get("inputTokens", 0), u.get("outputTokens", 0))


# --------------------------------------------------------------------------- factory + governance gate
def make_llm(backend: str, model: str, data_classification: str, allowed_external: list[str], **kw) -> LLM | None:
    """Return an LLM client, or None for the offline backends.

    Responsible-AI gate: dossier data may only go to an external API if its classification is
    explicitly allowed (by default only synthetic / non-sensitive data). Confidential dossiers
    must use a local model (ollama) or the offline checker.
    """
    if backend.startswith("offline"):
        return None
    if backend in EXTERNAL_BACKENDS and data_classification not in allowed_external:
        raise DataClassificationError(
            f"Dossier classified '{data_classification}' may not be sent to external backend '{backend}'. "
            f"Allowed classifications: {allowed_external}. Use backend 'ollama' or 'offline-heuristic'.")
    cls = {"anthropic": AnthropicLLM, "openai": OpenAILLM, "bedrock": BedrockLLM, "ollama": OllamaLLM}.get(backend)
    if cls is None:
        raise ValueError(f"Unknown backend {backend!r}")
    return cls(model, **kw)
