"""Guardrails that run in code, independent of the model.

1. Quote verification: every quote the model cites must exist in the cited chunk.
2. Prompt-injection detection in dossier text (dossiers are untrusted input).
3. Answer grounding for the agent: requirement IDs in the answer must exist in the run.
"""
from __future__ import annotations

import re
import unicodedata

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above)",
    r"mark (every|all) requirements? .* as met",
    r"you are now",
    r"system prompt",
    r"automated reviewers?",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
REQ_ID_RE = re.compile(r"\b(?:ICH|US|IN|JP)-[A-Z0-9]+(?:-[A-Z0-9]+)*\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("|", " ").replace("*", " ")
    text = re.sub(r"[‐-―−]", "-", text)                # unify dashes
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(quote: str | None, source_text: str, min_len: int = 8) -> bool:
    """True if the (whitespace/markup-normalised) quote is a substring of the source."""
    if not quote:
        return False
    q = normalize(quote).strip(" .\"'")
    return len(q) >= min_len and q in normalize(source_text)


def detect_injection(text: str) -> list[str]:
    return [m.group(0) for m in _INJ.finditer(text)]


def ungrounded_ids(answer: str, known_ids: set[str]) -> list[str]:
    """Requirement IDs mentioned in an answer that do not exist in the run results."""
    return sorted({m for m in REQ_ID_RE.findall(answer) if m not in known_ids})
