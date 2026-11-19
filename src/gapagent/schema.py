"""Typed data model. Everything that crosses a module boundary is a pydantic model."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    MET = "Met"
    PARTIAL = "Partial"
    MISSING = "Missing"
    EXPERT = "Needs expert review"
    NA = "Not applicable"


GAP_VERDICTS = {Verdict.PARTIAL, Verdict.MISSING}


class Source(BaseModel):
    doc: str
    clause: str


class RuleSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Requirement(BaseModel):
    id: str
    market: str                      # "ICH" for core, else "US" / "IN" / "JP"
    title: str
    ctd_section: str
    region: str | None = None        # restricts Module 1 evidence to m1/<region>
    source: Source
    text: str
    evidence_criteria: str
    check_type: Literal["llm", "rule", "expert"]
    rule: RuleSpec | None = None
    keywords: list[list[str]] = Field(default_factory=list)
    severity: Literal["critical", "major", "minor"] = "major"
    applies_if: dict[str, Any] = Field(default_factory=dict)
    overrides: str | None = None
    verified: bool = False


class Chunk(BaseModel):
    chunk_id: str
    dossier_id: str
    file: str                        # file stem, e.g. "3.2.P.8.3_stability_data"
    path: str                        # relative path inside the dossier
    module: str                      # "m1".."m5"
    region: str | None               # only for m1/<region>
    section: str                     # deepest CTD section number known for this chunk
    heading: str
    text: str
    line_start: int
    line_end: int


class Dossier(BaseModel):
    dossier_id: str
    title: str
    product: str
    reference_market: str
    data_classification: str
    root: str
    chunks: list[Chunk]
    files: dict[str, str]            # file stem -> full text


class Product(BaseModel):
    product_id: str
    name: str
    active_substance: str
    dosage_form: str
    route: str
    new_active_substance: bool
    applicant: dict[str, str]
    manufacturing_sites: list[dict[str, str]]
    approvals: list[dict[str, str]]
    data_classification: str


class Pathway(BaseModel):
    market: str
    name: str
    facts: dict[str, Any]            # values that `applies_if` conditions are matched against
    rationale: list[str]


class Evidence(BaseModel):
    chunk_id: str | None = None
    file: str | None = None
    quote: str | None = None
    quote_verified: bool = False


class Finding(BaseModel):
    req_id: str
    market: str                      # market the row is shown under ("ICH" for inherited core)
    title: str
    severity: str
    check_type: str
    verdict: Verdict
    reasoning: str
    evidence: list[Evidence] = Field(default_factory=list)
    source: Source
    inherited: bool = False
    overrides: str | None = None    # core requirement replaced by this market row
    flags: list[str] = Field(default_factory=list)   # e.g. "quote_failed", "injection_detected"


class RunStats(BaseModel):
    judge_calls: int = 0            # evidence judgements (LLM or offline) actually executed
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    requirements_checked: int = 0
    requirements_inherited: int = 0
    duration_ms: int = 0


class GapRun(BaseModel):
    run_id: str
    created_at: str
    dossier_id: str
    markets: list[str]
    mode: Literal["full", "diff"]
    backend: str
    model: str
    config_hash: str
    corpus_hash: str
    pathways: dict[str, Pathway]
    findings: list[Finding]
    stats: RunStats
