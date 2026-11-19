"""Deterministic checks. Anything that is arithmetic or a table lookup is done here, not by an LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import section_matches
from .schema import Dossier, Evidence, Requirement, Verdict

COND_RE = re.compile(r"(\d+)\s*°\s*C[^/\n]*/\s*(\d+)\s*%\s*RH", re.IGNORECASE)
MONTH_RE = re.compile(r"^(\d+)\s*m$", re.IGNORECASE)
CONCLUSION_RE = re.compile(r"shelf life\s*\(zone\s*([ivab]+)[^)]*\)\s*:\s*(\d+)\s*months", re.IGNORECASE)
LABEL_SHELF_RE = re.compile(r"shelf[ -]life[^0-9\n]{0,30}?(\d+)\s*months", re.IGNORECASE)


@dataclass
class StabilityStudy:
    kind: str                     # "long-term" | "accelerated"
    temp: int
    rh: int
    heading: str
    file: str
    batches: dict[str, int] = field(default_factory=dict)   # batch -> last month with data


@dataclass
class RuleResult:
    verdict: Verdict
    reasoning: str
    evidence: list[Evidence]


def parse_stability(dossier: Dossier) -> list[StabilityStudy]:
    studies: list[StabilityStudy] = []
    for c in dossier.chunks:
        if not section_matches("3.2.P.8", c.section):
            continue
        cm = COND_RE.search(c.heading)
        if not cm:
            continue
        kind = "accelerated" if "accelerated" in c.heading.lower() else "long-term"
        st = StabilityStudy(kind, int(cm.group(1)), int(cm.group(2)), c.heading, c.file)
        months: list[int | None] = []
        for line in c.text.splitlines():
            if not line.strip().startswith("|"):
                continue
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            if cells and cells[0].lower() == "batch":
                months = [int(m.group(1)) if (m := MONTH_RE.match(x)) else None for x in cells[1:]]
                continue
            if not months or set(cells[0]) <= set("-: "):
                continue
            last = 0
            for mo, val in zip(months, cells[1:], strict=False):
                if mo is not None and re.match(r"^\d+(\.\d+)?$", val):
                    last = max(last, mo)
            st.batches[cells[0]] = last
        studies.append(st)
    return studies


def _ev(dossier: Dossier, file: str, quote: str) -> Evidence:
    cid = next((c.chunk_id for c in dossier.chunks if c.file == file and quote in c.text), None)
    return Evidence(chunk_id=cid, file=file, quote=quote, quote_verified=cid is not None)


def _matching(studies: list[StabilityStudy], temp: int, rh: int, kind: str = "long-term") -> list[StabilityStudy]:
    return [s for s in studies if s.kind == kind and s.temp == temp and s.rh == rh]


def rule_long_term_condition(dossier: Dossier, temp: int, rh: int, min_batches: int = 1) -> RuleResult:
    studies = _matching(parse_stability(dossier), temp, rh)
    if not studies:
        return RuleResult(Verdict.MISSING, f"No long-term stability study at {temp}°C/{rh}% RH found in 3.2.P.8.", [])
    s = max(studies, key=lambda x: len(x.batches))
    ev = [_ev(dossier, s.file, s.heading)]
    n = len(s.batches)
    if n >= min_batches:
        return RuleResult(Verdict.MET, f"{n} batches at {temp}°C/{rh}% RH ({', '.join(s.batches)}).", ev)
    return RuleResult(Verdict.PARTIAL, f"Only {n} batch(es) at {temp}°C/{rh}% RH; at least {min_batches} expected.", ev)


def rule_min_stability_batches(dossier: Dossier, min_batches: int, temp: int, rh: int) -> RuleResult:
    return rule_long_term_condition(dossier, temp, rh, min_batches)


def rule_min_long_term_months(dossier: Dossier, min_months: int, temp: int, rh: int) -> RuleResult:
    studies = _matching(parse_stability(dossier), temp, rh)
    if not studies or not any(s.batches for s in studies):
        return RuleResult(Verdict.MISSING, f"No long-term data at {temp}°C/{rh}% RH.", [])
    s = max(studies, key=lambda x: len(x.batches))
    worst = min(s.batches.values())
    ev = [_ev(dossier, s.file, s.heading)]
    if worst >= min_months:
        return RuleResult(Verdict.MET, f"All batches have ≥ {worst} months at {temp}°C/{rh}% RH (≥ {min_months} required).", ev)
    return RuleResult(Verdict.PARTIAL, f"Shortest batch has {worst} months; {min_months} required.", ev)


def rule_shelf_life_consistency(dossier: Dossier, region: str, zone: str) -> RuleResult:
    concl = None
    for c in dossier.chunks:
        if section_matches("3.2.P.8.1", c.section):
            for m in CONCLUSION_RE.finditer(c.text):
                if m.group(1).lower() == zone.lower():
                    concl = (int(m.group(2)), _ev(dossier, c.file, m.group(0)))
    label = None
    for c in dossier.chunks:
        if c.module == "m1" and c.region == region:
            m = LABEL_SHELF_RE.search(c.text)
            if m:
                label = (int(m.group(1)), _ev(dossier, c.file, m.group(0)))
                break
    if label is None:
        return RuleResult(Verdict.MISSING, f"No shelf life found in the {region.upper()} labeling.", [])
    if concl is None:
        return RuleResult(Verdict.MISSING, f"Label states {label[0]} months, but 3.2.P.8.1 has no shelf-life conclusion for Zone {zone}.", [label[1]])
    if label[0] == concl[0]:
        return RuleResult(Verdict.MET, f"Label and Zone {zone} stability conclusion both state {label[0]} months.", [label[1], concl[1]])
    return RuleResult(Verdict.PARTIAL,
                      f"Inconsistent: label states {label[0]} months but Zone {zone} stability supports {concl[0]} months.",
                      [label[1], concl[1]])


RULES = {
    "long_term_condition": rule_long_term_condition,
    "min_stability_batches": rule_min_stability_batches,
    "min_long_term_months": rule_min_long_term_months,
    "shelf_life_consistency": rule_shelf_life_consistency,
}


def run_rule(req: Requirement, dossier: Dossier) -> RuleResult:
    assert req.rule is not None
    fn = RULES.get(req.rule.name)
    if fn is None:
        raise ValueError(f"Unknown rule {req.rule.name} for {req.id}")
    return fn(dossier, **req.rule.params)
