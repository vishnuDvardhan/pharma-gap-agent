"""Regulatory pathway decision per market. Deterministic code, never the LLM.

The pathway produces `facts`; a requirement is active only if all of its `applies_if`
conditions equal the corresponding facts.
"""
from __future__ import annotations

from .schema import Pathway, Product, Requirement

ISO_ALIASES = {"GB": "UK"}


def decide_pathway(market: str, product: Product, market_cfg: dict) -> Pathway:
    approved_in = {ISO_ALIASES.get(a["market"], a["market"]) for a in product.approvals}
    applicant_country = ISO_ALIASES.get(product.applicant["country"], product.applicant["country"])
    site_countries = {s["country"] for s in product.manufacturing_sites}
    country_of_market = {"US": "US", "IN": "IN", "JP": "JP"}.get(market, market)

    facts = {
        "foreign_applicant": applicant_country != country_of_market,
        "imported": country_of_market not in site_countries,
        "new_active_substance": product.new_active_substance,
        "already_approved_here": market in approved_in,
    }
    rationale = [
        f"Applicant country {applicant_country} → foreign applicant: {facts['foreign_applicant']}.",
        f"Manufacturing sites in {sorted(site_countries)} → imported: {facts['imported']}.",
    ]

    if market == "US":
        name = "NDA 505(b)(1) (new molecular entity)" if product.new_active_substance else "NDA 505(b)(2) / ANDA"
        rationale.append(f"New active substance: {product.new_active_substance} → {name}.")
    elif market == "IN":
        ref = set(market_cfg.get("waiver_reference_markets", []))
        hits = sorted(approved_in & ref)
        if product.new_active_substance and hits:
            facts["pathway"] = "local_trial_waiver_eligible"
            name = "New drug import (CT-18), local clinical trial waiver may be requested (Rule 101)"
            rationale.append(f"Approved in {hits}, which are Rule 101 reference markets → waiver may be requested "
                             "(the CLA decides; not automatic).")
        else:
            facts["pathway"] = "local_trial_required"
            name = "New drug import (CT-18) with local clinical trial"
            rationale.append(f"No approval in a Rule 101 reference market ({sorted(ref)}) → local clinical trial required.")
    elif market == "JP":
        name = "New drug application (foreign data, ICH E5 bridging)"
        rationale.append("Pivotal data are foreign → ethnic-factor evaluation / bridging per ICH E5 is needed.")
    else:
        name = "Generic pathway"
    return Pathway(market=market, name=name, facts=facts, rationale=rationale)


def is_active(req: Requirement, pathway: Pathway | None) -> tuple[bool, str]:
    if not req.applies_if:
        return True, ""
    if pathway is None:
        return True, ""
    for key, want in req.applies_if.items():
        have = pathway.facts.get(key)
        if have != want:
            return False, f"Not applicable: requires {key}={want}, pathway has {key}={have}."
    return True, ""
