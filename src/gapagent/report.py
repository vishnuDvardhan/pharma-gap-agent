"""Self-contained HTML report (one file, no external assets)."""
from __future__ import annotations

from collections import Counter

from jinja2 import Environment, select_autoescape

from .pipeline import market_view
from .schema import GAP_VERDICTS, GapRun, Verdict

TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Gap analysis {{ run.dossier_id }} → {{ run.markets|join(', ') }}</title>
<style>
body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1200px;color:#1d2433;padding:0 16px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #dde1e8;padding-bottom:4px}
.muted{color:#5f6b7a} table{border-collapse:collapse;width:100%;margin:8px 0}
th,td{border:1px solid #dde1e8;padding:6px 8px;vertical-align:top;text-align:left} th{background:#f4f6f9}
.v{font-weight:600;white-space:nowrap;border-radius:4px;padding:1px 6px}
.Met{background:#e3f4e8;color:#16642f}.Partial{background:#fff3d6;color:#7a5200}
.Missing{background:#fde4e4;color:#9b1c1c}.Needs{background:#e6ecfb;color:#24459b}.Not{background:#eef0f3;color:#5f6b7a}
.cards{display:flex;gap:12px;flex-wrap:wrap}.card{border:1px solid #dde1e8;border-radius:8px;padding:10px 14px;min-width:200px}
.card b{font-size:20px} q{color:#33415c} .flag{font-size:12px;background:#fde4e4;color:#9b1c1c;border-radius:3px;padding:0 4px;margin-left:4px}
.banner{background:#fff3d6;border:1px solid #f0d58c;padding:8px 12px;border-radius:6px}
</style></head><body>
<h1>Regulatory gap analysis: {{ run.dossier_id }}</h1>
<div class="muted">Target markets: {{ run.markets|join(', ') }} · mode: {{ run.mode }} · run {{ run.run_id }} · {{ run.created_at }}</div>
<p class="banner">Prototype on synthetic data. Requirement texts are paraphrased and not verified against
official sources. Not for regulatory decisions. Every finding needs sign-off by a qualified reviewer.</p>

<h2>Summary</h2>
<div class="cards">
{% for m in run.markets %}{% set c = counts[m] %}
<div class="card"><div class="muted">{{ m }} · {{ run.pathways[m].name }}</div>
<b>{{ c.gaps }}</b> gaps ({{ c.critical }} critical) · {{ c.expert }} expert reviews · {{ c.met }} met</div>
{% endfor %}
</div>

<h2>Pathway decisions (rule-based)</h2>
<table><tr><th>Market</th><th>Pathway</th><th>Rationale</th></tr>
{% for m in run.markets %}<tr><td>{{ m }}</td><td>{{ run.pathways[m].name }}</td>
<td>{% for r in run.pathways[m].rationale %}{{ r }}<br>{% endfor %}</td></tr>{% endfor %}</table>

<h2>Gap matrix</h2>
<table><tr><th>Requirement</th><th>Severity</th>{% for m in run.markets %}<th>{{ m }}</th>{% endfor %}</tr>
{% for row in matrix %}<tr><td><b>{{ row.id }}</b><br><span class="muted">{{ row.title }}</span></td><td>{{ row.severity }}</td>
{% for m in run.markets %}<td>{% set v = row.cells.get(m) %}{% if v %}<span class="v {{ v.split()[0] }}">{{ v }}</span>{% endif %}</td>{% endfor %}</tr>{% endfor %}
</table>

{% for m in run.markets %}
<h2>{{ m }}: findings</h2>
<table><tr><th style="width:18%">Requirement</th><th style="width:11%">Verdict</th><th>Reasoning and evidence</th><th style="width:20%">Source</th></tr>
{% for f in views[m] %}<tr>
<td><b>{{ f.req_id }}</b>{% if f.inherited %} <span class="muted">(ICH core)</span>{% endif %}<br>{{ f.title }}</td>
<td><span class="v {{ f.verdict.value.split()[0] }}">{{ f.verdict.value }}</span>{% for fl in f.flags %}<span class="flag">{{ fl }}</span>{% endfor %}</td>
<td>{{ f.reasoning }}{% for e in f.evidence %}<br><q>{{ e.quote }}</q> <span class="muted">[{{ e.file }}{% if not e.quote_verified %}, UNVERIFIED{% endif %}]</span>{% endfor %}</td>
<td>{{ f.source.doc }}<br><span class="muted">{{ f.source.clause }}</span></td></tr>{% endfor %}
</table>{% endfor %}

<h2>Audit</h2>
<table>
<tr><th>Backend / model</th><td>{{ run.backend }} / {{ run.model }}</td></tr>
<tr><th>Config hash</th><td>{{ run.config_hash }}</td></tr>
<tr><th>Corpus hash (requirements + dossier)</th><td>{{ run.corpus_hash }}</td></tr>
<tr><th>LLM calls / tokens in / out</th><td>{{ run.stats.llm_calls }} / {{ run.stats.input_tokens }} / {{ run.stats.output_tokens }}</td></tr>
<tr><th>Requirements checked / inherited</th><td>{{ run.stats.requirements_checked }} / {{ run.stats.requirements_inherited }}</td></tr>
<tr><th>Duration</th><td>{{ run.stats.duration_ms }} ms</td></tr>
</table>
</body></html>"""


def build_matrix(run: GapRun) -> list[dict]:
    rows: dict[str, dict] = {}
    for m in run.markets:
        for f in market_view(run, m):
            row = rows.setdefault(f.req_id, {"id": f.req_id, "title": f.title, "severity": f.severity, "cells": {}})
            row["cells"][m] = f.verdict.value
        for f in run.findings:
            if f.market == m and f.overrides and f.overrides in rows:
                rows[f.overrides]["cells"][m] = f"Not applicable (replaced by {f.req_id})"
    return list(rows.values())


def market_counts(run: GapRun, market: str) -> dict:
    v = market_view(run, market)
    c = Counter(f.verdict for f in v)
    return {
        "gaps": sum(c[x] for x in GAP_VERDICTS),
        "critical": sum(1 for f in v if f.verdict in GAP_VERDICTS and f.severity == "critical"),
        "expert": c[Verdict.EXPERT],
        "met": c[Verdict.MET],
    }


def render_html(run: GapRun) -> str:
    env = Environment(autoescape=select_autoescape(default=True))
    return env.from_string(TEMPLATE).render(
        run=run, matrix=build_matrix(run),
        views={m: market_view(run, m) for m in run.markets},
        counts={m: market_counts(run, m) for m in run.markets},
    )
