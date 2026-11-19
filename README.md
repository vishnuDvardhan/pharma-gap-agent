# Pharma Regulatory Gap Agent

**Question it answers:** *"Our drug is approved in the EU. What is missing from the dossier before we can file in the US, India and Japan?"*

The tool checks a CTD dossier against a library of regulatory requirements for each target market. It returns a gap matrix (requirement × market) where every verdict carries a verbatim dossier quote and the regulation it comes from. On top of that:

- a **vector database** (built-in store or **ChromaDB**) holds the dossier chunks and the requirement library, with incremental indexing and metadata filters (dossier, CTD module, region, section);
- a **multi-agent crew** answers questions in plain language: a supervisor plans, a regulatory-intelligence agent works out the pathway, **market specialists run in parallel** to turn each gap into an owned remediation action, a synthesizer writes the answer, and a compliance reviewer checks it before it is returned. The agents only use the checker's results; they never decide verdicts.

> ⚠ Prototype on **synthetic data** (a fictional drug, "Zelvorin"). Requirement texts are my own paraphrases, marked `verified: false`. Not for regulatory use.

![report](docs/report_screenshot.png)

## Key idea: one ICH core + a thin overlay per market

CTD Modules 2–5 are largely harmonised through ICH. Most launch gaps sit in a few places: regional Module 1 (forms, labels, local agents), climatic-zone stability, local clinical / bridging data and local pharmacopoeia. So:

- `data/requirements/ich_core.yaml`: checked **once**, inherited by every market.
- `us_fda.yaml`, `in_cdsco.yaml`, `jp_pmda.yaml`: market overlays. An overlay can **override** a core item. Example: India replaces the 25°C/60% RH long-term condition with Zone IVb, 30°C/75% RH.
- **Diff mode** checks only the delta. On the eval set it needs 56 judge calls instead of 92 (**−39%**) and gives identical verdicts (enforced by a test).

## Architecture

```mermaid
flowchart LR
  Q[User question] --> A[Single agent, or<br/>multi-agent crew below]
  A -->|tool| P[Pathway rules<br/>e.g. India Rule 101 waiver]
  A -->|tool| G[Gap-check executor<br/>deterministic loop over requirements]
  A -->|tool| S[search_requirements / search_dossier<br/>BM25 + vector DB, RRF]
  S --> VDB[(Vector DB<br/>local or ChromaDB)]
  G --> R[Retrieval: CTD section + region filter<br/>then hybrid BM25 + vector DB, RRF]
  R --> VDB
  R --> J{check_type}
  J -->|rule| RU[Code: stability tables,<br/>shelf-life consistency]
  J -->|llm| L[Judge: Claude / OpenAI / Bedrock / Ollama<br/>or offline heuristic]
  J -->|expert| E[Always 'Needs expert review'<br/>with evidence attached]
  L --> V[Guardrails: verbatim quote check,<br/>injection flag, JSON schema]
  RU & V & E --> DB[(SQLite audit store<br/>append-only, hashed)]
  DB --> H[HTML report / REST API]
```

### Multi-agent crew (`multiagent.py`)

```mermaid
flowchart TB
  Q[Question] --> SUP[Supervisor<br/>plan: markets + focus]
  SUP --> RI[Regulatory-intelligence agent<br/>tools: product profile, pathway, requirement search]
  SUP --> EX[Gap-check executor<br/>deterministic code, not an LLM]
  EX --> US[US specialist] & IN[IN specialist] & JP[JP specialist]
  RI -.pathway notes.-> US & IN & JP
  US & IN & JP -->|validated briefs| SY[Synthesizer]
  SY --> RV{Compliance reviewer<br/>code checks + LLM critique}
  RV -->|rejected: issues| SY
  RV -->|approved| ANS[Answer + action plan + trace]
```

| Agent | Tools it may call (enforced in code) | Output |
|---|---|---|
| Supervisor | none | plan (markets, focus); falls back to rules if the LLM plan is invalid |
| Regulatory intelligence | `get_product_profile`, `decide_pathway`, `search_requirements` | pathway brief per market |
| Gap-check executor | (code) `run_gap_check` | verdicts. **The only source of verdicts.** |
| Market specialist ×N, in parallel | `get_finding`, `search_dossier`, `search_requirements` | JSON plan: owner, action and dossier location for each gap |
| Synthesizer | none | final answer from the validated briefs |
| Compliance reviewer | none | approve, or reject with issues (max `multi_agent.max_revisions` loops) |

How the crew is kept honest:

- **Least privilege.** Each role gets its own tool allowlist. It is applied both to the tool specs sent to the model and inside `ToolBox.call`, so a specialist that asks for `run_gap_check` gets an error (tested).
- **LLM proposes, code validates.** Specialist output is parsed and checked. Requirement IDs that are not gaps get dropped, gaps it skipped get filled in with a rule-based action, and verdict, severity and source are always copied from the executor. Every repair is logged as a flag.
- **The reviewer is code first.** It checks that every ID in the answer came from a tool, every gap is covered, and the disclaimer is present; the LLM critique comes on top. If a draft still fails after the revision budget, the crew returns the rule-rendered, fully grounded answer and says so (`review_failed:used_rule_based_answer`).
- **Parallel specialists.** One thread per market. With a simulated 1 s LLM latency per specialist, a 3-market question takes 1.19 s instead of 3.17 s (2.7×).
- **Cost routing.** `multi_agent.models` can give each role its own model, e.g. a smaller model for the specialists. Every agent reports its own LLM calls, tokens and time.
- **Offline path.** Every agent has a deterministic path, so the whole crew runs in CI without an API key and produces the same trace shape.

### Vector database (`vectorstore.py`)

- **Two backends, one interface:** `LocalVectorStore` (numpy + JSON on disk, exact cosine; the default, no extra dependency) and `ChromaVectorStore` (ChromaDB persistent client, HNSW, cosine). Both run the same test contract.
- **Collections:** `dossier_chunks` holds every dossier, with the dossier ID as metadata (multi-tenant filtering). `requirements` holds the regulation library.
- **Incremental indexing:** each record stores a content hash, so re-indexing re-embeds only changed text, and `sync` deletes chunks whose source disappeared. Re-indexing an unchanged dossier takes about 5 ms and re-embeds nothing.
- **Embedder pinning:** a collection records the embedder that built it. Opening it with a different one rebuilds it instead of mixing incompatible vectors.
- **Metadata filters** (`$in`, `$nin`, `$eq`, `$ne`, `$and`) are the same for both backends. Retrieval passes the section and region candidates as a filter, so **EU Module 1 can never be evidence for a US Module 1 requirement**, with the vector DB too (tested).
- **Stateless embedder:** the default `hash` embedder (hashed char 3–5-grams, 1024 dims) gives the same vector for the same text in every process, which a persisted index needs. A re-fitted TF-IDF cannot do that. On the retrieval eval it matches TF-IDF exactly. `st:BAAI/bge-small-en-v1.5` switches to neural embeddings.

Design rules I followed:

1. **The LLM never decides what to check.** Requirements are the outer loop, so nothing gets skipped. The agent plans and explains, but verdicts come only from the executor.
2. **Arithmetic and lookups go in code, not the LLM.** Batch counts, months of data, temperature/RH and label-vs-stability shelf life are all handled by `rules.py`.
3. **Every Met/Partial needs a verbatim quote.** If the quote isn't found in the cited chunk, the verdict is downgraded to *Needs expert review* (`guardrails.verify_quote`).
4. **Dossiers are untrusted input.** The global dossier contains a planted prompt-injection string. It gets flagged, and it cannot change verdicts.
5. **Module 1 is regional.** An EU application form can never satisfy the US Form 356h check (tested).
6. **Data classification gate.** Dossiers not marked `public-synthetic`/`non-sensitive` cannot be sent to external APIs. They must use Ollama or the offline checker (`llm.make_llm`).
7. **Audit trail.** Each run stores model, config hash, corpus hash, tokens and timing. SQLite triggers block UPDATE/DELETE, and a content hash detects tampering.

## Mapping to the DSAI Early Career – AI Engineering JD

| JD item | Where in this repo |
|---|---|
| AI agents, tool/function calling, multi-step workflows, state, guardrails | `agent.py`: 6 tools, a reusable tool loop for Claude/OpenAI/Bedrock, step limit, argument validation, per-conversation run memory, grounding check on the final answer |
| Multi-agent systems: orchestration, delegation, parallelism, agent-to-agent review | `multiagent.py`: supervisor, regulatory-intelligence agent, parallel market specialists, synthesizer, reviewer with a bounded revision loop; per-role tool allowlists, per-role model routing, code validation of agent output, per-agent token/latency accounting |
| Vector databases, embeddings, semantic search | `vectorstore.py`: built-in store + ChromaDB behind one interface, incremental upsert/sync, metadata filters, embedder pinning; `gapagent index/search`, `/vectors/*` endpoints |
| RAG: grounding, retrieval quality, source/citation handling | `retrieval.py` (BM25 + vector DB, RRF hybrid, metadata filter), `judge.py`, `guardrails.verify_quote` |
| Prompts, retrieval patterns, systematic improvement | `JUDGE_SYSTEM` prompt, retrieval ablation (`eval/eval_retrieval.py`), keyword → heuristic → LLM judge ladder |
| Evaluation: test cases, error analysis, regression testing, safety testing | `eval/eval_gaps.py` (gap recall/precision, confusion matrix, list of misses), CI gate on gap recall, injection and hallucinated-quote tests |
| Claude models, OpenAI models, AWS Bedrock | `llm.py`: `AnthropicLLM`, `OpenAILLM`, `BedrockLLM` (Converse API), `OllamaLLM`; one neutral message format |
| Python, REST APIs, SQL, Git, testing, CI/CD, observability | FastAPI (`api.py`), SQLite + SQL aggregates (`audit.py`), 56 pytest tests, GitHub Actions, JSON request logs, `/metrics`, token/latency stats per run |
| Package and deploy | `Dockerfile` (non-root, healthcheck), `docker-compose.yml` (+ optional local Ollama), CI builds and smoke-tests the image |
| Responsible AI, privacy, security, governance | classification gate, human sign-off by design (`expert` checks), immutable audit log, "unverified" banner, untrusted-input handling |
| Copilot Studio / Power Automate | The REST API exposes OpenAPI at `/openapi.json`, which can be imported as a custom connector (not built here) |

## Quickstart

```bash
pip install -e ".[anthropic,openai,bedrock,vectordb,dev]"   # vectordb = ChromaDB (optional)
pytest -q                                            # 56 tests, no network needed
gapagent check zelvorin_global --markets US,IN,JP    # gap matrix + HTML report in reports/
gapagent ask "What blocks a launch in India and Japan?" --trace           # single agent
gapagent ask --multi "What blocks a launch in India and Japan?" --trace   # multi-agent crew
gapagent index                                       # build / refresh the vector DB (incremental)
gapagent search "translation into Japanese" --dossier zelvorin_global --module m1
gapagent serve                                       # http://localhost:8000 (UI) and /docs (API)
```

Switch the vector DB to ChromaDB with `GAPAGENT_VECTOR_STORE=chroma` (or `retrieval.vector_store: chroma` in `config.yaml`).
REST: `POST /agent/multi {"question": "..."}`, `POST /vectors/index`, `GET /vectors/search?q=...&dossier_id=...&module=m1&region=us`.

With a real model (the dossiers are synthetic, so external APIs are allowed):

```bash
export ANTHROPIC_API_KEY=...
gapagent --backend anthropic --model claude-sonnet-5-5 check zelvorin_global
python eval/eval_gaps.py --backends anthropic --model claude-sonnet-5-5
# also: --backend openai --model gpt-4.1-mini | --backend bedrock --model <model id> | --backend ollama --model qwen2.5:7b-instruct
```

Docker: `docker compose up --build`, then `POST /gap-checks {"dossier_id":"zelvorin_global","markets":["US","IN","JP"]}`.

## Results (offline judges; run the LLM rows yourself)

Test set: 2 dossiers × 40 requirement rows = 80 gold verdicts, 32 gold gaps, 9 deliberately tricky cases (documents only promised, 41 of 46 investigators covered, an accreditation pending, evidence inside a table, a label/stability mismatch, prompt injection).

| judge | mode | retrieval filter | gap recall | gap precision | exact acc | quotes verified | judge calls |
|---|---|---|---|---|---|---|---|
| keyword baseline | diff | on | 0.844 | 1.000 | 0.925 | 1.000 | 56 |
| heuristic | diff | on | **1.000** | 0.970 | 0.975 | 1.000 | 56 |
| heuristic | full | on | 1.000 | 0.971 | 0.985 | 1.000 | 92 |
| heuristic | diff | **off** | 0.969 | 0.969 | 0.925 | 1.000 | 56 |
| Claude / GPT / Bedrock | diff | on | *run `eval_gaps.py`* | | | | |

Gap-eval rows use the default retrieval: hybrid BM25 + local vector DB.

Retrieval (27 requirement→evidence pairs, `eval/eval_retrieval.py`): with the section+region filter, BM25 recall@1 rises from 0.926 to 1.000 and hybrid from 0.889 to 0.963. recall@3 is 1.0 for every setting. The persistent vector DBs give **exactly the same** recall and MRR as the in-memory index:

| method | dense index | filter | recall@1 | recall@3 | MRR |
|---|---|---|---|---|---|
| hybrid | in-memory TF-IDF | on | 0.963 | 1.000 | 0.981 |
| hybrid | vector DB (local) | on | 0.963 | 1.000 | 0.981 |
| hybrid | vector DB (ChromaDB) | on | 0.963 | 1.000 | 0.981 |
| hybrid | any of the three | off | 0.889 | 1.000 | 0.944 |

The keyword baseline marks "Debarment certification: to be provided" as **Met**. This is the failure mode that the heuristic and LLM judges exist to catch.

### Honest limitations

- **The numbers are optimistic.** I wrote the gold labels, the dossiers *and* the heuristic cues. Gap recall of 1.0 on 80 items written by one author says little about real dossiers. The corpus is also tiny, so retrieval is close to saturated.
- **The requirement library is unverified.** About 40 paraphrased requirements, each pointing to a source clause. Before relying on any of them, check each against the official text (NDCT Rules 2019, 21 CFR 314, PMD Act, ICH guidelines) and flip `verified: true`.
- **Not tested against live APIs here.** The LLM paths are covered with mocked SDK clients and a scripted fake LLM. The first real run may surface prompt or format issues.
- **The heuristic judge has a known false positive.** "Initial U.S. Approval: [pending]" makes it call the PLR label Partial. I left this unfixed on purpose: tuning cues to my own test set would be overfitting.
- **The vector DB is not yet *needed* at this scale.** 76 chunks fit in memory. The vector DB is there for persistence, incremental re-indexing and multi-dossier filtering, and it matches the in-memory recall; it does not improve it here. The default `hash` embedder works on character n-grams, so it is lexical rather than semantic. Use `st:` embeddings for real paraphrase matching.
- **The multi-agent LLM path is tested with scripted fakes only.** The tests cover routing, tool refusal, invented and missed IDs, and the reject → revise loop. Real models will need prompt tuning. The crew also costs more calls than the single agent: 1 + 1–3 + N specialists × 1–6 + 1–2 + 1–2 for N markets.
- **Scope.** One small-molecule tablet, 3 markets, Modules 1/3 plus a few clinical checks. No eCTD XML parsing; PDF parsing is optional (PyMuPDF).

## Next steps (in order of value)

1. Run `eval_gaps.py` with Claude and one other model. Add the rows to the table and study the misses.
2. Have someone else write 3–5 extra dossiers with planted gaps, so the eval is not graded by its own author.
3. Verify the requirement library against official texts.
4. Add a reranker (cross-encoder) and a larger, messier dossier (scanned PDF) to make retrieval matter.
5. Run the multi-agent crew with Claude, compare its answers with the single agent's (coverage, groundedness, cost), and add a crew eval.

## Repo map

```
data/requirements/   ICH core + US/IN/JP overlays (YAML)      data/gold/  hand-written gold verdicts + evidence
data/dossiers/       zelvorin_ref (EU only), zelvorin_global   data/products/, data/markets.yaml
src/gapagent/        loader, retrieval, vectorstore, rules, pathway, judge, llm, guardrails, pipeline,
                     agent, multiagent, audit, report, api, cli
eval/                eval_gaps.py, eval_retrieval.py, results/  tests/  56 tests
```
