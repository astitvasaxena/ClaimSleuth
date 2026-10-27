# ClaimSleuth: Agentic AI Warranty Investigation System

ClaimSleuth is a robust, transparent, and auditable warranty investigation platform that bridges agentic AI reasoning with deterministic eligibility verification and cryptographic audit logging.

Built for academic presentation and real-world compliance, ClaimSleuth solves the dual dangers of automated claim processing: rigid rule fragility and reckless generative AI hallucinations.

---

## Architecture Highlights

1. **Agentic Investigation Loop**:
   - `PLAN → EXECUTE → OBSERVE → ASSESS → REPLAN or STOP`
   - Bounded execution (hard limits of 5 iterations and 8 model calls per investigation).
   - Uses **Google Gemini** structured output when configured; otherwise a deterministic, state-aware rule-based planner (the default; see Integration Status).
   - Genuine **REPLAN** capability: when extraction fails or evidence is missing, the agent alters its plan with an explicit `replan_reason`.
2. **Restricted Tool Registry**:
   - Enforces an allowlist of 6 deterministic tools with strict Pydantic input schemas.
   - Dynamic discovery: tool descriptions and parameter schemas are injected into model prompts.
   - Tool execution is validated before dispatch; unauthorized tools are rejected.
3. **Dual-Mode Evidence Extractor with Provenance**:
   - **Gemini Extraction**: Extracts facts using structured JSON generation with verbatim raw snippets.
   - **Offline Regex Fallback**: Deterministic heuristic fallback when Gemini is unavailable.
   - **Snippet Invariant**: Verifies that every extracted `raw_snippet` is a character-for-character substring of the untrusted document before granting `VERIFIED` status.
4. **Deterministic Dual-Gate Verifier**:
   - **Gate 1 (Substantiation)**: Every extracted fact requires a verbatim character-level quote with confidence $\ge 0.85$ and zero unresolved conflicts.
   - **Gate 2 (Deterministic Rules & Math)**: Zero LLM math. Warranty eligibility windows and settlement payouts ($Payout = \min(Cost - Deductible, Limit) \times (1 - Depreciation)$) are 100% pure Python arithmetic.
5. **Human-in-the-Loop (HITL) Gate**:
   - Consequential decisions (payouts $> \$150$, rejections, and fraud/serial escalations) require one-click human authorization.
   - Secured `/api/adjudicate/{claim_id}` endpoint with Bearer token authorization.
6. **Cryptographic Append-Only Audit Ledger**:
   - Every tool execution, state mutation, and verifier decision is hashed in a SHA-256 chain log (`.jsonl`).
   - Integrated `verify_chain()` method to mathematically detect tampering.

---

## Current Integration Status (verified 27 Oct 2026)

What actually runs today. `GET /api/runtime-status` reports the same thing from the live server.

| Component | Status |
|---|---|
| Deterministic tools (coverage window, exclusions, settlement, serial anomalies) | Working, tested |
| Investigation loop | Working in deterministic fallback mode (rule-based planner). Gemini planning is implemented but has not been verified against a live API key. |
| Gemini | Only active when `USE_LLM=true`, `GOOGLE_API_KEY` is set **and** `google-generativeai` is installed; otherwise deterministic fallback |
| Dual-gate verifier, human approval with state-transition guard, hash-chained audit ledger | Working, tested |
| MCP | **Remote server over stdio and streamable HTTP** (`src/mcp/remote_server.py`, 6 tools). The ADK pipeline uses it through ADK's `McpToolset` with `tools=mcp_stdio|mcp_http` (verified with a server-side call log). The original loop still calls the legacy server **in-process**. No authentication: localhost only. |
| Google ADK | **Executed** via `POST /api/adk/investigate/{scenario}` (real `Runner` + session state; `Workflow` and `SequentialAgent` backends), selectable in the dashboard; results go through the same human-adjudication rules. The dashboard's default engine is still the original loop. |
| CrewAI | **Real `crewai` 1.15.30** in a separate environment (`.venv-crewai`): a CrewAI `Flow` runs the forensic checks and an optional CrewAI `Crew` writes a narrative (tested with a fake LLM only). Used via `?forensics=crewai_a2a`. The old `/api/investigate-crew` endpoint is still a placeholder that does not use crewai. |
| A2A | **ADK `RemoteA2aAgent` → CrewAI service over HTTP** (a2a-sdk 1.3.0, Agent Card + JSON-RPC), verified with a service-side request log. ADK marks its A2A support experimental. No authentication: localhost only. The in-memory `A2AMessage` log in `blackboard.py` is not the protocol. |

---

## Quickstart & Demo Guide

### 1. Run Automated Unit & Regression Tests

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```
*(212 tests passing and 2 live Gemini tests skipped without a key, as of 27 Oct 2026. The CrewAI A2A tests need `.venv-crewai`, see below; without it they are skipped.)* The test package redirects the audit ledger to a temporary directory, so test runs (pytest or unittest) never write inside the repository.

### 2. Google ADK pipeline (runs alongside the original loop)

`POST /api/adk/investigate/{scenario_id}?backend=workflow|sequential` runs the claim through a real
`google.adk` `Runner` with an in-memory session. The session state is the Blackboard (`bb:*` keys,
documented in `src/adk_app/steps.py`). Seven agents run in order: Intake, Evidence, Anomaly, Warranty,
Settlement, Verifier and Report. Each agent calls the existing deterministic code; the verifier and the
shared status/payout rules in `src/core/adjudication.py` decide the outcome. Each run writes a Pydantic
`FinalReport` as JSON and Markdown to `data/runtime/reports/` (gitignored; override with
`CLAIMSLEUTH_REPORTS_DIR`).

- **No key (default):** deterministic planner, no network access (covered by an offline test).
- **Model mode:** an ADK `LlmAgent` (`InvestigatorLLM`) chooses which documents and fields to investigate
  through validated tools (`src/adk_app/tools.py`). It has no settlement tool; a payout figure it states is
  recorded as non-authoritative, and a mismatch with the deterministic settlement escalates the claim.
  Enabled with `USE_LLM=true` plus `GOOGLE_API_KEY`. Automated tests use a fake `BaseLlm` test double;
  the live Gemini test is separate: `CLAIMSLEUTH_LIVE_TESTS=1 python -m pytest tests/live -v -rs`.
- **Backends:** `workflow` (default) uses `google.adk.Workflow`. `sequential` uses `SequentialAgent`, which
  google-adk 2.12.0 marks *deprecated in favor of Workflow*; it is kept only as a fallback and for parity
  tests, and should be removed once the Workflow backend is accepted. Both backends run the same step
  functions, so removing it needs no logic changes. Note from the deprecation message: a Workflow cannot
  yet be used as an `LlmAgent` sub-agent (the reverse, an `LlmAgent` node inside a Workflow, is used here).
- **Tools:** `tools=local` (default) or `tools=mcp_stdio` / `tools=mcp_http`, which route the agents' tool calls
  to the MCP server through ADK's `McpToolset` (`src/adk_app/tool_gateway.py`). MCP results are proposals: the
  VerifierAgent always recomputes locally. A tampered settlement is escalated, a failing tool makes the claim
  `NEEDS_EVIDENCE`, and an unreachable server aborts the run with HTTP 503 and no result.
- **Human adjudication:** an ADK run is registered for `POST /api/adjudicate/{claim_id}` with the same validation
  as the original loop. After a decision a separate `<claim>_<run>_adjudicated.json/.md` is written; the original
  report is left unchanged. Only a human `APPROVED` decision produces `AUTHORIZED`.
- **Dashboard:** the engine selector next to *Initiate Investigation* chooses the original loop (default),
  ADK Workflow with local tools, or ADK Workflow with the MCP server.

### 3. MCP tool server over a real transport

`src/mcp/remote_server.py` runs as its own process and serves six deterministic tools over MCP (SDK 2.3.0):
`list_claim_documents`, `extract_claim_document_facts` (dossier-only file parser with snippet provenance),
`get_policy_terms` (policy lookup), `check_policy_coverage` (coverage window + exclusions),
`calculate_policy_settlement` (proposal only, never authorization) and `flag_serial_anomalies`.
Policy terms are looked up by `policy_id` on the server; callers cannot supply or override deductibles or limits.

```bash
python -m src.mcp.remote_server --transport stdio                       # client spawns it
python -m src.mcp.remote_server --transport http --port 8102            # endpoint http://127.0.0.1:8102/mcp
```

Client: `src/mcp/remote_client.py` (`RemoteMCPClient.stdio()`, `RemoteMCPClient.http(url)`, `call_remote_tool(...)`).
Tool errors arrive as MCP `isError` results and raise `MCPToolError`. `tests/test_mcp_remote.py` exercises every tool
over both transports against a separate server process. The investigation pipelines do **not** call this server yet;
they still use the legacy in-process `src/mcp/tool_server.py`. The HTTP server has no authentication and binds to
localhost by default.

### 4. CrewAI forensics service over A2A (separate environment)

CrewAI cannot share the main environment: `crewai` 1.15.30 requires `mcp<2`, `pydantic<2.13` and `openai<3`.
It runs as its own service in its own virtual environment and is reached only over the A2A protocol.

```bash
python -m venv .venv-crewai
.venv-crewai/Scripts/python -m pip install -r requirements-crewai-service.txt     # ~900 MB
.venv-crewai/Scripts/python -m src.crew_service.a2a_server --port 8101          # Agent Card: /.well-known/agent-card.json
```

Then call `POST /api/adk/investigate/{scenario}?forensics=crewai_a2a` (or pick *ADK + MCP + CrewAI forensics* in the
dashboard). ADK's `RemoteA2aAgent` sends the claim to the service, the CrewAI `ForensicsFlow` checks serial
consistency, purchase/incident chronology and exclusions, and returns a `ForensicsReport`
(`src/core/forensics_schema.py`). That schema has no payout or approval fields; extra fields are rejected.
The deterministic `ForensicsReviewAgent` escalates any CrewAI flag or any disagreement with ADK's own anomaly check,
and fails closed (`NEEDS_EVIDENCE`) if the service is down, times out, or returns an invalid report.
CrewAI telemetry is disabled in the service. Service URL: `CLAIMSLEUTH_CREWAI_A2A_URL` (default `http://127.0.0.1:8101`).
CrewAI-only unit tests: `.venv-crewai/Scripts/python -m unittest tests.crew_service.test_forensics_flow -v`.

### 5. Agent patterns and messaging architectures (comparison)

`POST /api/adk/investigate/{scenario}?pattern=sequential|parallel|hierarchical&messaging=blackboard|direct|broadcast`
(Workflow backend; code in `src/adk_app/patterns.py`):

- **sequential**: one fixed chain of agents.
- **parallel**: ADK `Workflow` fan-out after evidence gathering (Anomaly ‖ Warranty, plus the CrewAI A2A call when
  enabled), an ADK `JoinNode`, then a `MergeAgent`. Branches write only their own output; the merge combines them and
  fails closed if two branches disagree on a value.
- **hierarchical**: a deterministic `ClaimManager` delegates to worker agents, which return control to it (ADK
  routing with cycles). It chooses the next worker from the case state (for example, it escalates straight to the
  verifier once a conflict exists) and records every decision in the report.
- **messaging** controls what context each agent receives: the whole shared state (`blackboard`), only messages
  routed to it (`direct`, see `DIRECT_ROUTES`), or every message (`broadcast`).

All combinations use the same deterministic verifier and adjudication rules. Compare them with:

```bash
python -m src.eval.compare_patterns --repeats 3      # writes data/runtime/eval/<timestamp>/patterns.csv and .md
```

### 6. Benchmark evaluation, guardrails and tracing

**Benchmark (C-10).** `python -m src.eval.dataset` regenerates the seeded synthetic dataset in `data/benchmark/`
(144 claims = 12 case types x 4 claimant personas x 3 splits; labels in `data/benchmark/labels/`, outside the
dossiers). Labels come from how each case was constructed, not from running ClaimSleuth.

```bash
python -m src.eval.benchmark --splits dev,val                    # development splits
python -m src.eval.benchmark --splits test --confirm-heldout     # held-out split; logged to data/benchmark/heldout_runs.jsonl
```

Each run writes `runs.csv`, `agent_metrics.csv`, `summary.md`/`summary.json` and SVG boxplots and per-agent heat
maps to `data/runtime/eval/benchmark_<timestamp>/`. Token and cost figures are only reported when a live model
reported usage; deterministic runs say "not measured". Prompt variants (`baseline`, `cot`, `react`,
`evidence_first`) need `--model` and a `GOOGLE_API_KEY`.

**Guardrails (C-8).** `src/core/guardrails.py` validates the claim form, detects prompt injection in the claim
text and in every dossier document, blocks threats, and redacts insults/profanity. Blocked claims get status
`BLOCKED_BY_GUARDRAIL` and can only be rejected or sent back for information. In model mode, ADK
`before_model_callback` blocks injected requests (the model is not called) and redacts toxic words, and
`after_tool_callback` type-checks tool results. The checks are pattern/keyword based, not trained classifiers.

**Tracing and monitoring (C-5/C-9).** Each ADK investigation gets one W3C trace id. ADK spans are written to
`data/runtime/traces/<trace_id>.jsonl` (`GET /api/trace/{trace_id}`), every audit entry carries the trace id,
the MCP server logs it from the request `_meta`, and the CrewAI A2A service logs it from the `traceparent`
header. Reports include per-agent metrics (latency, tool calls, errors, tokens when measured) and a
structured error list.

### 7. Optional: Configure Gemini API Key
To enable real Gemini model-guided planning and extraction:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export USE_LLM=true
# Optional: export GEMINI_MODEL="gemini-2.0-flash"
```
*(If no API key is set or `USE_LLM=false`, ClaimSleuth automatically operates in deterministic fallback mode with zero degradation in benchmark pass rates).*

### 8. Launch the Interactive Web Dashboard
Run the FastAPI server:

```bash
python -m uvicorn src.ui.app:app --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000`**

Useful endpoints: `GET /health`, `GET /api/runtime-status`, `GET /api/audit/{claim_id}?limit=200`, `GET /api/audit/{claim_id}/verify` (recomputes the hash chain), `GET /api/audit-legacy` (read-only verification of the pre-fix ledgers).

Investigation responses include a `payout` object: `PROPOSED_PENDING_APPROVAL` (verifier-approved, not paid), `AUTHORIZED` (human-approved) or `NOT_PAYABLE`. For a conflicted or rejected claim any amount appears only as `diagnostic_calculation`, never as a payout.

Optional environment variables: `CLAIMSLEUTH_ADJUDICATION_TOKEN` (require a Bearer token for human decisions; the dashboard prompts for it), `CLAIMSLEUTH_CORS_ORIGINS` (comma-separated allowed origins), `CLAIMSLEUTH_AUDIT_LOG_DIR` (where audit ledgers are written; default `data/runtime/audit_logs/`, gitignored).

### Human decision rules
`POST /api/adjudicate/{claim_id}` accepts only `APPROVED`, `REJECTED` or `INFO_REQUESTED` (anything else → 422). Approval is allowed only for `VERIFIED_PENDING_APPROVAL` claims with a verifier-approved settlement; conflicted, expired or closed claims return 409, and the blocked attempt is written to the audit ledger.

---

## The 3 Benchmark Scenarios

| Scenario | Case Description | Technical Demonstration | Final Adjudication |
|---|---|---|---|
| **Scenario 1** | **Legitimate iPhone Repair** (`CLM-2024-001`) | Active 24-mo TechGuard warranty. $280 OLED repair. Quoted invoice & diagnostic report. | **VERIFIED PASS** $\rightarrow$ Net settlement: **$251.00**. 1-click Human Authorization. |
| **Scenario 2** | **Expired Laptop Warranty** (`CLM-2024-002`) | OmniLaptop power IC failure claimed 3 months after 12-month warranty expired. | **REJECTED DETERMINISTICALLY** $\rightarrow$ Deterministic date engine halts approval early; no LLM override allowed. |
| **Scenario 3** | **Conflicting Serial Number** (`CLM-2024-003`) | Refrigerator compressor claim. Purchase invoice has serial `REF-HC-77100-A` while repair bill has `REF-HC-99400-B`. | **CONFLICT ESCALATED** $\rightarrow$ Anomaly detector & generalized conflict detection halt loop, record conflict in ledger, escalate to Human. |

---

## Directory Structure

```
claim_sleuth/
├── README.md                      # Presentation & usage guide
├── configs/
│   ├── policies.json              # Warranty contracts (limits, deductibles, exclusions)
│   └── settings.py                # Loop bounds, Gemini config, auth tokens
├── data/
│   ├── sample_claims/             # 3 Benchmark scenarios with raw evidence
│   ├── benchmark/                 # Seeded synthetic benchmark: claims/, labels/, manifest.json, heldout_runs.jsonl
│   ├── audit_logs/                # LEGACY ledgers from 850d757 with INVALID chains (kept as evidence; see LEGACY_INVALID_README.md)
│   └── runtime/audit_logs/        # Ledgers written by the running app (gitignored)
├── src/
│   ├── adk_app/                   # Google ADK path: steps (agents), tools, pipeline backends, FinalReport
│   ├── crew_service/              # CrewAI ForensicsFlow + A2A server (runs in .venv-crewai)
│   ├── eval/                      # Benchmark dataset + harness, pattern comparison, SVG plots
│   ├── core/
│   │   ├── state.py               # Pydantic schemas (AgentAction, InvestigationStep, EvidenceItem)
│   │   ├── audit.py               # Cryptographic SHA-256 logger with verify_chain()
│   │   ├── tool_registry.py       # Restricted tool registry with Pydantic action schemas
│   │   └── llm_client.py          # Gemini structured client with retry and backoff
│   ├── tools/
│   │   ├── policy_rules.py        # Deterministic coverage windows & exclusions
│   │   ├── math_engine.py         # Pure arithmetic settlement calculator
│   │   ├── anomaly_detector.py    # Serial mismatch & fraud detector
│   │   └── evidence_store.py      # Sandboxed document retrieval
│   ├── agents/
│   │   ├── extractor.py           # Dual-mode extractor (Gemini + regex) with substring verification
│   │   └── investigator.py        # Lead Investigator with Gemini reasoning & replanning
│   ├── verifier/
│   │   └── independent_verifier.py# Dual-Gate rule & math verifier
│   └── ui/
│       ├── app.py                 # FastAPI demo backend with token auth
│       └── templates/index.html   # Glassmorphic responsive web dashboard with REPLAN display
└── tests/
    ├── __init__.py, conftest.py   # Redirect audit logs to a temp dir during tests (unittest + pytest)
    ├── test_claim_sleuth.py       # Core rules, verifier, registry, extractor, agent loop (31 tests)
    ├── test_milestone_50.py       # Blackboard, MCP tools, adapters, monitoring, security (19 tests)
    ├── test_wp1_regressions.py    # Regression tests for the 27 Oct 2026 defect fixes (24 tests)
    ├── test_payout_and_legacy_logs.py # Payout-view and legacy-ledger regression tests (10 tests)
    ├── test_adk_pipeline.py       # Google ADK pipeline acceptance tests (20 tests, offline)
    ├── test_mcp_remote.py         # MCP over stdio + streamable HTTP, separate server process (17 tests)
    ├── test_adk_mcp.py            # ADK pipeline through McpToolset; server failures; tampered results (15 tests)
    ├── test_adk_adjudication.py   # ADK results through human adjudication + dashboard compatibility (12 tests)
    ├── test_crewai_a2a.py         # ADK <-> CrewAI over A2A: protocol, pipeline, failures, conflicts (17 tests)
    ├── test_patterns.py           # Sequential / parallel / hierarchical x blackboard / direct / broadcast (16 tests)
    ├── test_guardrails.py         # Validation, injection, toxicity; blocked claims never payable (12 tests)
    ├── test_tracing.py            # Trace ids across ADK spans, audit and the MCP server; per-agent metrics (7 tests)
    ├── test_benchmark.py          # Dataset reproducibility, held-out discipline, plots, harness (11 tests)
    ├── crew_service/              # CrewAI unit tests, run inside .venv-crewai (8 tests)
    ├── fixtures/faulty_mcp_server.py # Fault-injection MCP server used by the tests
    └── live/                      # Live Gemini tests (skipped unless explicitly enabled)
```
