# Build log

What was done in each step of the rebuild, what differs from the implementation plan and why, and the checks that were run.

## Installed versions (checked 2026-10-30)

| Package | Main environment | `.venv-crewai` |
| --- | --- | --- |
| google-adk | 2.12.0 | not installed |
| google-genai | 2.31.0 | not installed |
| mcp | 2.3.0 | 1.30.0 |
| a2a-sdk | 1.3.0 | 1.3.0 |
| crewai | not installed | 1.15.30 |
| pydantic | 2.13.5 | 2.12.5 |
| fastapi | 0.141.1 | n/a |

Python 3.12.10 for both.

APIs named in the plan, confirmed to exist in these versions: `LlmAgent` (`output_schema`, `output_key`, `planner`, callbacks), `SequentialAgent`, `ParallelAgent`, `AgentTool`, `PlanReActPlanner`, `RemoteA2aAgent`, `McpToolset`, `StreamableHTTPConnectionParams`, `RunConfig.max_llm_calls`, `DatabaseSessionService`; CrewAI `Agent.mcps`, `Task.guardrail`, `Task.guardrail_max_retries`, `Task.output_pydantic`, `Crew.step_callback`, `Crew.task_callback`, `Crew.usage_metrics`.

## Differences from the plan

- **CrewAI MCP client:** `crewai-tools` is not installed. CrewAI 1.15.30 has native MCP support (`Agent(mcps=[MCPServerHTTP(url=..., headers=...)])` in `crewai.mcp`), so the crew uses that instead of `crewai_tools.MCPServerAdapter`.
- **Model:** `gemini-3.5-flash`, chosen on 2026-10-30. In a timing check that day, `gemini-3.8-flash` returned two 503 "high demand" errors and took 7 to 49 s per answer; `gemini-3.5-flash` answered in 2.2 s. `gemini-2.5-flash` is no longer available to new keys (404).
- **Package root:** the plan's `src/claimsleuth/` layout is used; the package is installed with `pip install -e .` (`pyproject.toml`), so `python -m claimsleuth.api.app` works from the project folder.

## Step 1: remove the duplicates

Deleted: the hand-written agent loop (`src/agents/`, `core/tool_registry.py`, `core/llm_client.py`, the regex extractor and Gemini planner), the fake CrewAI and unused ADK adapters (`src/frameworks/`), the legacy MCP server and unused client, the write-only blackboard, the callback-less monitor, the duplicate injection check (`core/security.py`; its patterns are a subset of `guardrails.py`), the deterministic ADK pipeline (`src/adk_app/`: rule-based `manager_decision`, `FunctionNode` agents, broadcast messaging, the duplicate `SequentialAgent` backend, the MCP transport switch), the dashboard's engine picker, `backups/`, legacy ledgers in `data/audit_logs/`, `RESTORE_BACKUP.md` and the beginner guide.

Also deleted, with the user's approval on 2026-10-30 ("delete when replaced"), because their only user was the deleted engine: the old verifier (`src/verifier/`), the old benchmark (`src/eval/`, `data/benchmark/`), `core/state.py`, `core/adjudication.py`, `rag/policy_evidence.py` (fixed RAG questions for the deterministic pipeline), the old FastAPI app (`src/ui/app.py`) and the tests that covered only deleted code.

Kept and moved into `src/claimsleuth/`: the RAG retriever and corpus (`rag/`), `tracing.py`, `audit.py`, the guardrail patterns (`callbacks/guardrails.py`) and the dashboard page (`web/index.html`, the style reference for step 7). Settings moved from `configs/settings.py` to `src/claimsleuth/config.py`, which reads `.env`.

Kept until their step replaces them: `src/mcp/remote_server.py` and `src/tools/` (step 3), `src/crew_service/` and its tests (step 5).

No model key: every `/api/*` route answers `503` with "No model key configured; investigations are disabled", and the page shows the same message (`test_api_refuses_without_model_key`).

**Check (2026-10-30):**

- `python -m claimsleuth.api.app` started with no import errors; `/healthz` answered `{"status":"ok", ...}`.
- `pytest`: 21 passed (RAG retrieval, audit chain, tracing, guardrail patterns, API start-up).
- `grep -rn "investigator\|crewai_adapter\|tool_registry" src` returned nothing.

## Step 2: data models and SQLite schema

`src/claimsleuth/models.py` holds every shared shape (`Quote`, `Fact`, `Facts`, `Coverage`, `Payout`, `FraudFinding`, `Fraud`, `GuardrailHit`, `FinalReport`) plus `ManagerSubmission`, `ClaimRecord`, `DocumentRecord`, `review_level()` and `to_markdown()`.

Differences from the plan:

- **`FinalReport` is written fresh, not extended.** The old `FinalReport` lived in `src/adk_app/report.py` and described the deleted deterministic pipeline (status codes, verifier gates). Its fields do not fit the new agents, so the plan's field list is used as is.
- **`ManagerSubmission`:** the Manager's `submit_report` takes only `decision`, `reasons`, `evidence` and an optional `payout_amount`. Code builds the `FinalReport` from the blackboard (payout, coverage, fraud, guardrail hits) and sets `review`, `pattern` and `prompt_style`, so the model cannot set them. A `payout_amount` that differs from `bb_payout` is overwritten and recorded as `bad_output` (step 6).

`src/claimsleuth/db.py`: plain `sqlite3`, WAL mode, the seven tables from the plan, created on start-up. The three demo scenarios are loaded into `claims` (`source='demo'`) and `documents`; `past_claims` is seeded from `data/past_claims.json`. That file includes a prior claim on serial `REF-HC-99400-B` by a different claimant, which is the serial on Scenario 3's diagnostic report, so the History analyst has a real record to find.

**Check (2026-10-30):** `pytest tests/test_models.py`: 9 passed (round-trip of every model, exact money, invalid values rejected, the review rule, database init and idempotence). `python -m claimsleuth.db --init` created the tables and loaded 3 claims, 6 documents and 4 past claims.

## Step 3: MCP server with five tools

`src/claimsleuth/mcp_server/server.py`: `parse_document`, `search_policy`, `check_coverage`, `compute_payout`, `claim_history`, each with typed arguments and a Pydantic return model. It is served over streamable HTTP at `/mcp` inside the FastAPI app, and every request needs `X-ClaimSleuth-Secret`. Without the header, with a wrong value, or when no secret is configured, the answer is `401`. `src/claimsleuth/mcp_server/pdf.py` reads PDFs with pypdf 6.21.4.

Differences from the plan:

- **`MCPServer`, not `FastMCP`:** in MCP SDK 2.3.0 the high-level server class is `mcp.server.mcpserver.MCPServer` (FastMCP's successor, same decorator API).
- **Errors:** tools raise `ToolError`, which MCP returns as an error result (`isError: true`) with the message. That is the protocol's own error channel, so a caller can never mistake an error for data, and the tool's output schema stays exact.
- **`search_policy` result shape:** `{passages: [{doc_id, section, score, lines: [{n, text}]}], warnings}` instead of `{section, text, line, score}`. Returning numbered lines lets the Policy agent quote the exact line of a clause. Policy wording has `doc_id` `policy:<POLICY-ID>`; an uploaded policy keeps its document id.
- **`claim_history` takes an optional `claim_id`** to exclude the claim under investigation and to measure `last_claim_days_ago` from its incident date.
- **The MCP SDK's localhost-only host check is off** (`enable_dns_rebinding_protection=False`), because it would reject the cloud host name. The shared secret is checked before any request reaches the MCP app.
- **Stateless HTTP:** `stateless_http=True`, so no MCP session state lives in the server.

Deleted: `src/mcp/remote_server.py` (replaced). `src/tools/` stays until step 5, because the old CrewAI service still imports it.

**Check (2026-10-30):** `python eval/mcp_smoke.py --url http://127.0.0.1:8090/mcp` (separate client process, `mcp.client.streamable_http`, against the running app on port 8090, because port 8000 was used by another program on this machine) printed PASS: exactly five tools listed, each called successfully with demo data, a scanned PDF returned "no text layer: scanned document, OCR not supported", and a request without the secret got `401`. `pytest`: 49 passed, including `tests/test_mcp_tools.py` (normal, edge and error cases for each tool and the `401` checks).

## Step 4: ADK agents and the three patterns

Four `LlmAgent`s (`src/claimsleuth/agents/`: `evidence.py`, `policy.py`, `settlement.py`, `manager.py`), instructions loaded from `prompts/direct/` and `prompts/react/` (the ReAct style also sets `PlanReActPlanner()`). Each specialist gets only its own MCP tools (`McpToolset` + `StreamableHTTPConnectionParams`, `tool_filter`, secret header, `header_provider` adding the W3C `traceparent`) and writes the blackboard through one recording tool (`record_facts`, `record_coverage`, `record_payout`). The Manager ends with `submit_report`. Specialists use `include_contents="none"`, so results reach them only through the blackboard placeholders (`{bb_facts?}` and so on). `patterns.py` builds fresh agents for `sequential`, `parallel` (`ParallelAgent` of Policy and the crew) and `hierarchical` (Manager with `AgentTool`s). `runner.py` runs a claim with `RunConfig(max_llm_calls=30)` and a 15-minute timeout and saves the run; a run without a submitted report is `failed`.

Differences from the plan:

- **`PolicyTerms` added to `Coverage`** (warranty months, deductible, limit, depreciation, each quoted). The Settlement agent needs these numbers for `compute_payout`, and the Policy agent has already found them, so they travel on the blackboard instead of a second policy search.
- **`AgentTool(RemoteA2aAgent)` works in ADK 2.12.0**, so the `transfer_to_agent` fallback was not needed.
- **`record_facts(facts, missing)` takes two flat arguments.** With a single nested `Facts` argument, the model kept passing a bare list (5 rejected calls in one live run).
- **Model-call timeout:** every ADK agent and the crew's LLM get `HttpOptions(timeout=120 s)`. Without it, a stalled call to an overloaded model hung a live run until the 15-minute run limit (2026-10-30).

## Step 5: CrewAI fraud crew over A2A

`src/claimsleuth/crew/crew.py`: Forensics analyst (`parse_document`) and History analyst (`claim_history`) through CrewAI's native MCP client, two tasks in `Process.sequential`, the second with `output_pydantic=Fraud`. Each task's `guardrail` rejects output that is not valid `Fraud` JSON, whose quotes are not at their line, or produced without the agent calling its tool (every document read; `claim_history` called), with `guardrail_max_retries=2`. Tool use is recorded from CrewAI's `ToolUsageFinishedEvent`; tokens per agent come from each agent's own `LLM` object. `src/claimsleuth/crew/a2a_server.py` refuses requests without the secret (`401`, agent card and `/healthz` excepted). `src/claimsleuth/agents/fraud_remote.py` is the ADK `RemoteA2aAgent`; its after-agent callback writes `bb_fraud` or records `bad_output`, which forces a full review.

Differences from the plan:

- **CrewAI's native Gemini provider** (`crewai[google-genai]==1.15.30`, which adds google-genai 1.65 to `.venv-crewai`). The OpenAI-compatible endpoint returned HTTP 400 on tool calls ("Function call is missing a thought_signature") with Gemini 3 models.
- **The tool-use check in the guardrail was added after a live run** in which both analysts answered from the facts in the prompt without calling a tool (0 tool calls, 5.6 s).

**Live check, Scenario 3, hierarchical, direct prompts, `gemini-3.5-flash-lite` (2026-10-30), run `6e93f40cc2ce`:** completed in 113 s. The Manager called Evidence, then Policy, then the crew, and skipped Settlement. `bb_fraud`: `flagged: true` with two high findings, "serial_number_mismatch" (invoice line 9 `REF-HC-77100-A` against repair estimate line 9 `REF-HC-99400-B`) and "serial_number_reuse" (`REF-HC-99400-B` claimed by Karen Whitfield 189 days earlier). Report: `request_evidence`, `full_review`, no payout. Trace `2d87fc5c195ff9b41bb3ef4e114b798b` (108 spans) contains `mcp.claim_history <- crewai.crew <- a2a ... send_message <- invoke_agent fraud_crew <- execute_tool fraud_crew <- invoke_agent claim_manager <- claimsleuth.investigation`, so one trace crosses ADK, A2A, CrewAI and MCP.

Earlier live runs: Scenario 1, sequential, `gemini-3.5-flash-lite`: approve, $251.00, `full_review`, 76 s. Two Scenario 3 attempts on `gemini-3.8-flash` failed: one with a TLS error during a network drop, one stopped by the 15-minute limit while the crew's model call hung (no timeout then; fixed above).

**Not yet done for step 4:** the 9-run check (3 demos x 3 patterns). The Gemini free tier allows about 20 requests per model per day; `gemini-3.5-flash` ran out on 2026-10-30.

## Step 6: callbacks for guardrails, monitoring, heat maps and tracing

`src/claimsleuth/callbacks/hooks.py` holds the callbacks, attached to every agent by `attach_callbacks()` in `patterns.build()` (`tests/test_patterns.py` checks every agent in every pattern). `before_model` screens the model input for injection and abuse (the patterns kept from the first version), records each hit once with where it came from (customer statement or document id), redacts abuse, wraps the statement in `<customer_statement>` tags and adds a security notice when an injection was found. `before_tool` refuses tools not on the agent's list and documents or claim ids of another claim. `after_tool` validates MCP results against the tool's return model (`bad_tool_result`). `on_tool_error` turns exceptions into errors the model sees. `after_agent` checks every Evidence fact quote (bad ones dropped, `bad_quote`) and the Policy clause and term quotes. `submit_report` checks the Manager's quotes and replaces a stated payout that differs from `bb_payout` (`bad_output`).

`src/claimsleuth/callbacks/metrics.py` collects time, tokens, model calls, tool calls, errors and guardrail hits per agent; the runner writes them to `agent_metrics` and the totals to `runs`. The crew's agents appear as `forensics` and `history` with their own numbers. `src/claimsleuth/charts.py` draws the per-run heat map.

Tracing: MCP tool calls run in an `mcp.<tool>` span whose parent is the caller's `traceparent` header (set by the `/mcp` middleware). The crew service has OpenTelemetry's SDK disabled, so it returns span records for the crew and its two tasks in the A2A reply, and the ADK callback writes them into the same trace file. The crew's MCP calls carry the crew span as their parent.

**Check so far:** `tests/test_guardrails.py` (injection and abuse patterns, statement and document attribution, quote check drops an invented quote, a malformed tool result becomes `bad_tool_result`, tool allowlists, other-claim refusals, payout replacement, approval without a payout refused). The injector-persona live check is part of step 8.

## Step 7: API, uploads, storage, login and dashboard

`src/claimsleuth/api/app.py`: every `/api/*` route needs `Authorization: Bearer $ADJUSTER_TOKEN` (`401` otherwise; `503` if no token is configured, so the API never runs open). `/healthz`, `/mcp` (its own secret) and the page shell are the exceptions; the page holds no data until the token is accepted. Routes: `GET/POST /api/claims`, `GET /api/claims/{id}`, `GET /api/documents/{doc_id}`, `POST /api/claims/{id}/investigate`, `GET /api/runs/{run_id}/events` (server-sent events), `GET /api/claims/{id}/report`, `GET /api/runs/{run_id}/heatmap.png`, `GET /api/runs/{run_id}/trace`, `POST /api/claims/{id}/review`, `GET /api/audit/{id}/verify`, `GET /api/evaluation`. Uploads: up to 10 files of 5 MB, PDF/TXT/MD checked by content (`%PDF-` header; UTF-8 text without NUL bytes) as well as extension, saved under `DATA_DIR/uploads/<claim_id>/`; an uploaded policy document is searched for that claim only. `src/claimsleuth/api/jobs.py` runs investigations in the background, at most 2 at a time and one per claim (`409`). The dashboard (`src/claimsleuth/web/index.html`, same palette and fonts as before) has the login, Claims (demos and uploads), New claim, Claim (live timeline from the event stream read with `fetch`, report, quotes that open the document at their line, fraud findings, guardrail hits, heat map, trace, review) and Evaluation pages.

Differences from the plan:

- **`runs.trace_id` column added**, so a run's trace can be fetched (`/api/runs/{id}/trace`); older databases get it through `ALTER TABLE` on start-up.
- **The demo and upload pages are one "Claims" page.** The pattern and prompt-style pickers are on the claim page, so the same controls work for demos and uploads.
- **Model calls are counted when a response arrives** (`after_model`), not when a call is attempted, so the metric equals the calls made (the call-cap test showed 31 counted for 30 made).
- **The Manager's seconds include its specialists' time** in the hierarchical pattern, because it waits for each of them.

Bug found by the tests: the `investigate` route was synchronous, so `asyncio.create_task` failed in FastAPI's worker thread and the error was reported as `409`. It is now async, and "already running" has its own exception.

**Check (2026-10-30, `gemini-3.5-flash-lite`):**

- Through the API: an uploaded claim with two text PDFs (`UPL-20261030-BC7E6C`, a Samsung Galaxy S24 with a $310.00 repair under POL-TECH-CARE-01) ran in the background in 69 s (run `cfdad4e7b00d`): approve, $281.00 from `compute_payout`, `full_review`. A second run on the same claim during the first got `409`. A review without a note got `422`; with a note, `200`. After stopping and restarting the app, the report, the review and its note were still there, and the audit chain verified (5 entries).
- In the dashboard, driven in headless Microsoft Edge with Playwright: a wrong token was refused; the same two PDFs uploaded through the New claim form (`UPL-20261030-7730B1`) and investigated with the Investigate button (run `fd1de031e1da`, 92 s, 55 timeline lines): approve, full review; approve without a note was refused, then accepted with one; a quote opened its document at line 4; the trace showed 121 spans. At 390 px wide the page first overflowed sideways; after a CSS fix the Claims, New claim and Claim pages measured exactly 390 px.
- `tests/test_api.py` (token, upload limits and file-type checks, restart persistence, review rules, `409`) and `tests/test_runner.py` (with ADK's real `Runner` and a fake model, the run stops at the 30-call cap with status `failed`, no report, 30 calls recorded). `pytest`: 78 passed.
