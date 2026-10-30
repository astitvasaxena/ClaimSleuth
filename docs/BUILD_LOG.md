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

## Pinned model (2026-10-30)

`gemini-3.5-flash-lite` is the one model for every agent, the crew, the remaining live checks and the whole evaluation. It is a constant in `src/claimsleuth/config.py`, deliberately not read from the environment (the `GEMINI_MODEL` lines were removed from `.env` and `.env.example`), so no run can use another model. `eval/run_eval.py` also refuses to run if the server or any row already in `runs.csv` reports another model. Reasons: it completed every live run on 2026-10-30 (Scenario 1 sequential, Scenario 3 hierarchical, two uploaded PDF claims); it answered a probe in 0.8 s while `gemini-3.8-flash` returned 503s and hung one run, `gemini-3.7-flash` timed out (504) and `gemini-3.5-flash` hit its free daily quota.

Price, from Google's pricing page (https://ai.google.dev/gemini-api/docs/pricing), read 2026-10-30, paid tier, standard: input $0.30 per million tokens (text, image, video, audio), output $2.50 per million tokens including thinking tokens. Stored in `config.PRICES` with the date and link; every run's `cost_usd` is computed from it. The free tier is "free of charge" for this model. Google's rate-limit page (https://ai.google.dev/gemini-api/docs/rate-limits) does not publish per-model numbers; it says limits are per project and shown in Google AI Studio.

## Step 8: evaluation claim set and runner (code only; no runs yet)

- `eval/make_claim_set.py` wrote the set once; the output is committed: `eval/claims/E1`..`E8` (claim.json and documents; E2, E3, E5 and E8 contain PDFs made with reportlab, with fixed bytes), `eval/claims/labels.json` (accepted decisions, whether fraud must be flagged, the exact payout for E1 and E5), `eval/claims/past_claims.json` (E7's earlier claim on the same serial), and `eval/personas/<claim>_<persona>.txt` (40 statements: honest, abusive, vague, fraudster, injector). Expected payouts follow the policy formula: E1 $180.00 - $29.00 = $151.00; E5 $1150.00 - $75.00 = $1075.00, capped at $800.00, minus 5% depreciation = $760.00.
- `db.init()` loads the set as 40 claims with `source='eval'` (`EVAL-<claim>-<persona>`; same documents, the persona's statement) and adds the E7 history row. The claim-history seeds are now matched row by row, because the first version skipped the main seed once the E7 row existed.
- `claim_history` ignores other `source='eval'` claims: the five persona variants of a packet share a claimant and serial and would otherwise count as each other's earlier claims.
- `eval/check_eval_set.py` passes: 8 packets, known policies, readable documents, 4 packets with PDFs, a label for each, 40 statements, the abusive and injector statements trip the guardrail patterns and the honest ones do not, E7's serial is in the history seed.
- `eval/run_eval.py`: experiment A (3 patterns x 2 prompt styles x 8 claims, honest) and B (the best configuration of A, highest accuracy then lowest mean cost, x 40 variants); calls the API like a client (`POST .../investigate`, the event stream, `GET /api/runs/{id}`); resumable; on a 429 it waits (60 to 900 s, logged as `rate_limit_wait_s`, not agent time) and on a per-day quota error it stops so it can be resumed. `GET /api/runs/{run_id}` was added for it.
- `eval/charts.py`: boxplots of seconds and cost per configuration and per persona, the agents x configurations heat map, the per-run heat maps (downloaded from the API) and `summary.md` (model, prices with date and link, accuracy, guardrail catch rate for the abusive and injector personas).
- `tests/test_eval.py` covers the plan, the scoring, the choice of configuration and the chart code; its chart test uses a made-up CSV in a temporary folder, never `eval/results/`.

## Step 9, shared part: containers

`deploy/Dockerfile.app` (python:3.12-slim, CPU-only PyTorch, the embedding model downloaded at build time and loaded offline at run time, Litestream 0.3.13, a non-root user, `HEALTHCHECK` on `/healthz`), `deploy/entrypoint-app.sh` (Litestream restore and replicate only when `LITESTREAM_BUCKET` is set), `deploy/litestream.yml`, `deploy/Dockerfile.crew`, `deploy/docker-compose.yml` (app published on `127.0.0.1:${APP_HOST_PORT:-8000}`, the crew only inside the Compose network, data on a named volume, secrets from `.env`), `.dockerignore`, `.gitattributes` (shell scripts keep LF line endings). Both images use an editable install (`pip install -e .`), because the code finds `configs/` and `data/` relative to `src/`. `pypdf`, `matplotlib` and `reportlab` were missing from `requirements.txt` and were added.

## Step 10, the parts that do not need results

- `tests/test_crew_a2a.py`: the request text, a valid crew reply becoming `bb_fraud` (and its spans stored), unusable replies becoming `bad_output` and forcing a full review.
- `tests/crew/test_crew_service.py` (run in `.venv-crewai` with `unittest`, which has no pytest): the secret on the A2A endpoint (`401`; agent card and `/healthz` open), request parsing, and the task guardrail (unread document, `claim_history` not called, invalid JSON, invented quote, a good answer). 7 passed.
- `tests/test_live.py` (`pytest -m live`): the 9 demo runs with expected decisions and Scenario 3's fraud flag. Written, not run (it spends model calls).
- `README.md` rewritten as one page; `docs/CREWAI.md` and `docs/RAG.md` deleted (out of date). The README has no evaluation numbers yet, because there are none.

`pytest`: 87 passed, 9 live tests deselected.

**Step 9 shared part, check (2026-10-30, Docker 29.8.0, Compose 5.5.1, no model calls):** `docker compose -f deploy/docker-compose.yml build` built `claimsleuth-app` (606 MB, PyTorch 2.15.0+cpu) and `claimsleuth-crew` (1.55 GB). `APP_HOST_PORT=8095 docker compose ... up -d`: both containers healthy; `/healthz` answered on both; inside the app container `search_policy` ranked "2. Deductible" first with `embedding:sentence-transformers/all-MiniLM-L6-v2` (the model loaded offline, no TF-IDF fallback); the crew container read both Scenario 3 documents from `http://app:8000/mcp` with the secret; the app container fetched the crew's agent card from `http://crew:8101`; `/api/claims` without a token got `401`; a row written before `docker compose restart app` was still there after it (named volume). Not done: deployment to a cloud target (step 9 options A or B).

**Measured per run so far (`gemini-3.5-flash-lite`, hierarchical, direct):** run `6e93f40cc2ce` (Scenario 3, before the `record_facts` fix) 22 model calls, 75,756 + 5,408 tokens, 113 s, $0.0362 at the paid price; runs `cfdad4e7b00d` and `fd1de031e1da` (uploaded PDF claim) 21 calls each, 52,587 + 2,308 and 53,286 + 2,462 tokens, 69 s and 92 s, $0.0215 and $0.0221.

## Free tier only (2026-10-30)

The project stays on `gemini-3.5-flash-lite` on the free tier; no paid key will be used.

- **Evaluation reduced from 88 to 44 runs:** experiment A is 3 patterns x 2 prompt styles x claims E1, E3, E4, E6 = 24 runs; experiment B is the best configuration of A x the same 4 claims x 5 personas = 20 runs. All 8 packets and 40 persona statements stay in `eval/`. `eval/run_eval.py` (`EVAL_CLAIMS`), `eval/charts.py`, the tests and the README match.
- **Rate limits** (`src/claimsleuth/ratelimit.py`): model calls are spaced to `config.MODEL_MAX_RPM` (15 until the AI Studio figure is known): by `before_model_callback` in the app and a CrewAI `before_llm_call` hook in the crew service. The measured waits are subtracted from the agent timings: an agent excludes its own waits, and agents that contain others (the Claim Manager and the pattern wrappers) exclude every wait in their span, so a parallel agent is not charged for another's wait. The crew reports its waits in the A2A reply. Runs store them in `runs.wait_seconds`; `runs.seconds` is time without them. A 429 is no longer retried inside a run (removed from both clients' retry lists, which keep 5xx), so no wait can hide inside an agent's time. A 429 ends the run with an error starting `RATE_LIMIT_DAILY` or `RATE_LIMIT_MINUTE`, classified from the full error text (Gemini's message names the quota, for example "requests per day"); a crew-side 429 fails the run the same way instead of being recorded as `bad_output`. `eval/run_eval.py` stops on a daily limit with "daily quota reached, resume tomorrow" (finished runs stay in `runs.csv`; the interrupted run is redone next time) and keeps its 60 to 900 s backoff for per-minute limits, logged in `rate_limit_wait_s` and not in agent time.
- **Free-tier data note** added to the README: Google may use free-tier inputs to improve its products; the demo uses synthetic claims only, and real customer documents must not be used on the free tier.

### Input tokens: where they go and one change

From `agent_metrics` (runs `cfdad4e7b00d` and `fd1de031e1da`, hierarchical, about 52,600 and 53,300 input tokens each), the Policy agent uses the most: 20,532 and 20,767 input tokens (39%), about 4,100 per model call over 5 calls, because each call re-sends the earlier search results. Then the Claim Manager (23%), Evidence (13%), Settlement (11%).

**Change made (does not change what any agent sees or does):** ADK passed every MCP result to the model twice, as indented JSON text in `content` and again as `structuredContent`. `after_tool_callback` now passes the validated structured result once (`{"result": ...}`). Measured on the real `McpToolset` output for a typical run's calls (offline, no model calls): `parse_document` 2,654 -> 1,025 characters, `search_policy` 3,764 -> 1,433 and 3,020 -> 1,157, `check_coverage` 366 -> 138, `compute_payout` 611 -> 234, about 61-62% less per result; the Policy agent's three results go from 7,150 to 2,728 characters per copy, and it re-sends them on every later call. The crew is not affected (it uses CrewAI's own MCP client). The live per-agent input tokens after the change come from the step 4 check below.

**Suggested, not applied (it would change behaviour):** let the Policy agent search for several topics in one query instead of "one topic at a time". It made 2 to 4 searches and 5 model calls per run; one fewer model call saves about 4,000 input tokens. This changes the agent's instructions, so it waits for a decision.
