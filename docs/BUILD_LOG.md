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
