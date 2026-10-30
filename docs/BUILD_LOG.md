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
