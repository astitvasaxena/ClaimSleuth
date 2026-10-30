# ClaimSleuth

ClaimSleuth is a team of AI agents that investigates a product warranty claim and hands a human adjuster a cited recommendation: approve, reject or ask for more evidence. The agents decide; MCP tools do the date maths, the money and the database lookups; guardrail callbacks check every step. Nothing is paid without a person's approval.

You can run the three demo claims or upload your own (PDF, TXT or MD). Both go through the same agents.

## How it works

```
Dashboard (login, upload, live timeline, report, review)
   |
   v
claimsleuth-app  (FastAPI + Google ADK)                    claimsleuth-crew  (CrewAI)
   Claim Manager (LlmAgent) --- delegates to ---------.      Fraud Crew: Forensics analyst
     Evidence  -> parse_document                      |                  History analyst
     Policy    -> search_policy, check_coverage       +--A2A--> (RemoteA2aAgent "fraud_crew")
     Settlement-> compute_payout                      |
   Blackboard = ADK session state (bb_claim, bb_facts, bb_coverage, bb_fraud, bb_payout, bb_report)
   Callbacks on every agent: guardrails + metrics      |
   MCP server at /mcp (5 tools) <----------------------'  (the crew calls parse_document, claim_history)
   SQLite: claims, documents, reports, reviews, runs, agent_metrics, past_claims
   OpenTelemetry trace across ADK -> A2A -> CrewAI -> MCP; hash-chained audit log
```

- **Model:** one pinned model for everything, `gemini-3.5-flash-lite` (`src/claimsleuth/config.py`).
- **Patterns:** sequential, parallel (Policy and the crew at the same time) and hierarchical (the Manager chooses who to call; the default).
- **Prompt styles:** `direct` and `react` (ReAct instructions plus ADK's `PlanReActPlanner`).

## Run it

You need Docker and a Gemini API key.

```bash
cp .env.example .env          # set GOOGLE_API_KEY, A2A_SHARED_SECRET, ADJUSTER_TOKEN
docker compose -f deploy/docker-compose.yml up --build
```

Open `http://127.0.0.1:8000` and sign in with `ADJUSTER_TOKEN`. To use another host port, set `APP_HOST_PORT`.

Without Docker: `pip install -r requirements.txt && pip install -e .`, start the app with `python -m claimsleuth.api.app`. For the crew, create `.venv-crewai`, install `requirements-crewai-service.txt` and `pip install -e . --no-deps` into it, then run `.venv-crewai/Scripts/python -m claimsleuth.crew.a2a_server --port 8101`.

## The 11 requirements

| # | Requirement | Where | Proof |
| --- | --- | --- | --- |
| 1 | Blackboard + another messaging style | `agents/common.py` (`bb_*` keys), `agents/fraud_remote.py` (A2A request/reply) | `python -m claimsleuth.runner --claim CLM-2024-003` prints the blackboard |
| 2 | MCP, at least two external tools | `mcp_server/server.py`: pypdf, RAG embeddings, date maths, calculator, SQLite | `python eval/mcp_smoke.py --url http://127.0.0.1:8000/mcp` |
| 3 | Pydantic structured output | `models.py` (`FinalReport`), `submit_report` | `pytest tests/test_models.py` |
| 4 | Hierarchical, parallel, sequential, compared | `agents/patterns.py` | `pytest -m live`; experiment A in `eval/results/summary.md` |
| 5 | Monitoring callbacks, per-agent heat map | `callbacks/hooks.py`, `callbacks/metrics.py`, `charts.py` | `GET /api/runs/{run_id}/heatmap.png` after a run |
| 6 | A2A between ADK and CrewAI | `agents/fraud_remote.py`, `crew/a2a_server.py`, `crew/crew.py` | `GET /api/runs/{run_id}/trace` shows `crewai.crew` under `fraud_crew` |
| 7 | Only CrewAI and ADK | `requirements*.txt` | `grep -rEi "langchain|langgraph|autogen|llama_index" src requirements*.txt` finds nothing |
| 8 | Guardrails, tool-result checks, prompt variants, personas | `callbacks/hooks.py`, `crew/crew.py` (task guardrails), `agents/prompts/`, `eval/personas/` | `pytest tests/test_guardrails.py`; `python eval/check_eval_set.py` |
| 9 | Tracing, audit, provenance | `tracing.py`, `audit.py`, quotes checked against documents | `GET /api/audit/{claim_id}/verify`; `GET /api/runs/{run_id}/trace` |
| 10 | ~40 runs, cost, time, prompt variance, boxplots | `eval/run_eval.py`, `eval/charts.py` | `python eval/run_eval.py --experiment A`, then `B`, then `python eval/charts.py` |
| 11 | Cloud deployment | `deploy/` | not deployed yet |

Tests: `pytest` (offline; fake models are labelled as test doubles), `.venv-crewai/Scripts/python -m unittest tests.crew.test_crew_service` (crew side), `pytest -m live` (spends model calls).

## Evaluation

Not run yet. The claim set is ready: 8 claim packets with known answers in `eval/claims/` and 40 customer statements in 5 personas (honest, abusive, vague, fraudster, injector) in `eval/personas/`. Experiment A is 48 runs and experiment B is 40 runs. Results, charts and costs will be in `eval/results/summary.md`, and the dashboard's Evaluation page shows them.

## Limits

- No OCR: scanned PDFs without a text layer are stored but cannot be read.
- No email intake and no payments: a person approves, and paying is outside the system.
- The policy library is three synthetic policies (`configs/policies.json`); uploaded policies are searched for their own claim only.
- The injection and abuse screens are pattern lists. They catch the phrasings they list, not every paraphrase.
- The Gemini free tier allows about 20 requests per model per day on some models; one investigation uses about 20 model calls.

The build history, every difference from the plan and every check is in `docs/BUILD_LOG.md`.
