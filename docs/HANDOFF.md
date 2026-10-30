# ClaimSleuth handoff (31 Oct 2026)

Branch `rebuild`. The plan is [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Every step, deviation and check is in [BUILD_LOG.md](BUILD_LOG.md). Deadlines: 70% by 30 Oct 2026, final by 2 Nov 2026.

## Where things stand

| Step | Status |
| --- | --- |
| 1. Remove the duplicates | Done |
| 2. Data models and SQLite schema | Done |
| 3. MCP server | Done. It has 6 tools; `check_invoice_totals` was added after the step 4 check |
| 4. ADK agents and the three patterns | Done. The live check found 8 of 9 correct; after two fixes, 3 of 3 were correct |
| 5. CrewAI fraud crew over A2A | Done |
| 6. Callbacks, guardrails, heat maps, tracing | Done |
| 7. API, uploads, storage, login, dashboard | Done |
| 8. Evaluation claim set and runner | Code done. The evaluation is **paused at 12 of 44 runs** |
| 9. Deployment | Dockerfiles and compose file done; the deployment target is not chosen yet |
| 10. Tests and README | 96 pytest tests and 9 crew tests pass. README numbers wait for the evaluation |

The plan says 88 runs. The evaluation was reduced to 44 runs to fit the free tier:

- **Experiment A:** 3 patterns × 2 prompt styles × claims E1, E3, E4, E6 (24 runs).
- **Experiment B:** the best configuration from A × the same 4 claims × 5 personas (20 runs).

Day 1 ran on commit `8ca4763` with `gemini-3.5-flash-lite`, starting 2026-10-30T19:31:47+0530. It stopped at the daily quota. The results are in `eval/results/`: `runs.csv`, `run_info.json`, `rate_limit_retries.csv` and the per-run `heatmaps/`.

Experiment A so far is **8 of 12 correct**:

- 1 failed run: E1 sequential/react, "the Claim Manager did not submit a report".
- 3 wrong decisions:
  - E3 sequential/react gave request_evidence instead of reject.
  - E6 sequential/direct and E6 parallel/direct gave reject instead of request_evidence.

Experiment B has not started.

## The pending decision (the project owner decides)

E6 is a claim with no invoice, so the right answer is `request_evidence`. The Policy prompt (`agents/prompts/direct/policy.md`, line 18, and the matching line in `react/policy.md`) says that with no purchase date it should set `covered` to false. The Manager then reads that as "not covered" and rejects the claim. Both direct-prompt E6 runs failed this way.

- **Option 1, resume as is:** run the remaining 32 runs on `8ca4763` (about 460 model calls) and report the flaw as a finding.
- **Option 2, fix and restart:** fix the prompt, commit, move the day-1 results into an archive folder (for example `eval/results/day1_8ca4763/`), and rerun all 44 runs on the new commit (about 630 calls, two days on the free tier).

Either way, a different machine keeps the same commit and model, but some packages are not pinned. Compare your installed versions with the "Installed versions" table in BUILD_LOG and write down any differences.

## Setup from a fresh clone (Windows, PowerShell, Python 3.12)

```powershell
git clone <repo-url>; cd claim_sleuth; git checkout rebuild
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt; .venv\Scripts\pip install -e .
py -3.12 -m venv .venv-crewai
.venv-crewai\Scripts\pip install -r requirements-crewai-service.txt; .venv-crewai\Scripts\pip install -e . --no-deps
copy .env.example .env
```

Fill in `.env`. Never commit it.

| Variable | What it is |
| --- | --- |
| `GOOGLE_API_KEY` | A Gemini API key from AI Studio (free tier) |
| `A2A_SHARED_SECRET` | Any long random string, shared by the app, `/mcp` and the crew |
| `ADJUSTER_TOKEN` | Any long random string; it is the dashboard login and the eval runner's token |
| `PORT` | Set it to `8090`. It is not in `.env.example`; the default is 8000 |

Both services read the same `.env`. Start them in two terminals:

```powershell
.venv\Scripts\python -m claimsleuth.api.app                                   # app, MCP at /mcp: 127.0.0.1:8090
.venv-crewai\Scripts\python -m claimsleuth.crew.a2a_server --port 8101        # CrewAI fraud crew: 127.0.0.1:8101
curl.exe http://127.0.0.1:8090/healthz   # {"status":"ok","model":"gemini-3.5-flash-lite","model_configured":true}
curl.exe http://127.0.0.1:8101/healthz   # {"status":"ok","model":"gemini-3.5-flash-lite"}
```

The first start creates `data/runtime/` and seeds the demo and evaluation claims. It may also download the embedding model. Run the offline tests with `.venv\Scripts\python -m pytest` and the crew tests with `.venv-crewai\Scripts\python -m unittest tests.crew.test_crew_service`.

**Resuming the evaluation.** The runner skips runs that are already in `runs.csv`. It refuses to start if the evaluated code differs from the commit in `run_info.json`. It stops with exit code 3 and the message "daily quota reached, resume tomorrow". The quota resets at midnight Pacific (12:30 IST).

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUNBUFFERED="1"
$t = "local: app and crew on this machine (127.0.0.1:8090), not deployed"
.venv\Scripts\python eval\run_eval.py --url http://127.0.0.1:8090 --experiment A --target $t
.venv\Scripts\python eval\run_eval.py --url http://127.0.0.1:8090 --experiment B --target $t   # only after A is complete
.venv\Scripts\python eval\charts.py --url http://127.0.0.1:8090                              # after all 44
```

After each day, commit and push `eval/results/`.

## Rules

- **One model only:** `gemini-3.5-flash-lite`, hard-coded in `config.py`. Don't switch models, and never mix models within the evaluation.
- **No code, prompt or config changes during an evaluation.** If you find a bug, stop and report it. A crash, timeout or missing report counts as a failed run. Only runs cut off by the daily quota are redone (the runner does this itself).
- **Free tier, synthetic data only.** Google may use free-tier data, so never upload real customer documents.
- Secrets live only in `.env`, and `.env.example` keeps empty values. Never touch `main` until the merge.

## What's left

1. Finish the evaluation (the decision above first), then run `eval/charts.py`, which writes `summary.md` and the boxplots.
2. Put the numbers from `eval/results/summary.md` into the README.
3. Deployment (plan step 9): choose Cloud Run or a free host, deploy both containers, and check `/healthz` and one investigation on the deployed URL.
4. Merge `rebuild` into `main`.
