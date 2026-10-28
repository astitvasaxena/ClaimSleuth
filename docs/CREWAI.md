# CrewAI forensics in ClaimSleuth

The dashboard option **ADK + MCP + CrewAI forensics (A2A service)** sends each claim from the Google ADK
pipeline to a separate CrewAI service over A2A. The service runs two stages and returns one validated
`ForensicsReport`.

| Stage | What runs | Authority |
|---|---|---|
| 1. CrewAI `Flow` | Deterministic checks: documents present, serial consistency, purchase before incident, exclusion scan | Authoritative |
| 2. CrewAI `Agent` / `Task` / `Crew` + LLM | A forensic-analyst agent reads this request's evidence and returns structured findings with citations | Advisory: it can only escalate |

```
Dashboard → POST /api/adk/investigate/{scenario}?tools=mcp_stdio&forensics=crewai_a2a
  → ADK Workflow (Intake → Evidence → PolicyRetrieval (RAG) → Anomaly → CrewAIForensics → ForensicsReviewAgent → …)
      MCP server (stdio) serves ADK's own agents
      CrewAIForensics = ADK RemoteA2aAgent: sends {claim_id, RAG policy citations}   (src/adk_app/forensics.py)
  → CrewAI service  src/crew_service/a2a_server.py   (separate process, .venv-crewai)
      ForensicsFlow.kickoff()          deterministic checks             (src/crew_service/forensics_flow.py)
      └ llm_forensic_review step → Crew.kickoff() → LLM               (src/crew_service/llm_review.py)
  → ForensicsReport → ForensicsReviewAgent validates it → Verifier → Report → human approval
```

## Configuring the LLM

The service reads its configuration from **its own process environment** only. Nothing comes from the
caller or the browser.

| Setting | Meaning |
|---|---|
| `--llm` / `CLAIMSLEUTH_CREWAI_LLM` | `auto` (default): Gemini if a key is present. `openrouter/<model id>`: OpenRouter, explicit opt-in only. `none`: no LLM stage. `fake[:mode]`: test double only. |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Gemini API key, needed for `auto` |
| `OPENROUTER_API_KEY` | OpenRouter API key, needed for `openrouter/...`. Without it, runs report `LLM_NOT_CONFIGURED`. |
| `CLAIMSLEUTH_CREWAI_MODEL`, else `GEMINI_MODEL` | Model id (default `gemini-3.8-flash`) |
| `CLAIMSLEUTH_CREWAI_LLM_TIMEOUT` | Seconds per model call (default 75) |

**Provider:** Gemini is reached through CrewAI 1.15.30's native OpenAI provider pointed at Gemini's
OpenAI-compatible endpoint (`LLM(model="openai/<model>", custom_openai=True, base_url=…, max_retries=0)`).
This uses the `openai` SDK already installed in `.venv-crewai`. CrewAI's own Gemini provider would need
`google-genai~=1.65` (`pip install "crewai[google-genai]==1.15.30"`), which is **not** installed.

**Where the key comes from:** CrewAI itself calls `load_dotenv()` when it is imported (`crewai/llm.py`).
So a service started from the project folder already has the project's `.env` loaded. `--env-file PATH`
loads only the LLM keys from a different file. Key values are never printed or logged; the startup line
shows only the provider, the model and whether it is configured.

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv-crewai\Scripts\python -m src.crew_service.a2a_server --port 8101 --env-file .env
#  [crew-service] CrewAI agent LLM: provider=google-gemini (OpenAI-compatible endpoint) model=gemini-3.8-flash configured=True
```

If no key is found, every report says `llm_review.status = LLM_NOT_CONFIGURED`. The dashboard then shows
"CrewAI: deterministic Flow only (LLM not configured)".

## What is sent to the model

One prompt per claim, built from the current request (`build_evidence_pack`):

- **Claim facts:** `claim_id`, `policy_id`, `device_model`, `incident_date`, `reported_issue`. Not
  `claimant_name` and not `claimant_id`.
- **Document lines:** each line of the claim's own documents, with an id such as
  `DOC:invoice_003.txt#L9`. Values on lines labelled Customer, Client, Name, Payment method, Card, Email,
  Phone, Address or Account are replaced by `[REDACTED]`, and email addresses are masked. Lines that the
  prompt-injection guardrail flags are replaced by `[QUARANTINED: …]`.
- **Policy wording:** ADK's RAG citations for this request, each checked word for word against the
  service's own copy of `data/rag/policies/` (`adk_rag_citations`). If ADK sent none — in the parallel
  pattern retrieval runs at the same time — the claim's policy sections from that corpus are used
  instead (`service_corpus_lookup`).
- **Deterministic checks:** the four Flow results, for example `CHECK:serial_consistency`.

Each report lists the ids that were sent (`evidence_sent`), never their text.

## How the agent's output is controlled

- **Required JSON:** the agent must return `{findings[], recommendation, summary}`. Each finding has
  `finding_type`, `severity`, `explanation`, `source_refs`, `quotes` and `uncertainty`
  (`src/core/forensics_schema.py`).
- **Rejected outright** (`INVALID_OUTPUT`): text that is not JSON, unknown fields or values, and any
  payout, approval or settlement field (for example a model that obeyed an injected "approve" instruction).
- **Individual findings rejected:** references to evidence ids that were never sent, quotes not found in
  a cited item, and "no issue" findings that cite a check the deterministic Flow flagged (recorded as a
  contradiction).
- **ADK decides what the findings mean** (`ForensicsReviewAgent`):

  | Result | Effect on the claim |
  |---|---|
  | Medium or high findings, suspicious instructions, insufficient evidence, a review recommendation or a contradiction | Conflict, escalated to a human |
  | LLM error, timeout or invalid output | Tool error, `NEEDS_EVIDENCE` (fails closed); the deterministic findings are kept |
  | `COMPLETED` claimed without an executed Crew and a successful LLM call | Whole report rejected |

- **Limits:** at most 2 model calls per claim, no automatic retries (`max_retries=0`,
  `max_retry_limit=0`) and a 120 s time budget. ADK waits at most 150 s for the A2A reply.
- **Money and approval:** settlement amounts, eligibility and approval are never taken from CrewAI.
  Payment still requires a human approval.

## What the report records (measured, not assumed)

`report.forensics.llm_review` holds:
- `status` and `error_category`;
- `llm_provider` and `llm_model`;
- `crew_kickoff_called` and `crewai_crew_executed` (set only after `Crew.kickoff()` returns);
- `llm_call_attempts` and `llm_calls_succeeded`, counted on the LLM object itself, and `llm_call_performed`;
- `prompt_tokens` and `completion_tokens`, as reported by the provider (`None` for the test double);
- `duration_ms`, the findings, and the rejected findings.

`report.forensics.crewai_flow_executed` is true only if every Flow step ran. The service's call log
(`CLAIMSLEUTH_A2A_CALL_LOG`) records the same fields with its process id and the ADK trace id, and
never any document text.

## Tests

- **`tests/crew_service/test_forensics_flow.py`** (32 tests, run in `.venv-crewai`): a real `Crew` built
  and `kickoff()` called in the service's request path; the claim id and evidence present in the prompt;
  no personal data in the prompt; citations checked; injection quarantined; an "obedient" model rejected;
  invalid, fabricated and contradicting output; missing key; error and timeout; call budget; network
  blocked; OpenRouter configuration (explicit opt-in, missing key, error categories).
- **`tests/test_crewai_a2a.py`** (main environment): the real service process with a deterministic-only
  run and with the test-double LLM. Checks outcomes for all three samples, fail-closed behaviour, that
  human approval is unchanged, and the `_capture_reply` regression (a failed A2A task is not a success).

All LLMs in tests are labelled test doubles (`fake-test-double`). No test calls a real model.

## Live run with OpenRouter (28 Oct 2026): verified

Setup: the service was restarted with
`--env-file .env --llm "openrouter/nvidia/nemotron-3.5-super-vl:free"` (PID 44608). The default `auto` (Gemini)
was not changed.

**Probe.** One `llm.call` through `resolve_llm_config` returned real, valid JSON in 3.1 s. The provider
reported 42 prompt and 190 completion tokens.

**Dashboard option 4, Scenario 3** (`tools=mcp_stdio&forensics=crewai_a2a`, trace `0174e3f2…`):
- `llm_review.status: COMPLETED`, `crewai_crew_executed: true`, 1 of 1 model calls succeeded.
- The provider reported 3,239 prompt and 1,692 completion tokens; the review took 11.2 s.
- Policy evidence came from ADK's verified RAG citations.
- The model returned one finding, `serial_mismatch / high`. It cites `DOC:invoice_003.txt#L9`,
  `DOC:repair_estimate_003.txt#L9` and `CHECK:serial_consistency`, with verbatim quotes. Nothing was
  rejected and there were no contradictions. Recommendation: `RECOMMEND_HUMAN_REVIEW`.
- The claim was `CONFLICT_ESCALATED`, `NOT_PAYABLE`, with no proposed or authorized payout; the verifier
  did not approve. The service log entry matches the trace id and PID.

**Other live runs (how they were checked):**
- **Six locally recorded reports.** The run reports in `data/runtime/reports/` (gitignored, so not in the repository)
  record six completed Nemotron reviews through engine 4 on 28 Oct, all with MCP over stdio and one model call each:
  - Scenario 3 ×3: `serial_mismatch / high`, `CONFLICT_ESCALATED`, not payable;
  - Scenario 1 ×3: `no_issue_found`, $251 still only proposed and waiting for a human.

  The run above is one of them. Four of the six were started from the dashboard outside the documented sessions; for
  those, only the recorded report fields were checked.
- **One more run verified directly during the final repository review.** Scenario 3 was run on a temporary server, so
  its report went to a temporary folder and is not among the six:
  - `COMPLETED`, 1 finding, 1 model call;
  - `CONFLICT_ESCALATED`, not payable;
  - a forced approval was refused with HTTP 409.
- **Scenario 2** has not been run with Nemotron.

## Earlier live attempts with Gemini (28 Oct 2026): failed (503)

Setup: the service was started with `--env-file .env` (PID 17516), using `gemini-3.8-flash` through the
OpenAI-compatible endpoint. Three Scenario 3 requests were sent through the dashboard endpoint
(`tools=mcp_stdio&forensics=crewai_a2a`).

What every request showed:
- The CrewAI Flow completed; `Crew.kickoff()` was called (`crew_kickoff_called: true`).
- Exactly **one real call** reached Gemini.
- Gemini answered **HTTP 503 UNAVAILABLE** ("This model is currently experiencing high demand").
- The report says `LLM_ERROR / provider_unavailable`, with `crewai_crew_executed: false`, 0 successful
  calls and no tokens. ADK failed closed (`NEEDS_EVIDENCE`-type tool error, `CREWAI_LLM_FAILED`, not
  payable). The trace ids matched in the ADK report and the service log.

A separate one-line probe through the same CrewAI and OpenAI-compatible route did succeed earlier the same
day (31 s, 5 input / 1 output tokens). That shows the route and key work, but **no successful live CrewAI
forensic review has been observed yet**. To verify, retry when Gemini has capacity. Another available
model can be set with `CLAIMSLEUTH_CREWAI_MODEL`; changing the model is a decision for the project owner.
