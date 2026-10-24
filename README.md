# ClaimSleuth: Agentic AI Warranty Investigation System

ClaimSleuth is a robust, transparent, and auditable warranty investigation platform that bridges agentic AI reasoning with deterministic eligibility verification and cryptographic audit logging.

Built for academic presentation and real-world compliance, ClaimSleuth solves the dual dangers of automated claim processing: rigid rule fragility and reckless generative AI hallucinations.

---

## Architecture Highlights

1. **Agentic Investigation Loop**:
   - `PLAN → EXECUTE → OBSERVE → ASSESS → REPLAN or STOP`
   - Bounded execution (hard limits of 5 iterations and 8 model calls per investigation).
   - Powered by **Google Gemini** structured output with automatic, state-aware adaptive fallback.
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

## Quickstart & Demo Guide

### 1. Run Automated Unit & Regression Tests
Run the complete test suite verifying deterministic date engines, arithmetic capping, dual-gate verifier blocks, tool registry schemas, auth enforcement, dual-mode extractor, adaptive replanning, and the 3 benchmark scenarios:

```bash
python -m unittest tests/test_claim_sleuth.py -v
```
*(All 31 tests pass with 100% assertion coverage).*

### 2. Optional: Configure Gemini API Key
To enable real Gemini model-guided planning and extraction:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export USE_LLM=true
# Optional: export GEMINI_MODEL="gemini-2.0-flash"
```
*(If no API key is set or `USE_LLM=false`, ClaimSleuth automatically operates in deterministic fallback mode with zero degradation in benchmark pass rates).*

### 3. Launch the Interactive Web Dashboard
Run the FastAPI server:

```bash
python -m uvicorn src.ui.app:app --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000`**

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
│   └── audit_logs/                # Immutable SHA-256 hashed audit logs (.jsonl)
├── src/
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
    └── test_claim_sleuth.py       # 31-test automated suite (Phase 1-3 validation)
```
