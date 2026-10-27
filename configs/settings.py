import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Hard safety constraints on investigation loops
MAX_INVESTIGATION_STEPS = 5
MAX_MODEL_CALLS = 8
HIGH_VALUE_THRESHOLD = 150.0  # Claims above $150 strictly require Human sign-off

# Gemini / LLM configuration (read from environment, never hardcoded)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
USE_LLM = os.environ.get("USE_LLM", "false").lower() in ("true", "1", "yes")

# Adjudication endpoint authorization token (required for human-approval API)
ADJUDICATION_TOKEN = os.environ.get("CLAIMSLEUTH_ADJUDICATION_TOKEN", "")

# Browser origins allowed to call the API cross-origin (comma-separated).
# The dashboard is served same-origin, so it does not depend on this list.
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CLAIMSLEUTH_CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",") if o.strip()
]

# Paths
DATA_DIR = BASE_DIR / "data"
POLICIES_FILE = BASE_DIR / "configs" / "policies.json"
# Runtime audit ledgers (gitignored). Override with CLAIMSLEUTH_AUDIT_LOG_DIR
# (tests use this to stay out of the repository entirely).
AUDIT_LOG_DIR = Path(os.environ.get("CLAIMSLEUTH_AUDIT_LOG_DIR", DATA_DIR / "runtime" / "audit_logs"))
# FinalReport JSON/Markdown files from the ADK pipeline (gitignored).
REPORTS_DIR = Path(os.environ.get("CLAIMSLEUTH_REPORTS_DIR", DATA_DIR / "runtime" / "reports"))
# OpenTelemetry span files, one JSONL file per trace id (gitignored).
TRACES_DIR = Path(os.environ.get("CLAIMSLEUTH_TRACES_DIR", DATA_DIR / "runtime" / "traces"))
# RAG: policy wording documents (committed) and the generated vector index (gitignored)
RAG_CORPUS_DIR = Path(os.environ.get("CLAIMSLEUTH_RAG_CORPUS_DIR", DATA_DIR / "rag" / "policies"))
RAG_INDEX_DIR = Path(os.environ.get("CLAIMSLEUTH_RAG_INDEX_DIR", DATA_DIR / "rag" / "index"))
# Ledgers committed in 850d757, written by the pre-fix logger (chain restarted at
# GENESIS on every run). Preserved unchanged as evidence; never written to.
LEGACY_AUDIT_LOG_DIR = DATA_DIR / "audit_logs"
# Claim dossiers the app investigates; CLAIMSLEUTH_CLAIMS_DIR points it at another folder
# (e.g. the synthetic benchmark in data/benchmark/claims)
SAMPLE_CLAIMS_DIR = Path(os.environ.get("CLAIMSLEUTH_CLAIMS_DIR", DATA_DIR / "sample_claims"))

# Ensure dirs exist
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
