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

# Paths
DATA_DIR = BASE_DIR / "data"
POLICIES_FILE = BASE_DIR / "configs" / "policies.json"
AUDIT_LOG_DIR = DATA_DIR / "audit_logs"
SAMPLE_CLAIMS_DIR = DATA_DIR / "sample_claims"

# Ensure dirs exist
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
