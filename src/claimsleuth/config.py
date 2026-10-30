"""Settings for ClaimSleuth. Everything that can change between machines comes from the environment."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Load KEY=value lines from .env without overriding variables already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(BASE_DIR / ".env")

# Model. One provider (Gemini); the name is pinned here and can be overridden with GEMINI_MODEL.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Secrets (empty means "not configured").
A2A_SHARED_SECRET = os.environ.get("A2A_SHARED_SECRET", "")
ADJUSTER_TOKEN = os.environ.get("ADJUSTER_TOKEN", "")

# Human review: payouts above this need a full review (USD).
FULL_REVIEW_THRESHOLD = 150.0

# Where the app listens, and where agents reach its MCP endpoint and the CrewAI service.
PORT = int(os.environ.get("PORT", "8000"))
APP_URL = (os.environ.get("APP_URL") or f"http://127.0.0.1:{PORT}").rstrip("/")
MCP_URL = f"{APP_URL}/mcp"
CREW_URL = (os.environ.get("CREW_URL") or "http://127.0.0.1:8101").rstrip("/")

# Limits per run.
MAX_LLM_CALLS = 30
RUN_TIMEOUT_S = 15 * 60
MODEL_RETRY_ATTEMPTS = 5          # retries on 429/5xx from the model API (Gemini returned 503s during the build)

# Model prices in USD per million tokens, from the provider's price list, with the date they were read.
# Cost is reported only for models listed here; for any other model it is "not measured".
PRICES: dict[str, dict] = {}

# Paths. DATA_DIR holds everything the app writes; override it in containers.
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data" / "runtime"))
POLICIES_FILE = BASE_DIR / "configs" / "policies.json"
DEMO_CLAIMS_DIR = BASE_DIR / "data" / "sample_claims"
AUDIT_LOG_DIR = Path(os.environ.get("CLAIMSLEUTH_AUDIT_LOG_DIR", DATA_DIR / "audit_logs"))
TRACES_DIR = Path(os.environ.get("CLAIMSLEUTH_TRACES_DIR", DATA_DIR / "traces"))
RAG_CORPUS_DIR = Path(os.environ.get("CLAIMSLEUTH_RAG_CORPUS_DIR", BASE_DIR / "data" / "rag" / "policies"))
RAG_INDEX_DIR = Path(os.environ.get("CLAIMSLEUTH_RAG_INDEX_DIR", BASE_DIR / "data" / "rag" / "index"))


def model_configured() -> bool:
    return bool(GOOGLE_API_KEY)
