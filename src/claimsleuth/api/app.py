"""
ClaimSleuth web API: the dashboard, the API routes and the MCP server at /mcp.

Every /api/* route needs `Authorization: Bearer $ADJUSTER_TOKEN`; /mcp needs the shared secret header; only
/healthz and the dashboard page itself (a shell with no data in it) are open.
"""
import hmac
import json
import re
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from claimsleuth import audit, charts, config, db, tracing
from claimsleuth.agents.common import PROMPT_STYLES
from claimsleuth.agents.patterns import PATTERNS
from claimsleuth.api.jobs import AlreadyRunning, Jobs
from claimsleuth.mcp_server import server as mcp_server
from claimsleuth.mcp_server.pdf import NoTextLayer, read_lines
from claimsleuth.models import FinalReport

NO_KEY_MESSAGE = "No model key configured; investigations are disabled"
SECRET_HEADER = "X-ClaimSleuth-Secret"
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
EVAL_RESULTS_DIR = config.BASE_DIR / "eval" / "results"
MAX_FILES, MAX_FILE_BYTES = 10, 5 * 1024 * 1024
ALLOWED_SUFFIXES = (".pdf", ".txt", ".md")
POLICIES = json.loads(config.POLICIES_FILE.read_text(encoding="utf-8"))


class McpEndpoint:
    """Forwards /mcp to the MCP app built for the current app lifespan (its session manager can run only once)."""
    app = None

    async def __call__(self, scope, receive, send):
        if self.app is None:
            raise RuntimeError("MCP endpoint used outside the app lifespan")
        await self.app(scope, receive, send)


mcp_endpoint = McpEndpoint()
jobs: Optional[Jobs] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs
    db.init()
    jobs = Jobs()
    mcp_endpoint.app = mcp_server.http_app()
    async with mcp_server.server.session_manager.run():
        yield
    mcp_endpoint.app = None


app = FastAPI(title="ClaimSleuth", lifespan=lifespan)


def secret_ok(value: Optional[str]) -> bool:
    return bool(config.A2A_SHARED_SECRET) and hmac.compare_digest(value or "", config.A2A_SHARED_SECRET)


def token_ok(header: Optional[str]) -> bool:
    scheme, _, token = (header or "").partition(" ")
    return (bool(config.ADJUSTER_TOKEN) and scheme.lower() == "bearer"
            and hmac.compare_digest(token.strip(), config.ADJUSTER_TOKEN))


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if path == "/mcp" or path.startswith("/mcp/"):
        if not secret_ok(request.headers.get(SECRET_HEADER)):
            return JSONResponse(status_code=401, content={"detail": f"missing or wrong {SECRET_HEADER} header"})
        token = tracing.REMOTE_TRACEPARENT.set(request.headers.get("traceparent"))
        try:
            return await call_next(request)
        finally:
            tracing.REMOTE_TRACEPARENT.reset(token)
    if path.startswith("/api/"):
        if not config.ADJUSTER_TOKEN:
            return JSONResponse(status_code=503, content={"detail": "ADJUSTER_TOKEN is not configured; the API is closed"})
        if not token_ok(request.headers.get("authorization")):
            return JSONResponse(status_code=401, content={"detail": "missing or wrong adjuster token"})
        if not config.model_configured():
            return JSONResponse(status_code=503, content={"detail": NO_KEY_MESSAGE})
    return await call_next(request)


# ─── Pages and health ──────────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": config.GEMINI_MODEL, "model_configured": config.model_configured()}


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/api/status")
def status():
    return {"model": config.GEMINI_MODEL, "patterns": PATTERNS, "prompt_styles": PROMPT_STYLES,
            "policies": [{"id": p["policy_id"], "name": p["name"]} for p in POLICIES.values()]}


# ─── Claims ────────────────────────────────────────────────────────────────────────────────

def latest(conn, table: str, claim_id: str):
    return conn.execute(f"SELECT * FROM {table} WHERE claim_id = ? ORDER BY rowid DESC LIMIT 1", (claim_id,)).fetchone()


@app.get("/api/claims")
def list_claims():
    with db.connect() as conn:
        claims = [dict(r) for r in conn.execute("SELECT * FROM claims WHERE source != 'eval' ORDER BY source, created_at, id")]
        for claim in claims:
            report = latest(conn, "reports", claim["id"])
            review = latest(conn, "reviews", claim["id"])
            claim["decision"] = json.loads(report["report_json"])["decision"] if report else None
            claim["review"] = dict(review) if review else None
    return claims


@app.get("/api/claims/{claim_id}")
def get_claim(claim_id: str):
    claim = db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "unknown claim")
    with db.connect() as conn:
        runs = [dict(r) for r in conn.execute("SELECT * FROM runs WHERE claim_id = ? ORDER BY started_at DESC", (claim_id,))]
    return {**claim, "documents": db.get_documents(claim_id), "runs": runs}


def _safe_name(name: str, taken: set) -> str:
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:60] or "document"
    candidate, n = f"{stem}{suffix}", 2
    while candidate in taken:
        candidate, n = f"{stem}_{n}{suffix}", n + 1
    taken.add(candidate)
    return candidate


def _check_file(name: str, data: bytes) -> str:
    """The file's real type, checked by content as well as extension."""
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"{name}: only PDF, TXT and MD files are accepted")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{name}: larger than 5 MB")
    if not data:
        raise HTTPException(422, f"{name}: empty file")
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(415, f"{name}: not a PDF file")
        return "pdf"
    if b"\x00" in data:
        raise HTTPException(415, f"{name}: not a text file")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(415, f"{name}: text files must be UTF-8") from None
    return "text"


def _parse_date(value: Optional[str], field: str, required: bool) -> Optional[str]:
    if not value:
        if required:
            raise HTTPException(422, f"{field} is required")
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise HTTPException(422, f"{field} must be YYYY-MM-DD") from None


@app.post("/api/claims")
async def create_claim(claimant: str = Form(...), device: str = Form(...), incident_date: str = Form(...),
                       statement: str = Form(...), serial: str = Form(""), purchase_date: str = Form(""),
                       policy_id: str = Form(""), files: list[UploadFile] = File(...),
                       policy_file: Optional[UploadFile] = File(None)):
    if not claimant.strip() or not device.strip() or not statement.strip():
        raise HTTPException(422, "claimant, device and statement are required")
    if len(statement) > 4000:
        raise HTTPException(422, "statement is longer than 4000 characters")
    if policy_id and policy_id not in POLICIES:
        raise HTTPException(422, f"unknown policy {policy_id}")
    uploads = [(f, "other") for f in files if f.filename] + ([(policy_file, "policy")] if policy_file and policy_file.filename else [])
    if not policy_id and not any(kind == "policy" for _, kind in uploads):
        raise HTTPException(422, "choose a policy or upload the policy document")
    if not any(kind == "other" for _, kind in uploads):
        raise HTTPException(422, "attach at least one claim document")
    if len(uploads) > MAX_FILES:
        raise HTTPException(413, f"at most {MAX_FILES} files")
    contents = []
    for upload, kind in uploads:
        data = await upload.read(MAX_FILE_BYTES + 1)
        contents.append((upload.filename, kind, data, _check_file(upload.filename, data)))
    claim_id = f"UPL-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(3).upper()}"
    folder = Path(config.DATA_DIR) / "uploads" / claim_id
    folder.mkdir(parents=True, exist_ok=True)
    taken: set = set()
    with db.connect() as conn:
        db.insert_claim(conn, {"id": claim_id, "source": "upload", "claimant": claimant.strip(), "device": device.strip(),
                               "serial": serial.strip() or None,
                               "purchase_date": _parse_date(purchase_date, "purchase_date", False),
                               "incident_date": _parse_date(incident_date, "incident_date", True),
                               "statement": statement.strip(), "policy_id": policy_id or None})
        for filename, kind, data, filetype in contents:
            path = folder / _safe_name(filename, taken)
            path.write_bytes(data)
            pages, has_text = 1, True
            if filetype == "pdf":
                try:
                    pages = read_lines(path)[1]
                except NoTextLayer:
                    has_text = False
                except Exception:
                    raise HTTPException(422, f"{filename}: the PDF cannot be read") from None
            db.insert_document(conn, claim_id, path, f"uploads/{claim_id}/{path.name}",
                               kind="policy" if kind == "policy" else None, pages=pages, has_text=has_text)
    audit.AuditLogger(claim_id).log_step(0, "CLAIM_UPLOADED", f"{len(contents)} files", reasoning=claimant.strip())
    return {"claim_id": claim_id}


@app.get("/api/documents/{doc_id:path}")
def get_document(doc_id: str):
    try:
        return mcp_server.parse_document(doc_id).model_dump()
    except Exception as exc:
        raise HTTPException(404, str(exc)) from None


# ─── Runs ──────────────────────────────────────────────────────────────────────────────────

@app.post("/api/claims/{claim_id}/investigate")
async def investigate(claim_id: str, pattern: str = "hierarchical", prompt_style: str = "direct"):
    if db.get_claim(claim_id) is None:
        raise HTTPException(404, "unknown claim")
    if pattern not in PATTERNS or prompt_style not in PROMPT_STYLES:
        raise HTTPException(422, f"pattern must be one of {PATTERNS}; prompt_style one of {PROMPT_STYLES}")
    try:
        job = jobs.start(claim_id, pattern, prompt_style)
    except AlreadyRunning as exc:
        raise HTTPException(409, str(exc)) from None
    return {"run_id": job.run_id}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str):
    job = jobs.get(run_id)
    if job is None:
        raise HTTPException(404, "unknown or finished run (see /api/claims/{id}/report)")

    async def stream():
        sent = 0
        while True:
            job.changed.clear()
            while sent < len(job.events):
                yield f"data: {json.dumps(job.events[sent])}\n\n"
                sent += 1
            if job.done:
                result = {k: v for k, v in (job.result or {}).items() if k != "blackboard"}
                yield f"event: done\ndata: {json.dumps(result, default=str)}\n\n"
                return
            await job.changed.wait()
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/claims/{claim_id}/report")
def get_report(claim_id: str):
    with db.connect() as conn:
        row = latest(conn, "reports", claim_id)
        if row is None:
            raise HTTPException(404, "no report for this claim yet")
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (row["run_id"],)).fetchone()
        review = latest(conn, "reviews", claim_id)
    return {"run_id": row["run_id"], "report": json.loads(row["report_json"]), "markdown": row["report_md"],
            "run": dict(run) if run else None, "review": dict(review) if review else None,
            "heatmap": f"/api/runs/{row['run_id']}/heatmap.png"}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    with db.connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise HTTPException(404, "unknown run")
        agents = [dict(r) for r in conn.execute("SELECT * FROM agent_metrics WHERE run_id = ? ORDER BY agent", (run_id,))]
        report = conn.execute("SELECT report_json FROM reports WHERE run_id = ?", (run_id,)).fetchone()
    return {"run": dict(run), "agents": agents, "report": json.loads(report["report_json"]) if report else None}


@app.get("/api/runs/{run_id}/heatmap.png")
def run_heatmap(run_id: str):
    png = charts.run_heatmap(run_id)
    if png is None:
        raise HTTPException(404, "no metrics for this run")
    return Response(png, media_type="image/png")


@app.get("/api/runs/{run_id}/trace")
def run_trace(run_id: str):
    with db.connect() as conn:
        run = conn.execute("SELECT trace_id FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None or not run["trace_id"]:
        raise HTTPException(404, "no trace for this run")
    return {"trace_id": run["trace_id"], "spans": tracing.read_trace(run["trace_id"], limit=2000)}


@app.post("/api/claims/{claim_id}/review")
async def review(claim_id: str, request: Request):
    body = await request.json()
    action, note = body.get("action"), (body.get("note") or "").strip()
    if action not in ("approve", "reject"):
        raise HTTPException(422, "action must be approve or reject")
    with db.connect() as conn:
        row = latest(conn, "reports", claim_id)
        if row is None:
            raise HTTPException(409, "there is no report to review")
        report = FinalReport.model_validate_json(row["report_json"])
        if report.review == "full_review" and not note:
            raise HTTPException(422, "this report needs a full review: write a note explaining your decision")
        conn.execute("INSERT INTO reviews (claim_id, run_id, reviewer, action, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                     (claim_id, row["run_id"], "adjuster", action, note or None, db.now()))
        conn.execute("UPDATE claims SET status = ? WHERE id = ?", (f"reviewed:{action}", claim_id))
    audit.AuditLogger(claim_id).log_human_decision(action, note or None)
    return {"claim_id": claim_id, "action": action, "run_id": row["run_id"]}


@app.get("/api/audit/{claim_id}/verify")
def verify_audit(claim_id: str):
    return audit.AuditLogger(claim_id).verify_chain()


# ─── Evaluation ────────────────────────────────────────────────────────────────────────────

@app.get("/api/evaluation")
def evaluation():
    summary = EVAL_RESULTS_DIR / "summary.md"
    charts_list = sorted(p.name for p in EVAL_RESULTS_DIR.glob("*.png")) if EVAL_RESULTS_DIR.is_dir() else []
    return {"summary": summary.read_text(encoding="utf-8") if summary.is_file() else None, "charts": charts_list}


@app.get("/api/evaluation/charts/{name}")
def evaluation_chart(name: str):
    path = EVAL_RESULTS_DIR / Path(name).name
    if path.suffix != ".png" or not path.is_file():
        raise HTTPException(404, "no such chart")
    return FileResponse(path, media_type="image/png")


# The MCP route (/mcp) comes last so the routes above take precedence.
app.mount("/", mcp_endpoint)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=config.PORT)
