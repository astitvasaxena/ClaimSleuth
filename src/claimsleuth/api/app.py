"""ClaimSleuth web API: the dashboard, the API routes and the MCP server at /mcp."""
import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from claimsleuth import config, db
from claimsleuth.mcp_server import server as mcp_server

NO_KEY_MESSAGE = "No model key configured; investigations are disabled"
SECRET_HEADER = "X-ClaimSleuth-Secret"

class McpEndpoint:
    """Forwards /mcp to the MCP app built for the current app lifespan (its session manager can run only once)."""
    app = None

    async def __call__(self, scope, receive, send):
        if self.app is None:
            raise RuntimeError("MCP endpoint used outside the app lifespan")
        await self.app(scope, receive, send)


mcp_endpoint = McpEndpoint()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    mcp_endpoint.app = mcp_server.http_app()
    async with mcp_server.server.session_manager.run():
        yield
    mcp_endpoint.app = None


app = FastAPI(title="ClaimSleuth", lifespan=lifespan)


def secret_ok(value: str | None) -> bool:
    return bool(config.A2A_SHARED_SECRET) and hmac.compare_digest(value or "", config.A2A_SHARED_SECRET)


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if path == "/mcp" or path.startswith("/mcp/"):
        if not secret_ok(request.headers.get(SECRET_HEADER)):
            return JSONResponse(status_code=401, content={"detail": f"missing or wrong {SECRET_HEADER} header"})
    elif path.startswith("/api/") and not config.model_configured():
        return JSONResponse(status_code=503, content={"detail": NO_KEY_MESSAGE})
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": config.GEMINI_MODEL, "model_configured": config.model_configured()}


@app.get("/api/status")
def status():
    return {"model": config.GEMINI_MODEL}


@app.get("/", response_class=HTMLResponse)
def index():
    if not config.model_configured():
        return f"<h1>ClaimSleuth</h1><p>{NO_KEY_MESSAGE}.</p>"
    return "<h1>ClaimSleuth</h1><p>API running.</p>"


# The MCP route (/mcp) comes last so the routes above take precedence.
app.mount("/", mcp_endpoint)


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
