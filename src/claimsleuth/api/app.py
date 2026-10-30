"""ClaimSleuth web API."""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from claimsleuth import config

NO_KEY_MESSAGE = "No model key configured; investigations are disabled"

app = FastAPI(title="ClaimSleuth")


@app.middleware("http")
async def require_model_key(request: Request, call_next):
    if request.url.path.startswith("/api/") and not config.model_configured():
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


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
