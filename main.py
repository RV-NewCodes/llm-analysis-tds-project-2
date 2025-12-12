from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import os
import time
import asyncio
import inspect
import logging

# Import your modules (adjust paths if different)
from agent import run_agent  # existing function in your repo
from shared_store import url_time, BASE64_STORE

# --- Config ---
load_dotenv()
EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
WATCHDOG_SECONDS = int(os.getenv("WATCHDOG_SECONDS", "180"))  # 3 minutes default

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-analysis-main")

# --- App setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME)}


def _clear_shared_state():
    """Clear per-request in-memory stores (safe to call after auth)."""
    try:
        url_time.clear()
    except Exception:
        logger.debug("url_time clear failed", exc_info=True)
    try:
        BASE64_STORE.clear()
    except Exception:
        logger.debug("BASE64_STORE clear failed", exc_info=True)


async def _call_run_agent_with_deadline(url: str, deadline_ts: float):
    """
    Call run_agent(url, deadline_ts?) with support for both sync and async run_agent.
    Returns whatever run_agent returns (or raises).
    """
    sig = inspect.signature(run_agent)
    params = sig.parameters
    # Decide whether to pass deadline_ts
    call_with_deadline = len(params) >= 2

    # Prepare a wrapper to run sync or async function uniformly
    if asyncio.iscoroutinefunction(run_agent):
        if call_with_deadline:
            return await run_agent(url, deadline_ts)
        else:
            return await run_agent(url)
    else:
        # sync function -> run in threadpool
        if call_with_deadline:
            return await asyncio.to_thread(run_agent, url, deadline_ts)
        else:
            return await asyncio.to_thread(run_agent, url)


@app.post("/solve")
async def solve(request: Request):
    received_ts = time.time()

    # 1) Parse JSON safely
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    url = data.get("url")
    provided_secret = data.get("secret")

    if not url or not provided_secret:
        raise HTTPException(status_code=400, detail="Missing required fields: 'url' and 'secret'")

    # 2) Secret validation -> 403 on mismatch
    if SECRET is None:
        # If server not configured with SECRET, treat as misconfiguration and reject.
        logger.error("Server SECRET not configured in environment.")
        raise HTTPException(status_code=500, detail="Server misconfigured (SECRET missing)")

    if provided_secret != SECRET:
        # Don't log the provided_secret
        logger.info("Secret mismatch for incoming request (email not shown here).")
        raise HTTPException(status_code=403, detail="Invalid secret")

    # 3) Auth OK -> clear any prior per-request caches / stores
    _clear_shared_state()

    # 4) Start watchdog and call agent synchronously under timeout
    deadline_ts = time.time() + WATCHDOG_SECONDS
    try:
        # Run agent under asyncio.wait_for to enforce overall timeout.
        agent_result = await asyncio.wait_for(
            _call_run_agent_with_deadline(url=url, deadline_ts=deadline_ts),
            timeout=WATCHDOG_SECONDS,
        )
        # agent_result might be any JSON-serializable value; keep size modest in response.
        response = {
            "status": "ok",
            "url": url,
            "received_at": int(received_ts),
            "completed_at": int(time.time()),
            "agent_result_summary": (
                agent_result if (isinstance(agent_result, (str, int, float, dict, list)) and (len(str(agent_result)) < 8000))
                else {"type": type(agent_result).__name__}
            ),
        }
        logger.info("Completed agent run for url=%s in %.1fs", url, time.time() - received_ts)
        return JSONResponse(status_code=200, content=response)

    except asyncio.TimeoutError:
        # Watchdog fired; we abort the attempt. Secret matched so we respond 200 with timeout info.
        logger.warning("Watchdog timeout after %d seconds for url=%s", WATCHDOG_SECONDS, url)
        response = {
            "status": "timeout",
            "message": f"Processing exceeded {WATCHDOG_SECONDS} seconds and was aborted.",
            "url": url,
            "received_at": int(received_ts),
            "aborted_at": int(time.time()),
        }
        return JSONResponse(status_code=200, content=response)

    except Exception as exc:
        # Any other error — do not leak secrets; return 200 summary (secret matched).
        logger.exception("Agent raised exception for url=%s: %s", url, str(exc))
        response = {
            "status": "agent_error",
            "message": "Agent raised an error during processing.",
            "error": str(exc),
            "url": url,
            "received_at": int(received_ts),
            "failed_at": int(time.time()),
        }
        return JSONResponse(status_code=200, content=response)


if __name__ == "__main__":
    # Host / port can be configured with env vars; defaults kept similar to your original.
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("main:app", host=host, port=port, log_level="info")
