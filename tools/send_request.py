from langchain_core.tools import tool
from shared_store import BASE64_STORE, url_time
import time
import os
import requests
import json
from collections import defaultdict
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL", "").strip()
SECRET = os.getenv("SECRET", "").strip()

cache = defaultdict(int)
retry_limit = 4


@tool
def post_request(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Any:
    """
    Send an HTTP POST request to the given URL with the provided payload.

    It automatically injects EMAIL and SECRET from env vars if missing.
    It also propagates the next URL so the agent can continue.
    """

    # -------------------------------------------------
    # 1) Expand BASE64 placeholder if needed
    # -------------------------------------------------
    ans = payload.get("answer")
    if isinstance(ans, str) and ans.startswith("BASE64_KEY:"):
        key = ans.split(":", 1)[1]
        payload["answer"] = BASE64_STORE.get(key, "")

    # -------------------------------------------------
    # 2) Ensure email & secret
    # -------------------------------------------------
    if not payload.get("email"):
        payload["email"] = EMAIL
    if not payload.get("secret"):
        payload["secret"] = SECRET

    headers = headers or {"Content-Type": "application/json"}

    try:
        cur_url = os.getenv("url", "")
        cache[cur_url] += 1

        # -------------------------------------------------
        # 3) Log (safe, no secrets)
        # -------------------------------------------------
        sending_log = {
            "answer": str(payload.get("answer"))[:100],
            "email": payload.get("email", ""),
            "url": payload.get("url", ""),
            "has_secret": bool(payload.get("secret")),
        }
        print(
            f"\nSending Answer\n{json.dumps(sending_log, indent=4)}\n→ POST {url}"
        )

        # -------------------------------------------------
        # 4) POST request
        # -------------------------------------------------
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()

        data = response.json()
        print("Got response:\n", json.dumps(data, indent=4), "\n")

        # -------------------------------------------------
        # 5) Timing bookkeeping
        # -------------------------------------------------
        delay = time.time() - url_time.get(cur_url, time.time())
        print("Delay:", delay)

        # -------------------------------------------------
        # 6) Handle next URL (THIS IS THE IMPORTANT FIX)
        # -------------------------------------------------
        next_url = data.get("url")
        if not next_url:
            return "Tasks completed"

        # Store timing for next task
        url_time[next_url] = time.time()

        # 🔴 CRITICAL: propagate next URL globally
        os.environ["url"] = next_url

        print("Formatted:\n", json.dumps(data, indent=4), "\n")
        return data

    except requests.HTTPError as e:
        try:
            err_data = e.response.json()
        except Exception:
            err_data = e.response.text
        print("HTTP Error Response:\n", err_data)
        return err_data

    except Exception as e:
        print("Unexpected error:", e)
        return str(e)
