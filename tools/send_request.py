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

    It automatically injects your EMAIL and SECRET env vars if they are
    missing or empty in the payload.
    """

    # 1) expand BASE64 placeholder if needed
    ans = payload.get("answer")
    if isinstance(ans, str) and ans.startswith("BASE64_KEY:"):
        key = ans.split(":", 1)[1]
        payload["answer"] = BASE64_STORE[key]

    # 2) ALWAYS ensure email & secret are set from env
    if not payload.get("email"):
        payload["email"] = EMAIL
    if not payload.get("secret"):
        payload["secret"] = SECRET

    headers = headers or {"Content-Type": "application/json"}

    try:
        cur_url = os.getenv("url")
        cache[cur_url] += 1

        # short log (don’t dump full secret)
        sending_log = {
            "answer": str(payload.get("answer"))[:100],
            "email": payload.get("email", ""),
            "url": payload.get("url", ""),
            "has_secret": bool(payload.get("secret")),
        }
        print(f"\nSending Answer \n{json.dumps(sending_log, indent=4)}\n to url: {url}")

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        print("Got the response: \n", json.dumps(data, indent=4), "\n")

        delay = time.time() - url_time.get(cur_url, time.time())
        print(delay)

        next_url = data.get("url")
        if not next_url:
            return "Tasks completed"

        if next_url not in url_time:
            url_time[next_url] = time.time()

        correct = data.get("correct")
        if not correct:
            cur_time = time.time()
            prev = url_time.get(next_url, time.time())
            if (
                cache[cur_url] >= retry_limit
                or delay >= 180
                or (prev != "0" and (cur_time - float(prev)) > 90)
            ):
                print("Not retrying, moving on to the next question")
                data = {"url": data.get("url", "")}
            else:
                os.environ["offset"] = str(url_time.get(next_url, time.time()))
                print("Retrying..")
                data["url"] = cur_url
                data["message"] = "Retry Again!"

        print("Formatted: \n", json.dumps(data, indent=4), "\n")
        forward_url = data.get("url", "")
        os.environ["url"] = forward_url
        if forward_url == next_url:
            os.environ["offset"] = "0"

        return data

    except requests.HTTPError as e:
        try:
            err_data = e.response.json()
        except ValueError:
            err_data = e.response.text
        print("HTTP Error Response:\n", err_data)
        return err_data

    except Exception as e:
        print("Unexpected error:", e)
        return str(e)
