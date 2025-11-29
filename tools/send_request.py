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

    IMPORTANT: This tool ALWAYS injects the correct email & secret
    from environment variables and NEVER trusts the LLM for identity fields.

    Args:
        url (str): The SUBMIT endpoint to send the POST request to.
        payload (Dict[str, Any]): The JSON-serializable request body.
            LLM may fill in 'answer' and 'url' (quiz URL), but 'email'
            and 'secret' are always overridden from env.
        headers (Optional[Dict[str, str]]): Optional HTTP headers.

    Returns:
        Any: Parsed JSON response if available, else raw text or error.
    """
    # Ensure headers
    headers = headers or {"Content-Type": "application/json"}

    # Ensure we always send our own identity, not LLM's
    payload = dict(payload)  # copy to avoid mutating original
    payload["email"] = EMAIL
    payload["secret"] = SECRET

    # If quiz URL is missing, fall back to current env url
    if not payload.get("url"):
        payload["url"] = os.getenv("url", "")

    # Handle BASE64 placeholder
    ans = payload.get("answer")
    if isinstance(ans, str) and ans.startswith("BASE64_KEY:"):
        key = ans.split(":", 1)[1]
        if key in BASE64_STORE:
            payload["answer"] = BASE64_STORE[key]

    try:
        cur_url = os.getenv("url", "")
        cache[cur_url] += 1

        # Short preview for logs (don’t spam full base64 or huge answers)
        sending_preview = {
            "answer": str(payload.get("answer", ""))[:100],
            "email": payload.get("email", ""),
            "url": payload.get("url", "")
        }
        print(f"\nSending Answer \n{json.dumps(sending_preview, indent=4)}\n to url: {url}")

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        # Try to parse JSON
        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text}

        print("Got the response: \n", json.dumps(data, indent=4), "\n")

        # Timing / retry logic
        now = time.time()
        start_time_for_cur = url_time.get(cur_url, now)
        delay = now - start_time_for_cur
        print(delay)

        next_url = data.get("url")
        if not next_url:
            # No next URL => quiz likely over
            return data

        # Initialize timing for new URL
        if next_url not in url_time:
            url_time[next_url] = time.time()

        correct = data.get("correct")

        if not correct:
            # Decide whether to retry or move on
            cur_time = time.time()
            prev = url_time.get(next_url, cur_time)
            too_many_retries = cache[cur_url] >= retry_limit
            too_slow = delay >= 180
            too_long_on_next = (cur_time - float(prev)) > 90

            if too_many_retries or too_slow or too_long_on_next:
                print("Not retrying, moving on to the next question")
                # Force move to whatever next URL server gave
                data = {"url": data.get("url", "")}
            else:
                # Retry current quiz
                os.environ["offset"] = str(url_time.get(next_url, cur_time))
                print("Retrying on same question..")
                data["url"] = cur_url
                data["message"] = "Retry Again!"

        print("Formatted: \n", json.dumps(data, indent=4), "\n")

        forward_url = data.get("url", "")
        os.environ["url"] = forward_url
        if forward_url == next_url:
            os.environ["offset"] = "0"

        return data

    except requests.HTTPError as e:
        resp = e.response
        try:
            err_data = resp.json()
        except ValueError:
            err_data = resp.text

        print("HTTP Error Response:\n", err_data)
        return err_data

    except Exception as e:
        print("Unexpected error:", e)
        return str(e)
