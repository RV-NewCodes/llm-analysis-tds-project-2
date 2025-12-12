from langchain_core.tools import tool
import requests

@tool
def get_rendered_html(url: str) -> dict:
    """
    Fetch raw HTML only. No JS, no browser.
    """
    print("\nFetching:", url)
    try:
        r = requests.get(url, timeout=10)
        text = r.text
        if len(text) > 200_000:
            text = text[:200_000] + "...[TRUNCATED]"
        return {
            "html": text,
            "url": url
        }
    except Exception as e:
        return {"error": str(e)}
