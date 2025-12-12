from langchain_core.tools import tool

@tool
def run_code(code: str) -> str:
    """
    Safe python execution only.
    No subprocess.
    No installs.
    """
    local_env = {}
    try:
        exec(code, {}, local_env)
        return str(local_env)
    except Exception as e:
        return str(e)
