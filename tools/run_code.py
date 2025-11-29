import subprocess
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

def strip_code_fences(code: str) -> str:
    code = code.strip()
    # Remove ```python ... ``` or ``` ... ```
    if code.startswith("```"):
        # remove first line (```python or ```)
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()

@tool
def run_code(code: str) -> dict:
    """
    Executes Python code.

    This tool:
      1. Takes in python code as input
      2. Writes code into a temporary .py file
      3. Executes the file with `uv run`
      4. Returns its output

    Returns
    -------
    dict
        {
            "stdout": <program output>,
            "stderr": <errors if any>,
            "return_code": <exit code>
        }
    """
    try:
        # Clean up ```python fences if the LLM included them
        code = strip_code_fences(code)

        filename = "runner.py"
        os.makedirs("LLMFiles", exist_ok=True)
        file_path = os.path.join("LLMFiles", filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        proc = subprocess.Popen(
            ["uv", "run", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="LLMFiles"
        )
        stdout, stderr = proc.communicate()

        # Truncate long outputs but keep the dict structure
        if len(stdout) > 10000:
            stdout = stdout[:10000] + "...truncated due to large size"
        if len(stderr) > 10000:
            stderr = stderr[:10000] + "...truncated due to large size"

        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": proc.returncode
        }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }
