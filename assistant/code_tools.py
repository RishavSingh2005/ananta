import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .tools import ToolError


def coding_example(prompt):
    """Return small, deterministic examples for common beginner requests."""
    normalized = prompt.casefold()
    if ("python" in normalized or "python code" in normalized) and ("addition" in normalized or "add two" in normalized or "sum" in normalized or "add numbers" in normalized):
        return "def add_numbers(first, second):\n    return first + second\n\nprint(add_numbers(2, 3))"
    if "python" in normalized and "hello" in normalized:
        return "print('Hello, world!')"
    if "python" in normalized and "factorial" in normalized:
        return "import math\n\nprint(math.factorial(5))"
    if "sql" in normalized and "insert" in normalized:
        return "INSERT INTO table_name (column1, column2)\nVALUES ('value1', 'value2');"
    return None


LANGUAGE_COMMANDS = {
    "python": lambda path: [os.fspath(Path(os.sys.executable)), "-I", os.fspath(path)],
    "javascript": lambda path: ["node", os.fspath(path)],
    "js": lambda path: ["node", os.fspath(path)],
}


def check_code(language, source):
    language = language.casefold()
    if language == "python":
        import ast
        try:
            ast.parse(source)
        except SyntaxError as error:
            return {"valid": False, "language": language, "error": error.msg, "line": error.lineno}
        return {"valid": True, "language": language, "error": None}
    if language in {"json"}:
        try:
            json.loads(source)
        except json.JSONDecodeError as error:
            return {"valid": False, "language": language, "error": error.msg, "line": error.lineno}
        return {"valid": True, "language": language, "error": None}
    if language in {"javascript", "js"}:
        if shutil.which("node") is None:
            return {"valid": None, "language": language, "error": "Node.js is not installed"}
        return {"valid": True, "language": language, "error": None, "note": "full syntax check occurs during sandboxed run"}
    return {"valid": None, "language": language, "error": "language checker is not installed"}


def run_code(language, source, timeout_seconds=3, max_output=12000):
    if os.getenv("ANANTA_ENABLE_CODE_RUNNER", "0") != "1":
        raise ToolError("code runner is disabled; set ANANTA_ENABLE_CODE_RUNNER=1 for local use")
    language = language.casefold()
    if language not in LANGUAGE_COMMANDS:
        raise ToolError("supported runnable languages are Python and JavaScript when installed")
    if len(source) > 50_000:
        raise ToolError("source is too large")
    with tempfile.TemporaryDirectory(prefix="ananta-run-") as directory:
        extension = ".py" if language == "python" else ".js"
        path = Path(directory) / ("main" + extension)
        path.write_text(source, encoding="utf-8")
        command = LANGUAGE_COMMANDS[language](path)
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        try:
            completed = subprocess.run(
                command,
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=max(1, min(timeout_seconds, 5)),
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ToolError("sandboxed code run failed or timed out") from error
    return {
        "language": language,
        "return_code": completed.returncode,
        "stdout": completed.stdout[:max_output],
        "stderr": completed.stderr[:max_output],
    }