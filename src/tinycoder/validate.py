"""Per-extension syntax validators.

After every successful write/edit, the workspace runs the file through the
appropriate validator. Errors are reported back to the LLM as part of the
action result so it can self-correct on the next turn. The file is NOT
rolled back — the user can still see the broken output and the LLM can
inspect it via read_file.

External validators (node, bash) are used opportunistically; if the binary
isn't on PATH we silently skip rather than failing the action.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path


def validate(path: Path, content: str) -> str:
    """Return "" if the file content parses cleanly, else a one-line error."""
    ext = path.suffix.lower()

    if ext == ".py":
        try:
            ast.parse(content, filename=str(path))
        except SyntaxError as e:
            return f"python syntax error at line {e.lineno}: {e.msg}"
        return ""

    if ext == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return f"json parse error at line {e.lineno}: {e.msg}"
        return ""

    if ext == ".toml":
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:
            return ""  # Python < 3.11
        try:
            tomllib.loads(content)
        except tomllib.TOMLDecodeError as e:
            return f"toml parse error: {e}"
        return ""

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            return ""
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            first_line = str(e).splitlines()[0] if str(e) else "parse failed"
            return f"yaml parse error: {first_line}"
        return ""

    if ext in (".js", ".mjs", ".cjs"):
        node = shutil.which("node")
        if not node:
            return ""
        return _run_check([node, "--check", str(path)], "javascript")

    if ext == ".sh":
        bash = shutil.which("bash")
        if not bash:
            return ""
        return _run_check([bash, "-n", str(path)], "bash")

    return ""


def _run_check(cmd: list[str], lang: str) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    if result.returncode == 0:
        return ""
    err = (result.stderr or result.stdout or "").strip()
    if not err:
        return f"{lang} syntax error (no detail)"
    last_line = err.splitlines()[-1]
    if len(last_line) > 200:
        last_line = last_line[:200] + " …"
    return f"{lang} syntax error: {last_line}"
