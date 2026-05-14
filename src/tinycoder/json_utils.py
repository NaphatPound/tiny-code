"""Robust extraction of JSON from model output.

Some models — especially cloud-routed ones — ignore the inference-time JSON
schema constraint and wrap their output in markdown code fences or pad it
with prose. We strip the noise here so the downstream Pydantic parser sees
clean JSON.

Order of operations:
  1. Strip whitespace.
  2. Strip leading ```json / ``` and trailing ``` fences.
  3. If still not valid JSON, isolate the substring between the first `{`
     (or `[`) and the matching closing bracket.
"""
from __future__ import annotations

import json
import re


_FENCE_OPEN = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?")
_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def extract_json(text: str) -> str:
    """Return a candidate JSON string from a possibly-noisy model response."""
    s = text.strip()

    # Strip an opening ```json / ```jsonc / ``` fence and a trailing ```
    m = _FENCE_OPEN.match(s)
    if m:
        s = s[m.end():]
    s = _FENCE_CLOSE.sub("", s).strip()

    # Fast path: already parses
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass

    # Find the first { or [ and its matching close via bracket counting that
    # respects string literals (so braces inside JSON strings don't confuse us).
    start = _first_open_bracket(s)
    if start == -1:
        return s
    end = _matching_close(s, start)
    if end == -1:
        return s
    return s[start : end + 1]


def _first_open_bracket(s: str) -> int:
    for i, ch in enumerate(s):
        if ch in "{[":
            return i
    return -1


def _matching_close(s: str, start: int) -> int:
    """Return the index of the bracket that matches s[start], or -1."""
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1
