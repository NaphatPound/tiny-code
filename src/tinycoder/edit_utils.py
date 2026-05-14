"""Lenient text-replacement helpers used by Workspace.edit_file.

Small LLMs frequently emit a `search` block whose whitespace doesn't match the
file byte-for-byte (off-by-one indent, missing/extra trailing spaces, tab vs
spaces). These helpers try progressively more permissive strategies before
giving up — and when they do give up, they return the closest-matching window
so the agent can self-correct on its next turn.
"""
from __future__ import annotations

import difflib
import textwrap
from dataclasses import dataclass


@dataclass
class MatchResult:
    new_text: str | None
    strategy: str  # "exact" | "trailing_ws" | "indent_normalized" | "ambiguous" | "no_match"
    detail: str = ""


def _strip_trailing(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.split("\n"))


def _leading_ws(line: str) -> str:
    i = 0
    while i < len(line) and line[i] in (" ", "\t"):
        i += 1
    return line[:i]


def smart_replace(text: str, search: str, replace: str) -> MatchResult:
    """Try exact → trailing-ws-normalized → indent-normalized matching.

    Returns MatchResult with `new_text` set on success, or None plus a reason
    code on failure.
    """
    if not search:
        return MatchResult(None, "no_match", "empty search")

    # 1. Exact match
    count = text.count(search)
    if count == 1:
        return MatchResult(text.replace(search, replace, 1), "exact")
    if count > 1:
        return MatchResult(None, "ambiguous", f"{count} exact matches; make search unique")

    # 2. Trailing-whitespace tolerant
    text_norm = _strip_trailing(text)
    search_norm = _strip_trailing(search)
    nc = text_norm.count(search_norm)
    if nc == 1:
        replaced = text_norm.replace(search_norm, replace, 1)
        return MatchResult(replaced, "trailing_ws")
    if nc > 1:
        return MatchResult(None, "ambiguous", f"{nc} matches after trailing-ws normalize")

    # 3. Indent-normalized: textwrap.dedent both sides, slide a window
    result = _indent_normalized_match(text, search, replace)
    if result is not None:
        return result

    return MatchResult(None, "no_match")


def _indent_normalized_match(text: str, search: str, replace: str) -> MatchResult | None:
    file_lines = text.split("\n")
    search_block = search.rstrip("\n")
    search_lines = search_block.split("\n")
    # Strip empty leading/trailing lines from the search block so we don't pin
    # the window to incidental blank lines.
    while search_lines and search_lines[0].strip() == "":
        search_lines.pop(0)
    while search_lines and search_lines[-1].strip() == "":
        search_lines.pop()
    if not search_lines:
        return None

    search_text = "\n".join(search_lines)
    search_dedented = textwrap.dedent(search_text)

    matches: list[tuple[int, str]] = []
    n = len(search_lines)
    for i in range(len(file_lines) - n + 1):
        window_lines = file_lines[i : i + n]
        window_text = "\n".join(window_lines)
        window_dedented = textwrap.dedent(window_text)
        if window_dedented == search_dedented:
            indent = ""
            for orig, ded in zip(window_lines, window_dedented.split("\n")):
                if ded.strip():
                    indent = orig[: len(orig) - len(ded)]
                    break
            matches.append((i, indent))

    if not matches:
        return None
    if len(matches) > 1:
        return MatchResult(None, "ambiguous", f"{len(matches)} matches after indent normalize")

    i, indent = matches[0]
    replace_block = replace.rstrip("\n")
    replace_dedented = textwrap.dedent(replace_block)
    reindented = "\n".join(
        (indent + line) if line.strip() else line for line in replace_dedented.split("\n")
    )
    new_lines = file_lines[:i] + reindented.split("\n") + file_lines[i + n :]
    return MatchResult("\n".join(new_lines), "indent_normalized")


def find_closest_block(text: str, search: str, context: int = 3) -> str | None:
    """Return a line-numbered slice of `text` containing the closest match.

    Used to give the LLM concrete feedback when an edit fails.
    """
    text_lines = text.split("\n")
    search_lines = search.rstrip("\n").split("\n")
    if not search_lines or not text_lines:
        return None

    # Find the run of consecutive lines in `text` most similar to `search`
    best_score = 0.0
    best_start = 0
    best_len = min(len(search_lines), len(text_lines))
    window = max(1, len(search_lines))
    for i in range(len(text_lines) - window + 1):
        candidate = "\n".join(text_lines[i : i + window])
        score = difflib.SequenceMatcher(
            None, candidate, "\n".join(search_lines), autojunk=False
        ).ratio()
        if score > best_score:
            best_score = score
            best_start = i

    if best_score < 0.2:
        return None

    start = max(0, best_start - context)
    end = min(len(text_lines), best_start + best_len + context)
    return "\n".join(
        f"{i + 1:>4}│ {line}" for i, line in enumerate(text_lines[start:end], start=start)
    )


def number_lines(text: str) -> str:
    """Format `text` with right-aligned line numbers using the U+2502 separator."""
    lines = text.split("\n")
    # If the file ends with a newline, splitlines() would drop the trailing empty
    # entry. Split keeps it, so we get a faithful representation.
    return "\n".join(f"{i + 1:>4}│ {line}" for i, line in enumerate(lines))
