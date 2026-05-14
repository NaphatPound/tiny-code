"""Workspace sandbox — all file operations are scoped to a single root dir."""
from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from tinycoder.edit_utils import find_closest_block, number_lines, smart_replace
from tinycoder.templates import check_completeness, detect_recipe
from tinycoder.validate import validate


# Commands that start a long-running server. If we encounter one, we run it
# briefly, watch for a "ready" signal, then kill the process — never let it
# block the agent loop indefinitely.
_SERVER_PATTERNS: tuple[str, ...] = (
    "npm run dev",
    "npm start",
    "yarn dev",
    "yarn start",
    "pnpm dev",
    "pnpm start",
    "next dev",
    "next start",
    "vite",
    "python -m http.server",
    "python3 -m http.server",
    "uvicorn ",
    "gunicorn ",
    "flask run",
    "manage.py runserver",
    "rails server",
    "rails s ",
    "fastapi dev",
    "serve ",
)


# Substrings that indicate the dev server is up and listening. Match
# case-insensitively. Generous on purpose.
_READY_PATTERNS: tuple[str, ...] = (
    "Local:",
    "ready in",
    "✓ Ready",
    "listening on",
    "running on",
    "Server running",
    "started server",
    "Compiled successfully",
    "You can now view",
    "App listening",
    "Serving Flask",
    "Serving HTTP",
    "Uvicorn running",
    "Started server process",
    "Starting development server",
)


def _is_server_command(command: str) -> bool:
    c = command.lower()
    return any(p.lower() in c for p in _SERVER_PATTERNS)


_COMPOUND_SPLIT = re.compile(r"\s*&&\s*")


def _split_compound(command: str) -> list[str]:
    """Split on top-level `&&`. Naïve but adequate — we don't try to honor
    quoting/escaping perfectly. Returns the original command in a 1-element
    list if no split happened."""
    parts = [p for p in _COMPOUND_SPLIT.split(command.strip()) if p]
    return parts if len(parts) > 1 else [command]


# Pattern -> timeout (seconds). Order matters: first match wins. The list is
# tuned to common dev workflows; override with TINYCODER_CMD_TIMEOUT_DEFAULT.
_TIMEOUT_PATTERNS: tuple[tuple[str, float], ...] = (
    ("npm install", 600.0),
    ("npm i ", 600.0),
    ("npm ci", 600.0),
    ("pnpm install", 600.0),
    ("yarn install", 600.0),
    ("yarn ", 600.0),
    ("pip install", 600.0),
    ("pip3 install", 600.0),
    ("uv sync", 600.0),
    ("uv pip install", 600.0),
    ("poetry install", 600.0),
    ("cargo build", 600.0),
    ("cargo test", 600.0),
    ("go build", 300.0),
    ("go test", 300.0),
    ("npm create", 300.0),
    ("npm init", 300.0),
    ("npm run build", 300.0),
    ("npx create-", 300.0),
    ("docker build", 600.0),
)


def _adaptive_timeout(command: str) -> float:
    """Pick a sensible timeout based on what the command appears to do.

    Networked installs and scaffolding get the longest allowance because
    they're the most common false-negative source. Everything else uses
    the default (controllable via TINYCODER_CMD_TIMEOUT_DEFAULT).
    """
    default = float(os.environ.get("TINYCODER_CMD_TIMEOUT_DEFAULT", "120"))
    cmd = command.lstrip()
    for pattern, value in _TIMEOUT_PATTERNS:
        if pattern in cmd:
            return value
    return default


class WorkspaceError(ValueError):
    """Raised when a path escapes the workspace or another safety check fails."""


_TS_INTENT_PATTERNS = (
    "typescript",
    " ts ",
    " ts.",
    " tsx",
    ".tsx",
    ".ts ",
    "ts project",
    "type script",
    "type-script",
)


def _has_ts_intent(text: str) -> bool:
    """Spot TypeScript intent in free-form user requests.
    Cheap substring scan — false positives are fine, the cost is just an
    extra HARD RULES block in workspace context."""
    if not text:
        return False
    t = " " + text.lower() + " "
    return any(p in t for p in _TS_INTENT_PATTERNS)


class Workspace:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Set externally (e.g. by Agent.run) when the user request makes TS
        # intent obvious. Lets guardrails fire from turn 1, before tsconfig
        # or .tsx files have been written. False by default.
        self.ts_intent: bool = False

    def _resolve(self, rel: str) -> Path:
        if not rel or rel.strip() == "":
            raise WorkspaceError("empty path")
        candidate = (self.root / rel).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise WorkspaceError(
                f"path {rel!r} escapes workspace root {self.root}"
            ) from e
        return candidate

    def write_file(self, rel: str, content: str) -> str:
        # Guardrail: writing empty / whitespace-only content is almost never
        # what the small AI intends — it's usually a misguided attempt to
        # "delete" the file by truncating it. Redirect to delete_file so the
        # next turn does it properly instead of leaving a 0-byte file that
        # still shadows real source files in tools/resolvers.
        if not content.strip():
            return (
                f"[error] refusing to write empty/whitespace-only content to {rel!r}. "
                "If you wanted to remove the file, use the delete_file action. "
                "If you wanted a placeholder, include at least one meaningful line."
            )
        # Guardrail: in a TypeScript project (tsconfig.json present), refuse
        # to write .jsx/.js source files under src/. The small AI defaults to
        # these extensions from training distribution even when the planner
        # says "TypeScript", which then shadows the correct .tsx file via
        # Vite's resolver. Returning an error here puts the right rule
        # directly in the model's next-turn context — no big-AI takeover.
        ts_block = self._refuse_js_in_ts_src(rel)
        if ts_block:
            return ts_block
        p = self._resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        msg = f"wrote {len(content)} bytes to {rel}"
        dropped = self._drop_js_sibling_for_ts(p, rel)
        if dropped:
            msg += f"\n[note] removed stale {dropped} (TS project — sibling .jsx/.js would shadow .tsx/.ts via Vite resolver)"
        err = validate(p, content)
        if err:
            msg += f"\n[validation] {err}"
        # TS-aware package.json dep check: catch missing typescript / @types
        # before the next `npm run build` does (saves an intervention round).
        deps_hint = self._check_ts_pkg_deps(p, rel, content)
        if deps_hint:
            msg += f"\n{deps_hint}"
        framework_hint = self._framework_completeness_hint()
        if framework_hint:
            msg += f"\n{framework_hint}"
        return msg

    def _is_ts_project(self) -> bool:
        """True if ANY signal indicates TypeScript. Broad on purpose: we want
        guardrails to fire from turn 1, not after tsconfig.json finally lands.

        Signals checked (any true → TS project):
          - explicit ts_intent flag set by the runtime from user request
          - tsconfig.json on disk
          - vite.config.ts on disk
          - any .ts or .tsx file anywhere in the workspace
          - `typescript` listed in package.json deps/devDeps
        """
        if self.ts_intent:
            return True
        if (self.root / "tsconfig.json").is_file():
            return True
        if (self.root / "vite.config.ts").is_file():
            return True
        # Cheap scan for any .ts/.tsx file. Limited depth to avoid blowing up
        # on big projects — covers the common src/ pattern.
        try:
            for p in self.root.rglob("*.tsx"):
                if "node_modules" not in p.parts:
                    return True
            for p in self.root.rglob("*.ts"):
                if "node_modules" not in p.parts:
                    return True
        except OSError:
            pass
        pkg_path = self.root / "package.json"
        if pkg_path.is_file():
            try:
                import json as _json
                pkg = _json.loads(pkg_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return False
            all_deps = {
                **(pkg.get("dependencies") or {}),
                **(pkg.get("devDependencies") or {}),
            }
            if "typescript" in all_deps:
                return True
        return False

    def _refuse_js_in_ts_src(self, rel: str) -> str | None:
        """If this is a TS project and the target is src/*.jsx or src/*.js,
        refuse the write with a redirect to the correct extension.

        We only block under src/ — not vite.config.js, eslint.config.js, etc.,
        which legitimately stay as .js even in TS projects.
        """
        if not self._is_ts_project():
            return None
        # Normalize to forward slashes; the workspace contract uses POSIX paths.
        norm = rel.replace("\\", "/")
        if not (norm.startswith("src/") or norm == "src" or "/src/" in norm):
            return None
        lower = norm.lower()
        replacement_ext = None
        if lower.endswith(".jsx"):
            replacement_ext = ".tsx"
        elif lower.endswith(".js"):
            replacement_ext = ".ts"
        if replacement_ext is None:
            return None
        suggested = norm[: -len(Path(norm).suffix)] + replacement_ext
        return (
            f"[error] this is a TypeScript project (tsconfig.json present) — "
            f"refusing to write {rel!r}. Source files under src/ MUST use "
            f".tsx (React component) or .ts (utility), NEVER .jsx / .js. "
            f"Retry with path={suggested!r}. If a stale {rel!r} already "
            "exists from an earlier turn, call delete_file on it first."
        )

    _TS_REQUIRED_DEVDEPS = ("typescript", "@types/react", "@types/react-dom")

    def _check_ts_pkg_deps(self, p: Path, rel: str, content: str) -> str | None:
        """If writing package.json in a TS+React project, hint about missing
        dev-dependencies (typescript / @types/react / @types/react-dom)
        so the small AI fixes the manifest before running `npm run build`."""
        if not self._is_ts_project():
            return None
        if Path(rel).name != "package.json":
            return None
        try:
            import json as _json  # local import; cheap
            pkg = _json.loads(content)
        except (ValueError, TypeError):
            return None
        all_deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        # Only complain when the project is actually React (otherwise @types/react
        # is irrelevant — could be plain ts-node, lib, etc.)
        if "react" not in all_deps:
            return None
        missing = [d for d in self._TS_REQUIRED_DEVDEPS if d not in all_deps]
        if not missing:
            return None
        return (
            f"[hint] TS+React project but package.json is missing devDependencies: "
            f"{', '.join(missing)}. Without these, `npm run build` will fail "
            "with 'tsc: not found' or 'JSX implicitly has type any'. Add them "
            "to devDependencies and rerun install."
        )

    def _drop_js_sibling_for_ts(self, p: Path, rel: str) -> str | None:
        """When writing a .tsx/.ts in a TS project, delete the same-basename
        .jsx/.js sibling so Vite's resolver doesn't pick the wrong file.

        Vite's default resolve.extensions is ['.mjs','.js','.mts','.ts',
        '.jsx','.tsx','.json'] — `.js` and `.jsx` come BEFORE `.tsx`, so an
        `import App from './App'` picks the stale .jsx/.js over the .tsx
        the user actually intends. Only fires if tsconfig.json is present.
        """
        ext = p.suffix.lower()
        ts_to_js = {".tsx": ".jsx", ".ts": ".js"}
        if ext not in ts_to_js:
            return None
        # Only apply in clearly-TS projects. tsconfig.json at workspace root
        # is the conventional signal.
        if not (self.root / "tsconfig.json").is_file():
            return None
        sibling = p.with_suffix(ts_to_js[ext])
        if not sibling.exists() or not sibling.is_file():
            return None
        try:
            sibling.unlink()
        except OSError:
            return None
        # Report relative path so the LLM sees the same path-style it uses.
        try:
            return str(sibling.relative_to(self.root))
        except ValueError:
            return sibling.name

    def read_file(self, rel: str, max_bytes: int = 64_000) -> str:
        p = self._resolve(rel)
        if not p.exists():
            raise WorkspaceError(f"{rel}: not found")
        if not p.is_file():
            raise WorkspaceError(f"{rel}: not a file")
        data = p.read_text(encoding="utf-8", errors="replace")
        truncated = ""
        if len(data) > max_bytes:
            truncated = f"\n... [truncated {len(data) - max_bytes} bytes]"
            data = data[:max_bytes]
        line_count = data.count("\n") + (0 if data.endswith("\n") or data == "" else 1)
        size_hint = "rewrite-friendly" if line_count <= 150 else "use edit_file"
        header = f"[file: {rel}, {line_count} lines, {size_hint}]"
        return f"{header}\n{number_lines(data)}{truncated}"

    def list_files(self, rel: str = ".", max_entries: int = 200) -> str:
        p = self._resolve(rel) if rel not in ("", ".") else self.root
        if not p.exists():
            raise WorkspaceError(f"{rel}: not found")
        if not p.is_dir():
            raise WorkspaceError(f"{rel}: not a directory")
        out: list[str] = []
        for entry in sorted(p.rglob("*")):
            if any(part.startswith(".") for part in entry.relative_to(self.root).parts):
                continue
            rel_path = entry.relative_to(self.root).as_posix()
            kind = "d" if entry.is_dir() else "f"
            out.append(f"{kind} {rel_path}")
            if len(out) >= max_entries:
                out.append(f"... [truncated, more than {max_entries} entries]")
                break
        return "\n".join(out) if out else "(empty)"

    def delete_file(self, rel: str) -> str:
        p = self._resolve(rel)
        if not p.exists():
            return f"[note] {rel}: nothing to delete (file does not exist)"
        if p.is_dir():
            raise WorkspaceError(f"{rel}: refusing to delete a directory")
        p.unlink()
        return f"deleted {rel}"

    def edit_file(self, rel: str, search: str, replace: str) -> str:
        p = self._resolve(rel)
        if not p.exists():
            raise WorkspaceError(f"{rel}: not found")
        text = p.read_text(encoding="utf-8")

        result = smart_replace(text, search, replace)
        if result.new_text is not None:
            p.write_text(result.new_text, encoding="utf-8")
            suffix = "" if result.strategy == "exact" else f" [matched via {result.strategy}]"
            msg = f"edited {rel} ({len(search)} -> {len(replace)} chars){suffix}"
            err = validate(p, result.new_text)
            if err:
                msg += f"\n[validation] {err}"
            framework_hint = self._framework_completeness_hint()
            if framework_hint:
                msg += f"\n{framework_hint}"
            return msg

        if result.strategy == "ambiguous":
            raise WorkspaceError(
                f"{rel}: ambiguous edit — {result.detail}. "
                "Include more surrounding context in `search` to make it unique."
            )

        closest = find_closest_block(text, search)
        hint = (
            f"\nClosest section in file:\n{closest}" if closest else "\n(no similar section found)"
        )
        line_count = text.count("\n") + (0 if text.endswith("\n") or text == "" else 1)
        rewrite_advice = (
            " This file is small — consider write_file with the full new content instead."
            if line_count <= 150
            else ""
        )
        raise WorkspaceError(
            f"{rel}: search block did not match (tried exact, trailing-ws, indent-normalized)."
            f"{rewrite_advice}{hint}"
        )

    def _framework_completeness_hint(self) -> str:
        """If a recognized recipe is detected, report any missing required files.

        In a TypeScript project, rewrite .jsx/.js paths in the suggestion to
        their .tsx/.ts equivalents so the message doesn't actively push the
        small AI back toward the wrong extension (it WILL follow the literal
        path in the hint — observed in real logs).
        """
        recipe = detect_recipe(self.root)
        if recipe is None:
            return ""
        missing = check_completeness(recipe, self.root)
        if not missing:
            return ""
        if self._is_ts_project():
            missing = [self._ts_rename(m) for m in missing]
        return (
            f"[framework: {recipe.name}] still missing required files: "
            f"{', '.join(missing)}. Create them before attempting to run."
        )

    @staticmethod
    def _ts_rename(path: str) -> str:
        """Map .jsx/.js names to canonical .tsx/.ts names for TS projects.

          src/App.jsx        -> src/App.tsx
          src/util.js        -> src/util.ts
          vite.config.js     -> vite.config.ts   (canonical in TS projects)
          others             -> unchanged
        """
        p = Path(path)
        if p.suffix == ".jsx":
            return str(p.with_suffix(".tsx"))
        if p.suffix == ".js":
            norm = str(p).replace("\\", "/")
            if norm.startswith("src/") or "/src/" in norm or norm == "vite.config.js":
                return str(p.with_suffix(".ts"))
        return path

    def run_command(self, command: str, timeout: float | None = None) -> str:
        """Run a shell command in the workspace, with smart handling for
        long-running dev servers and compound `&&` commands.

        Server commands (`npm run dev`, `vite`, `uvicorn`, etc.) would block
        forever — we Popen them, watch stdout for a "ready" signal, then kill
        the process. Compound commands like `rm -rf x && npm install && npm
        run dev` are split on `&&`: prefixes run blocking; the final server
        part runs in brief-watch mode.
        """
        parts = _split_compound(command)
        if len(parts) > 1:
            outputs: list[str] = []
            for idx, part in enumerate(parts):
                is_last = idx == len(parts) - 1
                if is_last and _is_server_command(part):
                    outputs.append(self._run_server_briefly(part))
                    break
                out = self._run_blocking(part, _adaptive_timeout(part) if timeout is None else timeout)
                outputs.append(out)
                if not out.startswith("[command ok"):
                    outputs.append(f"[chain stopped after failure of part {idx + 1}]")
                    break
            return "\n--- next ---\n".join(outputs)

        if _is_server_command(command):
            return self._run_server_briefly(command)
        return self._run_blocking(command, timeout if timeout is not None else _adaptive_timeout(command))

    def _run_blocking(self, command: str, timeout: float) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[command TIMEOUT after {timeout}s] command={command!r}"
        out = result.stdout or ""
        err = result.stderr or ""
        status = "ok" if result.returncode == 0 else "FAILED"
        header = f"[command {status}, exit={result.returncode}] command={command!r}"
        body = f"{header}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
        if len(body) > 8_000:
            body = body[:8_000] + "\n... [truncated]"
        return body

    def _run_server_briefly(self, command: str, watch_seconds: float = 20.0) -> str:
        """Start a dev server in the background, watch for a ready signal,
        kill it after the watch window. Returns a status report for the LLM.
        """
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(self.root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        q: queue.Queue[str] = queue.Queue()

        def _drain(stream, q):
            try:
                for line in iter(stream.readline, ""):
                    q.put(line)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t = threading.Thread(target=_drain, args=(proc.stdout, q), daemon=True)
        t.start()

        deadline = time.monotonic() + watch_seconds
        captured: list[str] = []
        ready_match: str | None = None
        crashed = False

        while time.monotonic() < deadline:
            try:
                line = q.get(timeout=0.25)
                captured.append(line)
                lower = line.lower()
                for pat in _READY_PATTERNS:
                    if pat.lower() in lower:
                        ready_match = pat
                        break
                if ready_match:
                    break
            except queue.Empty:
                pass
            if proc.poll() is not None:
                crashed = True
                break

        # Kill if still running
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Drain any leftover lines
        try:
            while True:
                captured.append(q.get_nowait())
        except queue.Empty:
            pass

        output = "".join(captured)
        if len(output) > 4_000:
            output = output[:4_000] + "\n... [truncated]"

        if ready_match:
            header = (
                f"[command ok, server READY then killed] command={command!r} "
                f"matched={ready_match!r}"
            )
        elif crashed and proc.returncode and proc.returncode != 0:
            header = (
                f"[command FAILED, server crashed exit={proc.returncode}] command={command!r}"
            )
        elif crashed:
            header = f"[command ok, server exited cleanly exit={proc.returncode}] command={command!r}"
        else:
            header = (
                f"[command FAILED, server started but no ready signal in "
                f"{watch_seconds}s; killed] command={command!r}"
            )
        return f"{header}\n--- output ---\n{output}"
