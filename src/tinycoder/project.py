"""Project type detection.

Before letting the LLM execute commands, we sniff the workspace for project
indicators (package.json, Cargo.toml, pyproject.toml, etc.) and report:

  - the detected project type
  - likely entry points
  - a suggested install command (if dependencies aren't fetched yet)
  - one or more suggested run commands

The detector never guesses if no indicators are present — it returns type
"unknown" so the agent's prompt can tell the LLM to inspect first.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path


TS_PROJECT_RULES: tuple[str, ...] = (
    "ALL source files in src/ MUST be .tsx (React components) or .ts (utilities). "
    "Writing src/*.jsx or src/*.js will be REFUSED by the workspace.",
    "Import paths in .ts/.tsx must NEVER include the extension. Write "
    "`import App from './App'`, never `import App from './App.tsx'`.",
    "Every .tsx React component file must end with `export default <Name>;` "
    "(or `export default function <Name>() {...}`).",
    "package.json MUST list `typescript`, `@types/react`, `@types/react-dom` in "
    "devDependencies — otherwise `npm run build` fails with tsc errors.",
    "Verification command is `npm run build` (runs tsc), not `npm run dev` — "
    "dev boots even when tsc would fail.",
)

# Canonical recent versions to stop the small AI from writing vite@^2 etc
# from older training distributions.
REACT_VITE_TS_VERSIONS: tuple[str, ...] = (
    "RECOMMENDED dependency versions (use these — small models often write "
    "outdated versions from training data):",
    "  dependencies:    react ^18.2.0, react-dom ^18.2.0",
    "  devDependencies: vite ^5.0.0, @vitejs/plugin-react ^4.2.0, "
    "typescript ^5.3.0, @types/react ^18.2.0, @types/react-dom ^18.2.0",
)


def _python_cmd() -> str:
    """Return the python executable to embed in suggested run commands.

    Many systems (macOS, modern Linux distros) ship `python3` but not `python`.
    Pick the first one that's actually on PATH so verification runs don't fail
    with 'python: command not found'.
    """
    for candidate in ("python3", "python"):
        if shutil.which(candidate):
            return candidate
    return "python3"  # safe default — closer to current convention


@dataclass
class ProjectInfo:
    type: str = "unknown"
    entry_points: list[str] = field(default_factory=list)
    install_command: str | None = None
    run_commands: list[str] = field(default_factory=list)
    notes: str = ""
    # Hard rules the model must obey for this project (extension policies,
    # required deps, etc). Surfaced under a HARD RULES section so the small
    # AI sees them at the start of every turn — not buried in the system
    # prompt where it may forget mid-task.
    rules: list[str] = field(default_factory=list)

    def to_context(self) -> str:
        lines = [f"project type: {self.type}"]
        if self.entry_points:
            lines.append(f"likely entry: {', '.join(self.entry_points)}")
        if self.install_command:
            lines.append(f"install: {self.install_command}")
        if self.run_commands:
            lines.append(f"suggested run: {' OR '.join(self.run_commands)}")
        if self.notes:
            lines.append(f"notes: {self.notes}")
        if self.rules:
            lines.append("HARD RULES (workspace will reject violations):")
            lines.extend(f"  - {r}" for r in self.rules)
        return "\n".join(lines)


def detect_project(root: Path, ts_intent: bool = False) -> ProjectInfo:
    """Detect project shape from files on disk.

    `ts_intent`: when True, force TypeScript rules even if no .tsconfig / .tsx
    file exists yet. The caller (typically Agent.run) sets this from the user
    request ("create xo game with TypeScript") so guardrails kick in on the
    very first .jsx write attempt — not after big AI scaffolds tsconfig.json.
    """
    info = _detect_project_impl(root, ts_intent=ts_intent)
    # Last-mile: ensure ts_intent always produces TS rules + label, even when
    # we couldn't classify the project (e.g. empty workspace before turn 1).
    if ts_intent and not any("TypeScript" in (info.type or "") for _ in (0,)):
        info.type = (info.type + " + TypeScript") if info.type and info.type != "unknown" else "TypeScript-intent (no files yet)"
    if ts_intent and not info.rules:
        info.rules = list(TS_PROJECT_RULES) + list(REACT_VITE_TS_VERSIONS)
    return info


def _detect_project_impl(root: Path, ts_intent: bool = False) -> ProjectInfo:
    try:
        entries = list(root.iterdir())
    except OSError:
        return ProjectInfo("unknown", notes="workspace not accessible")
    files_top = {p.name for p in entries if p.is_file()}
    dirs_top = {p.name for p in entries if p.is_dir()}

    # Order matters — most specific indicators first.

    if "Cargo.toml" in files_top:
        return ProjectInfo(
            "rust",
            entry_points=["src/main.rs", "src/lib.rs"],
            run_commands=["cargo run"],
            install_command=None if "target" in dirs_top else "cargo build",
        )

    if "go.mod" in files_top:
        return ProjectInfo("go", run_commands=["go run ."])

    if "package.json" in files_top:
        return _node_info(root, dirs_top, ts_intent=ts_intent)

    if "pyproject.toml" in files_top:
        return _pyproject_info(root, files_top, dirs_top)

    if "requirements.txt" in files_top:
        return _python_loose_info(
            root, files_top, install_cmd="pip install -r requirements.txt"
        )

    if "Pipfile" in files_top:
        return _python_loose_info(root, files_top, install_cmd="pipenv install")

    # Bare-Python fallback (no manifest)
    py = _python_cmd()
    py_files = sorted(f for f in files_top if f.endswith(".py"))
    if py_files:
        entry = _pick(py_files, ("main.py", "app.py", "run.py", "server.py", "__main__.py"))
        return ProjectInfo("python", entry_points=[entry], run_commands=[f"{py} {entry}"])

    html_files = sorted(f for f in files_top if f.endswith(".html"))
    if html_files:
        entry = _pick(html_files, ("index.html",))
        return ProjectInfo(
            "html",
            entry_points=[entry],
            run_commands=[
                f"{py} -m http.server 8000  # then open http://localhost:8000/{entry}",
                f"open {entry}",
            ],
            notes="static HTML — open in a browser or serve over http",
        )

    sh_files = sorted(f for f in files_top if f.endswith(".sh"))
    if sh_files:
        entry = _pick(sh_files, ("run.sh", "start.sh", "main.sh"))
        return ProjectInfo("shell", entry_points=[entry], run_commands=[f"bash {entry}"])

    if "Makefile" in files_top:
        return ProjectInfo("make", run_commands=["make", "make run"])

    if not files_top and not dirs_top:
        return ProjectInfo("unknown", notes="workspace is empty — create files first")

    found = sorted(files_top)[:8]
    return ProjectInfo(
        "unknown",
        notes=f"no known manifest. Files at root: {', '.join(found)}",
    )


def _pick(candidates: list[str], priority: tuple[str, ...]) -> str:
    for name in priority:
        if name in candidates:
            return name
    return candidates[0]


def _node_info(root: Path, dirs_top: set[str], ts_intent: bool = False) -> ProjectInfo:
    try:
        pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ProjectInfo("node", run_commands=["npm start"], notes="package.json unreadable")

    scripts = pkg.get("scripts", {}) or {}
    main = pkg.get("main") or "index.js"
    all_deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

    is_ts = (
        ts_intent
        or "typescript" in all_deps
        or (root / "tsconfig.json").exists()
        or "tsc" in " ".join(scripts.values())
        or any((root / "src").glob("**/*.tsx")) if (root / "src").exists() else False
    )

    # Sub-type detection for clearer hints
    sub_type = "node"
    notes = ""
    if "next" in all_deps:
        sub_type = "node (Next.js)"
    elif "vite" in all_deps:
        if "vue" in all_deps:
            sub_type = "node (Vue + Vite)"
        elif "react" in all_deps:
            sub_type = "node (React + Vite)"
        else:
            sub_type = "node (Vite)"
    elif "react-scripts" in all_deps:
        sub_type = "node (React + CRA — DEPRECATED)"
        notes = (
            "react-scripts/CRA is deprecated and may fail on Node 17+. "
            "If start fails, set NODE_OPTIONS=--openssl-legacy-provider, or migrate to Vite."
        )
    if is_ts:
        sub_type += " + TypeScript"

    runs: list[str] = []
    # For TypeScript projects, run `npm run build` FIRST so verification catches
    # type errors. `npm run dev` boots even when tsc would fail — making it
    # alone an unreliable signal that the project "works".
    if is_ts and "build" in scripts:
        runs.append("npm run build")
        notes = (
            (notes + " " if notes else "")
            + "TypeScript: prefer 'npm run build' for verification — "
            "'npm run dev' boots even when tsc would fail."
        ).strip()
    # Prefer dev for Vite/Next-style projects, start otherwise
    if "dev" in scripts and "npm run dev" not in runs:
        runs.append("npm run dev")
    if "start" in scripts and "npm start" not in runs:
        runs.append("npm start")
    if not runs:
        runs.append(f"node {main}")

    install = None if "node_modules" in dirs_top else "npm install"
    rules: list[str] = []
    if is_ts:
        rules.extend(TS_PROJECT_RULES)
        if "react" in all_deps or "react" in (pkg.get("dependencies") or {}) or ts_intent:
            rules.extend(REACT_VITE_TS_VERSIONS)
    return ProjectInfo(
        sub_type,
        entry_points=[main],
        install_command=install,
        run_commands=runs,
        notes=notes,
        rules=rules,
    )


def _pyproject_info(root: Path, files_top: set[str], dirs_top: set[str]) -> ProjectInfo:
    data: dict = {}
    try:
        import tomllib  # type: ignore[import-not-found]

        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except Exception:
        pass

    runs: list[str] = []
    install: str | None = None

    if "uv.lock" in files_top:
        install = "uv sync"
        runs.append("uv run python -m <package>")
    elif "poetry.lock" in files_top:
        install = "poetry install"
        runs.append("poetry run python -m <package>")
    elif ".venv" not in dirs_top and "venv" not in dirs_top:
        install = "pip install -e ."

    project_section = data.get("project", {}) or {}
    scripts = (project_section.get("scripts") or {}) if isinstance(project_section, dict) else {}
    if isinstance(scripts, dict) and scripts:
        first_script = next(iter(scripts))
        runs.insert(0, first_script)

    py = _python_cmd()
    py_files = sorted(f for f in files_top if f.endswith(".py"))
    entry_points: list[str] = []
    if py_files:
        entry = _pick(py_files, ("main.py", "app.py", "run.py", "server.py", "__main__.py"))
        entry_points = [entry]
        if not scripts:
            runs.append(f"{py} {entry}")

    if not runs:
        runs.append(f"{py} -m <module>")

    return ProjectInfo("python", entry_points=entry_points, install_command=install, run_commands=runs)


def _python_loose_info(root: Path, files_top: set[str], install_cmd: str) -> ProjectInfo:
    py = _python_cmd()
    py_files = sorted(f for f in files_top if f.endswith(".py"))
    if py_files:
        entry = _pick(py_files, ("main.py", "app.py", "run.py", "server.py"))
        return ProjectInfo(
            "python",
            entry_points=[entry],
            install_command=install_cmd,
            run_commands=[f"{py} {entry}"],
        )
    return ProjectInfo(
        "python",
        install_command=install_cmd,
        notes="manifest present but no .py file at root yet",
    )
