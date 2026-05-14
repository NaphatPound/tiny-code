"""Framework recipes — canonical minimal structure for common project types.

Small LLMs frequently produce "looks-like-React" code that won't actually run:
mixing import styles, missing entry HTML, picking deprecated tooling (CRA),
wrong scripts in package.json, etc. We give them concrete checklists.

Each recipe answers four questions for the LLM:
  1. What files must exist (and what's in them at minimum)?
  2. What's the package.json contract (scripts + deps)?
  3. What's the install command?
  4. What's the run command?

Two outputs:
  - `RECIPES`: human-readable text injected into the system prompt
  - `required_files_for(framework, root)`: programmatic completeness check
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Recipe:
    name: str
    summary: str
    required_files: list[str]
    package_scripts: dict[str, str] = field(default_factory=dict)
    install: str = ""
    run: str = ""
    notes: str = ""


REACT_VITE = Recipe(
    name="react-vite",
    summary="React app via Vite (PREFERRED — fast, light, works on modern Node)",
    required_files=[
        "package.json",
        "index.html",            # at repo root, NOT public/
        "vite.config.js",
        "src/main.jsx",
        "src/App.jsx",
    ],
    package_scripts={"dev": "vite", "build": "vite build", "preview": "vite preview"},
    install="npm install",
    run="npm run dev",
    notes=(
        "index.html sits at repo ROOT (not public/). It must contain "
        "<div id=\"root\"></div> and <script type=\"module\" src=\"/src/main.jsx\"></script>. "
        "vite.config.js must `import { defineConfig } from 'vite'` and "
        "`import react from '@vitejs/plugin-react'`. devDependencies: "
        "vite, @vitejs/plugin-react. dependencies: react, react-dom."
    ),
)

NEXT = Recipe(
    name="next",
    summary="Next.js app (use for SSR/routing needs)",
    required_files=["package.json", "app/page.tsx", "app/layout.tsx"],
    package_scripts={"dev": "next dev", "build": "next build", "start": "next start"},
    install="npm install",
    run="npm run dev",
    notes=(
        "App-router layout: app/layout.tsx must export default function with children. "
        "Dependencies: next, react, react-dom."
    ),
)

VUE_VITE = Recipe(
    name="vue-vite",
    summary="Vue 3 app via Vite",
    required_files=["package.json", "index.html", "vite.config.js", "src/main.js", "src/App.vue"],
    package_scripts={"dev": "vite", "build": "vite build"},
    install="npm install",
    run="npm run dev",
    notes="devDependencies: vite, @vitejs/plugin-vue. dependencies: vue.",
)

STATIC_HTML = Recipe(
    name="static-html",
    summary="Single-file or multi-file static HTML (no build step)",
    required_files=["index.html"],
    install="",
    run="python -m http.server 8000",
    notes="No package.json needed. Inline CSS/JS or link to local files only.",
)

PYTHON_SCRIPT = Recipe(
    name="python-script",
    summary="Single Python script (no package manager)",
    required_files=["main.py"],
    install="",
    run="python main.py  # or whatever the entry file is named",
    notes="If you need third-party packages, create requirements.txt and tell the user to pip install.",
)

PYTHON_FASTAPI = Recipe(
    name="python-fastapi",
    summary="FastAPI web server",
    required_files=["main.py", "requirements.txt"],
    install="pip install -r requirements.txt",
    run="uvicorn main:app --reload",
    notes="requirements.txt must include: fastapi, uvicorn[standard].",
)


ALL_RECIPES = [REACT_VITE, NEXT, VUE_VITE, STATIC_HTML, PYTHON_SCRIPT, PYTHON_FASTAPI]


_PROMPT_RECIPES = ("react-vite", "next", "vue-vite", "static-html")


def render_recipes_for_prompt() -> str:
    """Return a compact, prompt-friendly summary of UI-framework recipes.

    Python recipes are excluded — they're trivial enough to be covered by the
    general rules, and we want to keep the prompt tight for small models.
    """
    out = []
    for r in ALL_RECIPES:
        if r.name not in _PROMPT_RECIPES:
            continue
        body = [f"• {r.name}: files={', '.join(r.required_files)}; run={r.run}"]
        if r.notes:
            # Collapse to a single line but preserve full content for the
            # crucial details (deps list, file location requirements).
            note = " ".join(r.notes.split())
            body.append(f"  - {note}")
        out.append("\n".join(body))
    return "\n".join(out)


def recipe_by_name(name: str) -> Recipe | None:
    for r in ALL_RECIPES:
        if r.name == name:
            return r
    return None


def detect_recipe(root: Path) -> Recipe | None:
    """Infer which recipe the workspace is following (if any)."""
    pkg_path = root / "package.json"
    if pkg_path.exists():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        all_deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        if "next" in all_deps:
            return NEXT
        if "vite" in all_deps:
            if "vue" in all_deps:
                return VUE_VITE
            if "react" in all_deps:
                return REACT_VITE
        if "react" in all_deps and "react-scripts" in all_deps:
            return REACT_VITE  # we'll surface a "migrate to Vite" hint elsewhere
    if (root / "index.html").exists() and not pkg_path.exists():
        return STATIC_HTML
    if (root / "main.py").exists():
        try:
            content = (root / "main.py").read_text(encoding="utf-8", errors="replace")
            if "FastAPI(" in content or "from fastapi" in content:
                return PYTHON_FASTAPI
        except OSError:
            pass
        return PYTHON_SCRIPT
    return None


def check_completeness(recipe: Recipe, root: Path) -> list[str]:
    """Return a list of missing required files."""
    return [f for f in recipe.required_files if not (root / f).exists()]
