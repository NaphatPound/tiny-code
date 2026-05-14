"""Planner-reviewer: a smarter model that bookends the small executor.

Two phases:

  1. plan(user_request, workspace_context)
     The planner reads what the user asked for and what's in the workspace,
     then emits an ordered list of sub-tasks. Each sub-task is later passed
     to the executor as a single user message.

  2. review(user_request, workspace_state, run_output, action_log)
     After the executor has finished and we've optionally run the project's
     suggested command, the planner inspects the resulting workspace + run
     output and decides: "done" or "fix this specific issue".

Both phases use constrained JSON decoding (via the same Backend abstraction
as the executor) so output is guaranteed to parse.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

from tinycoder.backends.base import Backend
from tinycoder.json_utils import extract_json


# -- Plan phase ------------------------------------------------------------


class PlanStep(BaseModel):
    description: str = Field(
        description="One imperative sentence the executor will follow as its task."
    )
    success_criteria: str = Field(
        default="",
        description="Concrete signal that this step is complete.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_string(cls, value: Any) -> Any:
        # Cloud models sometimes emit each step as a plain string.
        if isinstance(value, str):
            return {"description": value}
        return value


class Plan(BaseModel):
    rationale: str = Field(
        default="",
        description="Two-to-three sentence explanation of the overall approach.",
    )
    steps: list[PlanStep] = Field(
        description="Ordered sub-tasks for the executor. Aim for 2-6 steps."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_alt_shapes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        # Some models wrap everything under "plan" instead of providing
        # rationale+steps at the top level.
        if "steps" not in value and "plan" in value:
            inner = value["plan"]
            if isinstance(inner, list):
                return {"rationale": value.get("rationale", ""), "steps": inner}
            if isinstance(inner, dict):
                merged = {**value, **inner}
                merged.pop("plan", None)
                return merged
        # Some models use "tasks" or "task_list" instead of "steps".
        for alt in ("tasks", "task_list", "actions"):
            if "steps" not in value and alt in value and isinstance(value[alt], list):
                value = {**value, "steps": value[alt]}
                value.pop(alt, None)
                break
        return value


# -- Review phase ----------------------------------------------------------


class ReviewDone(BaseModel):
    type: Literal["done"]
    summary: str = Field(description="One-sentence statement of what was achieved.")


class ReviewFix(BaseModel):
    type: Literal["fix"]
    issue: str = Field(description="One sentence describing what's wrong.")
    instruction: str = Field(
        description="A single imperative sentence telling the executor how to fix it."
    )


ReviewBody = Annotated[
    Union[ReviewDone, ReviewFix], Field(discriminator="type")
]


class ReviewResponse(BaseModel):
    review: ReviewBody

    @model_validator(mode="before")
    @classmethod
    def _unwrap_top_level(cls, value: Any) -> Any:
        # Some models emit the review body at the top level
        # (e.g. {"type":"done","summary":"…"}) instead of wrapping in "review".
        if isinstance(value, dict) and "review" not in value and "type" in value:
            return {"review": value}
        return value


# -- Intervention phase ----------------------------------------------------


class FileWrite(BaseModel):
    """A single file the planner wants to write directly during takeover."""

    path: str = Field(description="Workspace-relative path.")
    content: str = Field(description="Full new file content.")


class GuideIntervention(BaseModel):
    """Cheapest option: re-instruct the executor with sharper guidance."""

    type: Literal["guide"]
    rationale: str = Field(description="One sentence: why guidance is enough.")
    instruction: str = Field(
        description="Detailed, decomposed instruction the executor will follow next."
    )


class TakeoverIntervention(BaseModel):
    """Bypass the executor: planner writes file contents itself."""

    type: Literal["takeover"]
    rationale: str = Field(description="One sentence: why direct takeover.")
    files: list[FileWrite] = Field(
        description="Files to write directly. At least one. Each must be complete."
    )
    post_command: str = Field(
        default="",
        description="Optional verification command to run after writing (e.g. 'npm run dev').",
    )


class ReplanIntervention(BaseModel):
    """Throw out remaining plan, propose a new one."""

    type: Literal["replan"]
    rationale: str
    new_steps: list[PlanStep] = Field(
        description="Replacement steps for the rest of the plan (1-5 ideal)."
    )


class AbortIntervention(BaseModel):
    """Stop. Useful when continuing would waste tokens with no path to success."""

    type: Literal["abort"]
    rationale: str
    summary: str = Field(description="What was achieved (or partly achieved) before stopping.")


InterventionBody = Annotated[
    Union[GuideIntervention, TakeoverIntervention, ReplanIntervention, AbortIntervention],
    Field(discriminator="type"),
]


class InterventionResponse(BaseModel):
    intervention: InterventionBody

    @model_validator(mode="before")
    @classmethod
    def _unwrap_top_level(cls, value: Any) -> Any:
        if isinstance(value, dict) and "intervention" not in value and "type" in value:
            return {"intervention": value}
        return value


# -- Scaffold phase --------------------------------------------------------


class ScaffoldFile(BaseModel):
    """One complete file the big AI writes up-front. Holes are marked with
    `TODO(small-ai): <description>` comments that the executor will later
    replace."""

    path: str = Field(description="Workspace-relative path.")
    content: str = Field(
        description=(
            "Full file content. Place TODO(small-ai): markers where the small "
            "executor should fill in implementation. Imports, exports, "
            "package.json deps, and structural code must already be correct."
        )
    )


class FillTask(BaseModel):
    """One unit of work for the small AI: replace a single TODO marker."""

    path: str = Field(description="File the marker lives in.")
    marker: str = Field(
        description=(
            "Exact text of the TODO marker line, as it appears in the scaffold. "
            "The small AI will edit_file with this as `search`."
        )
    )
    instruction: str = Field(
        description=(
            "One sentence telling the executor what to write in place of the marker."
        )
    )


class ScaffoldPlan(BaseModel):
    rationale: str = Field(
        default="",
        description="Two or three sentences explaining the design.",
    )
    files: list[ScaffoldFile] = Field(
        description="Complete skeleton files to write before the executor runs."
    )
    fill_tasks: list[FillTask] = Field(
        default_factory=list,
        description=(
            "Ordered list of TODO replacements the small AI will perform. "
            "Each task corresponds to exactly one TODO(small-ai) marker."
        ),
    )
    run_command: str = Field(
        default="",
        description=(
            "Command to verify the project runs (e.g. 'npm run dev'). "
            "Leave empty to use auto-detected project run command."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_alt_shapes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "files" not in value and "scaffold" in value:
            inner = value["scaffold"]
            if isinstance(inner, dict):
                merged = {**value, **inner}
                merged.pop("scaffold", None)
                return merged
        return value


# -- System prompts --------------------------------------------------------


PLANNER_SYSTEM = """You are the PLANNER for a coding agent.

Your job: read the user's request and the current workspace, then output an
ordered, minimal plan of sub-tasks that a less-capable executor agent will
carry out. The executor can only do one action per turn (read/write/edit a
file, list files, run a shell command) — so each plan step should be a
concrete, self-contained slice of work.

Rules:
  - 2-6 steps total. Fewer is better. Don't pad.
  - Each step is ONE imperative sentence (≤ 25 words).
  - Order matters: dependencies (install) come before things that need them.
  - If the workspace already contains relevant files, reuse them — don't
    re-create unless the user asked for a rewrite.
  - For framework projects, the suggested run command in [workspace context]
    is authoritative — don't second-guess it.
  - The last step (or final result) should leave the project in a runnable
    state — the system will auto-run it for verification before review.
  - LANGUAGE TAGGING: if the user request mentions TypeScript (or TS, .tsx, .ts)
    OR the workspace context says "+ TypeScript", EVERY step description that
    creates source files MUST be explicit about it:
      BAD:  "Write src/main and src/App with game logic"
      BAD:  "Write react entry files"
      GOOD: "Write src/main.tsx and src/App.tsx (TypeScript .tsx — NOT .jsx)"
      GOOD: "Create package.json with typescript, @types/react, @types/react-dom
             in devDependencies"
    Small executor models default to .jsx from training distribution if the
    step text doesn't actively push them to .tsx.

TypeScript projects — common bugs the small executor introduces (mention these
explicitly in the step's success_criteria so the small AI gets it right):
  - EVERY source file is .ts or .tsx. NEVER .js or .jsx. Two files with the
    same basename in different extensions (App.tsx + App.js) trigger TS5055
    and the build will not pass. If the workspace contains stale .js/.jsx
    files from a previous attempt, the FIRST plan step must delete them
    (via the delete_file action).
  - Import paths NEVER include the extension. Use `import App from './App'`,
    NEVER `import App from './App.tsx'` (TS rejects unless
    allowImportingTsExtensions is on, which it isn't by default).
  - Every function parameter must have an explicit type (no implicit any).
  - useState<T>(initial) — T must match what `initial` actually is. If the
    array can hold null, declare `useState<(string | null)[]>([...nulls])`,
    NOT `useState<string[]>([...nulls])`.
  - No dead code (unused functions / unused imports fail tsc with "declared
    but never read"). Use one canonical helper, not two near-duplicates.
  - Verification command will be `npm run build` (runs tsc), not `npm run dev`.

OUTPUT FORMAT — emit RAW JSON exactly in this shape (no markdown fences):

{
  "rationale": "Two or three sentences explaining the overall approach.",
  "steps": [
    {"description": "Write package.json with vite + react", "success_criteria": "package.json exists with correct deps"},
    {"description": "Write index.html + src/main.jsx + src/App.jsx with full game", "success_criteria": "files exist and App.jsx contains game logic"},
    {"description": "Run npm install", "success_criteria": "[command ok, exit=0]"}
  ]
}

CRITICAL: `steps` MUST be a list of OBJECTS each with `description` and
optionally `success_criteria`. NEVER emit `steps` as a list of plain strings.
"""


INTERVENE_SYSTEM = """You are the INTERVENTION decision-maker for a coding agent.

The small executor is stuck or has failed repeatedly. You receive the user
request, the current workspace state, recent error/failure context, and
how many fix attempts have already happened.

Pick ONE of four actions, optimizing for token efficiency:

  1. "guide"     CHEAP — Re-instruct the executor with sharper, more
                 decomposed steps. Use when the problem is a vague
                 instruction or a missed detail the executor can solve
                 if told precisely.
  2. "takeover"  MEDIUM — Write specific file contents yourself. The
                 executor will be bypassed for these files. Use when the
                 executor keeps producing wrong content for a specific
                 file (e.g. App.jsx logic), or when you can fix the bug
                 directly with one or two file rewrites.
  3. "replan"    EXPENSIVE — Throw out the remaining steps and propose
                 a new plan. Use when the original plan picked a bad
                 approach (wrong framework, wrong tool, wrong order).
  4. "abort"     CHEAP — Stop. Use when no further work is justified
                 (out of scope, ambiguous request, infrastructure missing).

Heuristic: try "guide" first for the initial stuck signal; escalate to
"takeover" if the same file keeps failing; "replan" only if the whole
direction is wrong; "abort" if continuing would waste user tokens.

CRITICAL — takeover content rules:
  - Every takeover file must contain the user's ACTUAL FEATURE in full.
    NEVER write a "Hello, World!" placeholder or a Vite starter template
    if the user asked for a game, a counter, a form, a CRUD app, etc.
    If the user asked for tic-tac-toe, your App.tsx must contain a
    working tic-tac-toe game — not a stub.
  - For TypeScript / TSX projects, every takeover file is REQUIRED to:
      * end with `export default <Name>;` if it defines a React component
        (or be written as `export default function <Name>() { ... }`).
        main.tsx imports `./App` expecting a default export — without it
        the build dies with `"default" is not exported by "src/App.tsx"`.
      * use import paths WITHOUT extensions: `import App from './App'`,
        NEVER `import App from './App.tsx'`. tsc rejects extensioned
        imports unless allowImportingTsExtensions is on (it isn't by
        default).
      * give every function parameter an explicit type (no implicit any).
      * useState<T>(initial) — T must allow every value in `initial`.
  - If stale duplicate files exist (App.js next to App.tsx), include a
    `post_command` that deletes them, e.g. "rm -f src/App.js src/main.js".
    Use `find src -name '*.js' -delete` if you don't know which exactly.
  - When in doubt, write the FULL set of files (App.tsx + main.tsx + the
    stragglers like vite-env.d.ts) as one takeover. The build only passes
    when the WHOLE source tree is consistent — fixing one file at a time
    while leaving stale neighbors is how rescue loops time out.

OUTPUT FORMAT — raw JSON, no fences, no prose. Wrap under "intervention":

  guide:
    {"intervention": {"type":"guide", "rationale":"...", "instruction":"..."}}

  takeover:
    {"intervention": {"type":"takeover", "rationale":"...", "files":[
      {"path":"src/App.jsx", "content":"<FULL FILE CONTENT>"}
    ], "post_command":"npm run dev"}}

  replan:
    {"intervention": {"type":"replan", "rationale":"...", "new_steps":[
      {"description":"step 1", "success_criteria":"..."}
    ]}}

  abort:
    {"intervention": {"type":"abort", "rationale":"...", "summary":"..."}}
"""


SCAFFOLDER_SYSTEM = """You are the SCAFFOLDER for a coding agent.

A small, less-capable executor (often a 1B–3B local LLM) will finish the
project. Your job is to write the FULL skeleton — every file, with correct
imports/exports, correct package.json deps, correct framework structure —
and leave clearly marked holes the small AI will replace.

Why this matters: the small AI is bad at designing a project from scratch
(wrong file paths, missing imports, deprecated tooling) AND bad at writing
multi-line code blocks (wrong indentation, dropped lines, syntax errors).
It is GOOD at replacing ONE TODO line with ONE replacement line.

Hard rules:

  1. Every file must be COMPLETE and PARSEABLE on its own. Imports,
     function signatures, JSX/HTML structure, exports — all correct.

  2. Each TODO hole is a SINGLE LINE the small AI will swap for ≤ 3 lines
     of trivial code (an expression, a return, a print, an event handler).
     NEVER leave a TODO for "implement the whole function" — write the
     function yourself with one TODO inside.

  3. Marker format (one per line, MUST be unique within the file):
       JS/TS/JSX/CSS/Java/C: // TODO(small-ai): <one-sentence what-to-write>
       Python / shell / YAML: # TODO(small-ai): <one-sentence what-to-write>
       HTML:                 <!-- TODO(small-ai): <one-sentence> -->

  4. Surrounding code must already wrap the hole. Examples:

     BAD (hole too big — small AI WILL produce broken code):
         def fizzbuzz(n):
             # TODO(small-ai): implement fizzbuzz from 1 to n

     GOOD (each branch is its own TODO; structure is fixed):
         def fizzbuzz(n):
             for i in range(1, n + 1):
                 if i % 15 == 0:
                     # TODO(small-ai): print 'fizzbuzz'
                     pass
                 elif i % 3 == 0:
                     # TODO(small-ai): print 'fizz'
                     pass
                 elif i % 5 == 0:
                     # TODO(small-ai): print 'buzz'
                     pass
                 else:
                     # TODO(small-ai): print the number i
                     pass

  5. For React+Vite: ALWAYS include package.json, vite.config.js, index.html
     at repo ROOT (not public/), src/main.jsx, src/App.jsx — never miss one.
     package.json must list every imported dependency.

  6. JSX files end in .jsx, not .js.

  6b. For React + Vite + TypeScript:
      - Files end in .tsx (component) and .ts (utility).
      - Include tsconfig.json with strict mode and a build script
        "build": "tsc && vite build". Verification will run `npm run build`,
        which means tsc errors WILL block done.
      - EVERY function parameter must be typed. No implicit any.
      - useState<T>(initial): T must accept everything `initial` contains.
        If the initial is `Array(9).fill(null)` you want
        `useState<(string | null)[]>(Array(9).fill(null))`,
        NOT `useState<string[]>(...)`.
      - Never write two functions that do the same thing — tsc will flag the
        unused one as "declared but never read".

  7. `run_command` MUST be what actually verifies the project:
       - python projects:     "python3 main.py"  (NEVER "python" — some
                              systems don't ship a `python` symlink)
       - node/vite:           "npm run dev"
       - static html:         "python3 -m http.server 8000"
     If the workspace context lists a suggested run command, use it verbatim.

  8. fill_tasks: one entry per TODO marker, in the order the small AI should
     handle them. `marker` MUST be the EXACT line text (including leading
     whitespace) as it appears in your scaffold file content.

OUTPUT FORMAT — raw JSON, no markdown fences, no prose:

{
  "rationale": "Two or three sentences.",
  "files": [
    {"path": "package.json", "content": "{...full json...}"},
    {"path": "src/App.jsx", "content": "...complete file with // TODO(small-ai): ... markers..."}
  ],
  "fill_tasks": [
    {"path": "src/App.jsx",
     "marker": "    // TODO(small-ai): call setCount(count + 1)",
     "instruction": "Replace the marker with: setCount(count + 1);"}
  ],
  "run_command": "npm run dev"
}
"""


REVIEWER_SYSTEM = """You are the REVIEWER for a coding agent.

You receive: the user's original request, the current workspace contents
(file listing + key file bodies), the verification run command output, and
the executor's action log.

Decide ONE of:
  - "done": the request is fully met AND the run succeeded (or no run was
    applicable). Provide a one-sentence summary.
  - "fix": something is missing, broken, or wrong. Pinpoint the SINGLE most
    important issue and write a one-sentence instruction the executor can
    follow on its next pass.

Rules:
  - If the verification run exited non-zero, you must emit "fix" unless the
    failure is unrelated to the user's request.
  - If the main feature file is empty / placeholder / boilerplate, emit "fix".
  - Be specific in the fix instruction — name the file and what to change.
  - Don't request optional polish (better styles, tests, comments) unless
    the user asked for them.

OUTPUT FORMAT — emit RAW JSON exactly in one of these two shapes:

  done case:
    {"review": {"type": "done", "summary": "What was achieved in one sentence."}}

  fix case:
    {"review": {"type": "fix", "issue": "What's wrong.", "instruction": "Imperative sentence telling the executor how to fix it."}}

No markdown fences, no prose, no leading/trailing text.
"""


# -- Planner class ---------------------------------------------------------


class Planner:
    """Plans up-front and reviews afterwards. Stateless across calls."""

    def __init__(self, backend: Backend, temperature: float = 0.2):
        self.backend = backend
        self.temperature = temperature
        self._plan_schema = Plan.model_json_schema()
        self._review_schema = ReviewResponse.model_json_schema()
        self._intervene_schema = InterventionResponse.model_json_schema()
        self._scaffold_schema = ScaffoldPlan.model_json_schema()

    def scaffold(self, user_request: str, workspace_context: str) -> ScaffoldPlan:
        """Produce complete skeleton files + ordered fill tasks."""
        user_msg = (
            f"[workspace context]\n{workspace_context}\n\n"
            f"[user request]\n{user_request}\n\n"
            "Produce the scaffold now. Every file must be complete and parseable. "
            "Place TODO(small-ai): markers for the small AI to fill. Return raw JSON only."
        )
        messages = [
            {"role": "system", "content": SCAFFOLDER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        return self._chat_and_parse(messages, self._scaffold_schema, ScaffoldPlan)

    def plan(self, user_request: str, workspace_context: str) -> Plan:
        user_msg = (
            f"[workspace context]\n{workspace_context}\n\n"
            f"[user request]\n{user_request}\n\n"
            "Produce the plan now. Return raw JSON only — no markdown fences, no prose."
        )
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        return self._chat_and_parse(messages, self._plan_schema, Plan)

    def review(
        self,
        user_request: str,
        workspace_listing: str,
        run_output: str,
        action_log: str,
        key_files: dict[str, str] | None = None,
    ) -> ReviewBody:
        key_files = key_files or {}
        files_block = "\n\n".join(
            f"--- {path} ---\n{content}" for path, content in key_files.items()
        ) or "(no file bodies captured)"

        user_msg = (
            f"[user request]\n{user_request}\n\n"
            f"[workspace listing]\n{workspace_listing}\n\n"
            f"[key files]\n{files_block}\n\n"
            f"[executor action log]\n{action_log}\n\n"
            f"[verification run output]\n{run_output}\n\n"
            "Emit your review now. Return raw JSON only — no markdown fences, no prose."
        )
        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        response = self._chat_and_parse(messages, self._review_schema, ReviewResponse)
        return response.review

    def intervene(
        self,
        user_request: str,
        situation: str,
        recent_errors: str,
        workspace_listing: str,
        key_files: dict[str, str] | None = None,
        fix_attempts: int = 0,
    ) -> InterventionBody:
        """Decide how the big AI should rescue a stuck executor."""
        key_files = key_files or {}
        files_block = "\n\n".join(
            f"--- {path} ---\n{content}" for path, content in key_files.items()
        ) or "(no file bodies captured)"

        user_msg = (
            f"[user request]\n{user_request}\n\n"
            f"[situation]\n{situation}\n\n"
            f"[recent errors / failure signals]\n{recent_errors}\n\n"
            f"[workspace listing]\n{workspace_listing}\n\n"
            f"[key files]\n{files_block}\n\n"
            f"[fix attempts so far]\n{fix_attempts}\n\n"
            "Choose the most token-efficient action. Return raw JSON only."
        )
        messages = [
            {"role": "system", "content": INTERVENE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        response = self._chat_and_parse(messages, self._intervene_schema, InterventionResponse)
        return response.intervention

    def _chat_and_parse(self, messages: list[dict[str, str]], schema: dict, model_cls):
        """Call backend, strip markdown fences, parse — retry once on failure.

        Cloud-routed models often ignore the inference-side JSON schema and
        wrap the body in ```json fences. We extract defensively and, if that
        still doesn't parse, give the model one more chance with an explicit
        reminder.
        """
        raw = self.backend.chat_json(messages, schema, temperature=self.temperature)
        cleaned = extract_json(raw)
        try:
            return model_cls.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError) as first_err:
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not parse as JSON matching the schema. "
                        f"Error: {first_err}. Re-emit the SAME content as RAW JSON only — "
                        "no ```json fences, no prose, no leading/trailing text."
                    ),
                },
            ]
            raw2 = self.backend.chat_json(retry_messages, schema, temperature=self.temperature)
            cleaned2 = extract_json(raw2)
            try:
                return model_cls.model_validate_json(cleaned2)
            except (ValidationError, json.JSONDecodeError) as second_err:
                raise RuntimeError(
                    f"planner output failed schema validation twice. "
                    f"First: {first_err}. Retry: {second_err}. "
                    f"Last raw output: {raw2[:400]}"
                ) from second_err
