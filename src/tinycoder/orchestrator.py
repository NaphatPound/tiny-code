"""High-level orchestrator: BIG model plans, SMALL model executes.

Flow per user request:
  1. PLAN     — planner emits ordered sub-tasks.
  2. EXECUTE  — for each step, drive the existing executor agent.
                If the executor gets stuck mid-step (doesn't reach `finish`),
                escalate to planner.intervene() right away.
  3. AUTO-RUN — invoke the project's suggested run command for verification.
  4. REVIEW   — planner reads the workspace + run output, decides done/fix.
  5. FIX LOOP — early fix rounds (<= intervention_after) use cheap review→fix.
                Once that limit is crossed, planner.intervene() decides whether
                to guide / takeover / replan / abort to save cloud tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tinycoder.agent import (
    Agent,
    RunResult,
    StepFailureContext,
    StepFailureResolution,
)
from tinycoder.planner import (
    AbortIntervention,
    GuideIntervention,
    InterventionBody,
    Plan,
    Planner,
    PlanStep,
    ReplanIntervention,
    ReviewBody,
    ReviewDone,
    ReviewFix,
    ScaffoldPlan,
    TakeoverIntervention,
)
from tinycoder.project import detect_project
from tinycoder.workspace import Workspace, WorkspaceError


@dataclass
class OrchestrationResult:
    plan: Plan | None = None
    executor_results: list[RunResult] = field(default_factory=list)
    review_rounds: list[dict] = field(default_factory=list)
    interventions: list[dict] = field(default_factory=list)
    finished: bool = False
    summary: str = ""
    stopped_reason: str = ""


_FAILURE_MARKERS = (
    "[command FAILED",
    "[command TIMEOUT",
    "[error]",
    "[validation]",
)

_GAVE_UP_PHRASES = (
    "failed",
    "fail to",
    "cannot proceed",
    "unable to",
    "couldn't",
    "could not",
    "did not succeed",
    "no luck",
    "giving up",
)


def _diagnose_stuck(exec_result, step_description: str) -> str:
    """Return a short reason if executor is effectively stuck, else ''."""
    # Hard signal: did not reach finish at all
    if not exec_result.finished:
        return exec_result.stopped_reason or "executor stopped before finish"

    # Soft signals: finish was called but recent steps show failure
    recent = exec_result.steps[-5:] if exec_result.steps else []
    fail_count = 0
    for s in recent:
        if not s.result:
            continue
        if any(marker in s.result for marker in _FAILURE_MARKERS):
            fail_count += 1
    if fail_count >= 2:
        return f"{fail_count} of the last {len(recent)} actions reported failure"

    # Finish summary admits defeat
    summary = (exec_result.summary or "").lower()
    if any(phrase in summary for phrase in _GAVE_UP_PHRASES):
        return f"executor's finish summary acknowledges failure: {exec_result.summary!r}"

    return ""


KEY_FILE_GLOBS = (
    "package.json",
    "vite.config.js",
    "vite.config.ts",
    "index.html",
    "pyproject.toml",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "src/App.jsx",
    "src/App.tsx",
    "src/main.jsx",
    "src/main.tsx",
    "src/App.vue",
    "src/main.js",
    "app/page.tsx",
    "app/layout.tsx",
    "main.py",
    "app.py",
)


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        executor: Agent,
        max_review_rounds: int = 3,
        intervention_after: int = 2,
        max_interventions: int = 3,
        max_mid_step_interventions: int = 2,
        auto_run: bool = True,
        on_event=None,
    ):
        self.planner = planner
        self.executor = executor
        self.workspace: Workspace = executor.workspace
        self.max_review_rounds = max_review_rounds
        self.intervention_after = intervention_after
        # `max_interventions` is the budget for REVIEW-LOOP rescues (the
        # critical final-verify step the user originally asked us to guarantee).
        # `max_mid_step_interventions` is a separate, smaller budget for in-
        # flight executor-step failures. Splitting them prevents a chatty
        # small AI from draining the budget needed for the build-verify rescue.
        self.max_interventions = max_interventions
        self.max_mid_step_interventions = max_mid_step_interventions
        self.auto_run = auto_run
        self.on_event = on_event or (lambda _kind, _payload: None)
        self._intervention_count = 0  # review-loop budget
        self._mid_step_intervention_count = 0  # mid-step budget
        self._current_user_request: str = ""
        self._current_step_label: str = ""
        self._current_result: OrchestrationResult | None = None
        # Hook the executor: any failed step now escalates to the planner
        # immediately, so the small model gets sharper guidance (or has files
        # rewritten under it) without waiting for the post-run review.
        self.executor.on_step_failure = self._on_executor_step_failure

    # -- public --------------------------------------------------------

    def run(self, user_request: str) -> OrchestrationResult:
        result = OrchestrationResult()
        self._intervention_count = 0
        self._mid_step_intervention_count = 0
        self._current_user_request = user_request
        self._current_step_label = ""
        # Stash the result so the in-flight failure callback can record
        # interventions on the same OrchestrationResult.
        self._current_result = result

        # Latch TS-intent from the ORIGINAL user request — step prompts
        # generated by the planner often omit the word "typescript", so we
        # can't rely on the executor catching it from those.
        from tinycoder.workspace import _has_ts_intent  # local import
        if _has_ts_intent(user_request):
            self.workspace.ts_intent = True

        # 1. PLAN
        project = detect_project(self.workspace.root, ts_intent=self.workspace.ts_intent)
        self.on_event("planning", {"workspace": project.to_context()})
        try:
            plan = self.planner.plan(user_request, project.to_context())
        except Exception as e:
            result.stopped_reason = f"planner failed: {e}"
            self.on_event("error", {"message": result.stopped_reason})
            return result
        result.plan = plan
        self.on_event(
            "plan",
            {"rationale": plan.rationale, "steps": [s.model_dump() for s in plan.steps]},
        )

        # 2. EXECUTE — index-based loop so replan can mutate plan.steps in place
        i = 0
        while i < len(plan.steps):
            step = plan.steps[i]
            self.on_event(
                "step_start",
                {"index": i + 1, "total": len(plan.steps), "description": step.description},
            )
            self._current_step_label = (
                f"step {i + 1}/{len(plan.steps)}: {step.description}"
            )
            step_prompt = (
                f"Task {i + 1} of {len(plan.steps)}: {step.description}\n"
                f"Success criteria: {step.success_criteria or '(none specified)'}\n"
                "\n"
                "STEP DISCIPLINE — read carefully:\n"
                "  - Do ONLY this step's task. Even if other files look needed,\n"
                "    do NOT write them now — they belong to other steps.\n"
                "  - Call finish AS SOON AS this step's named deliverable exists.\n"
                "    Don't try to satisfy the whole project here.\n"
                "  - 'still missing required files' hints from earlier writes\n"
                "    are for the OVERALL plan, not for this step. Ignore unless\n"
                "    the named file is part of THIS step's deliverable.\n"
                "  - If the step says 'Write X', you should typically need 1\n"
                "    action (write_file X) then finish. Two if you read first."
            )
            exec_result = self.executor.run(step_prompt)
            result.executor_results.append(exec_result)
            self.on_event(
                "step_end",
                {"index": i + 1, "finished": exec_result.finished, "summary": exec_result.summary},
            )

            if exec_result.stopped_reason and "quit" in exec_result.stopped_reason:
                result.stopped_reason = exec_result.stopped_reason
                return result

            # Mid-step intervention triggers when:
            #   - executor didn't reach `finish` (hit max_steps / error), OR
            #   - executor called `finish` but recent activity shows it gave up
            #     (failed commands, error markers, "failed" in summary).
            stuck_reason = _diagnose_stuck(exec_result, step.description)
            if stuck_reason:
                decision = self._invoke_intervention(
                    user_request=user_request,
                    situation=(
                        f"executor did not really complete step {i + 1}/{len(plan.steps)}: "
                        f"'{step.description}'. Signal: {stuck_reason}."
                    ),
                    fix_attempts=self._intervention_count,
                    result=result,
                )
                outcome = self._apply_intervention(decision, plan, i, result)
                if outcome == "abort":
                    return result
                if outcome == "replan":
                    # plan.steps was mutated; restart from current index
                    continue
                if outcome == "retry":
                    continue  # re-execute same step index
                # guide / takeover: move on to next step
            i += 1

        # 3-5. AUTO-RUN + REVIEW + FIX/INTERVENTION LOOP
        fix_round = 0
        while True:
            run_output = self._auto_run() if self.auto_run else "(auto-run disabled)"
            listing = self.workspace.list_files(".")
            key_files = self._gather_key_files()
            action_log = self._summarize_action_log(result.executor_results)

            self.on_event(
                "reviewing",
                {"round": fix_round + 1, "max": self.max_review_rounds + 1},
            )
            try:
                review = self.planner.review(
                    user_request=user_request,
                    workspace_listing=listing,
                    run_output=run_output,
                    action_log=action_log,
                    key_files=key_files,
                )
            except Exception as e:
                result.stopped_reason = f"reviewer failed: {e}"
                self.on_event("error", {"message": result.stopped_reason})
                return result

            review_record = {"round": fix_round + 1, "review": review.model_dump()}
            result.review_rounds.append(review_record)
            self.on_event("review", review_record)

            if isinstance(review, ReviewDone):
                result.finished = True
                result.summary = review.summary
                return result

            # ReviewFix path: either run a cheap fix or escalate to intervention
            if fix_round >= self.max_review_rounds:
                result.stopped_reason = (
                    f"review still requesting fixes after {self.max_review_rounds} rounds"
                )
                return result

            fix: ReviewFix = review  # type: ignore[assignment]

            if fix_round >= self.intervention_after:
                # Escalate — small AI is stuck on the same kind of issue.
                decision = self._invoke_intervention(
                    user_request=user_request,
                    situation=(
                        f"reviewer requested fix in round {fix_round + 1}: "
                        f"issue='{fix.issue}'. Previous rounds did not resolve it."
                    ),
                    fix_attempts=fix_round,
                    result=result,
                )
                outcome = self._apply_intervention(decision, None, None, result)
                if outcome == "abort":
                    return result
                # Otherwise (guide/takeover/replan) keep looping — next review will check.
            else:
                # Cheap path: hand the fix instruction to the small executor.
                self.on_event("fixing", {"issue": fix.issue, "instruction": fix.instruction})
                self._current_step_label = f"review-fix round {fix_round + 1}: {fix.issue}"
                fix_prompt = (
                    f"The reviewer found an issue: {fix.issue}\n"
                    f"Fix instruction: {fix.instruction}"
                )
                exec_result = self.executor.run(fix_prompt)
                result.executor_results.append(exec_result)
                if exec_result.stopped_reason and "quit" in exec_result.stopped_reason:
                    result.stopped_reason = exec_result.stopped_reason
                    return result

            fix_round += 1

    # -- mid-step failure hook ----------------------------------------

    def _on_executor_step_failure(
        self, ctx: StepFailureContext
    ) -> StepFailureResolution | None:
        """Fired by the executor when a step result looks like a failure.

        Calls the big model for an intervention (guide / takeover / replan /
        abort) and returns feedback the small model will see on its next turn.
        Uses the SEPARATE mid-step budget — drained from a chatty small AI
        early in the run, the final-verify rescue still has its own budget.
        """
        if self._current_result is None:
            return None
        if self._mid_step_intervention_count >= self.max_mid_step_interventions:
            self.on_event(
                "intervention_skipped",
                {"reason": (
                    f"mid-step budget exhausted "
                    f"({self._mid_step_intervention_count}/"
                    f"{self.max_mid_step_interventions}); "
                    "saving review budget for final verify"
                )},
            )
            return None

        situation = (
            f"executor failed mid-step ({self._current_step_label or 'no step label'}). "
            f"action={ctx.action_type}, consecutive_failures={ctx.consecutive_failures}. "
            f"output_preview={ctx.output[:400]!r}"
        )
        decision = self._invoke_intervention(
            user_request=self._current_user_request,
            situation=situation,
            fix_attempts=ctx.consecutive_failures,
            result=self._current_result,
            kind="mid_step",
        )
        if decision is None:
            return None

        if isinstance(decision, AbortIntervention):
            return StepFailureResolution(
                feedback=f"planner aborted: {decision.rationale}. {decision.summary}",
                request_stop=True,
            )
        if isinstance(decision, GuideIntervention):
            return StepFailureResolution(
                feedback=(
                    f"planner guidance: {decision.instruction} "
                    f"(why: {decision.rationale})"
                )
            )
        if isinstance(decision, TakeoverIntervention):
            written: list[str] = []
            errors: list[str] = []
            for fw in decision.files:
                try:
                    self.workspace.write_file(fw.path, fw.content)
                    written.append(fw.path)
                    self.on_event(
                        "takeover_write", {"path": fw.path, "result": "ok"}
                    )
                except WorkspaceError as e:
                    errors.append(f"{fw.path}: {e}")
                    self.on_event(
                        "takeover_error", {"path": fw.path, "error": str(e)}
                    )
            parts = [f"planner takeover: {decision.rationale}"]
            if written:
                parts.append(f"already wrote {', '.join(written)} for you")
            if errors:
                parts.append(f"failed to write: {'; '.join(errors)}")
            if decision.post_command:
                parts.append(f"next, run: {decision.post_command}")
            else:
                parts.append("continue from here")
            return StepFailureResolution(feedback=". ".join(parts))
        if isinstance(decision, ReplanIntervention):
            # Mid-step replan is awkward (we're inside the executor); degrade
            # to a guide so the small model at least gets the first new step.
            first = decision.new_steps[0] if decision.new_steps else None
            instruction = (
                first.description if first else "rethink your approach"
            )
            return StepFailureResolution(
                feedback=(
                    f"planner replanned: {decision.rationale}. "
                    f"next step: {instruction}"
                )
            )
        return None

    # -- intervention helpers -----------------------------------------

    def _invoke_intervention(
        self,
        user_request: str,
        situation: str,
        fix_attempts: int,
        result: OrchestrationResult,
        kind: str = "review",
    ) -> InterventionBody | None:
        if kind == "mid_step":
            self._mid_step_intervention_count += 1
            attempt = self._mid_step_intervention_count
            cap = self.max_mid_step_interventions
        else:
            if self._intervention_count >= self.max_interventions:
                self.on_event(
                    "intervention_skipped",
                    {"reason": f"already used {self._intervention_count} review interventions"},
                )
                return None
            self._intervention_count += 1
            attempt = self._intervention_count
            cap = self.max_interventions
        self.on_event(
            "intervening",
            {"attempt": attempt, "max": cap, "situation": situation, "kind": kind},
        )
        try:
            decision = self.planner.intervene(
                user_request=user_request,
                situation=situation,
                recent_errors=self._recent_errors(result),
                workspace_listing=self.workspace.list_files("."),
                key_files=self._gather_key_files(),
                fix_attempts=fix_attempts,
            )
        except Exception as e:
            self.on_event("error", {"message": f"intervene failed: {e}"})
            return None
        result.interventions.append({"attempt": attempt, "kind": kind, "decision": decision.model_dump()})
        self.on_event("intervention", decision.model_dump())
        return decision

    def _apply_intervention(
        self,
        decision: InterventionBody | None,
        plan: Plan | None,
        step_index: int | None,
        result: OrchestrationResult,
    ) -> str:
        """Return one of: 'guide', 'takeover', 'replan', 'abort', 'retry', 'noop'."""
        if decision is None:
            return "noop"
        if isinstance(decision, AbortIntervention):
            result.stopped_reason = f"intervention: abort — {decision.rationale}"
            result.summary = decision.summary
            return "abort"
        if isinstance(decision, GuideIntervention):
            # Hand the sharper instruction back to the executor and try again.
            exec_result = self.executor.run(
                f"[planner guidance] {decision.instruction}\n"
                f"Why: {decision.rationale}"
            )
            result.executor_results.append(exec_result)
            return "guide"
        if isinstance(decision, TakeoverIntervention):
            # Big AI writes the files directly. Bypass executor for this slice.
            written: list[str] = []
            for fw in decision.files:
                try:
                    msg = self.workspace.write_file(fw.path, fw.content)
                    written.append(f"  ✓ {fw.path}")
                    self.on_event("takeover_write", {"path": fw.path, "result": msg})
                except WorkspaceError as e:
                    written.append(f"  ✗ {fw.path}: {e}")
                    self.on_event("takeover_error", {"path": fw.path, "error": str(e)})
            if decision.post_command:
                from tinycoder.schemas import RunCommand
                self.on_event("takeover_run", {"command": decision.post_command})
                self.executor._handle_run_command(  # noqa: SLF001
                    RunCommand(
                        type="run_command",
                        reasoning="planner takeover verification",
                        command=decision.post_command,
                    )
                )
            return "takeover"
        if isinstance(decision, ReplanIntervention):
            if plan is not None and step_index is not None:
                plan.steps[step_index:] = list(decision.new_steps)
            elif plan is not None:
                plan.steps = list(decision.new_steps)
            return "replan"
        return "noop"

    # -- existing helpers ---------------------------------------------

    def _auto_run(self) -> str:
        project = detect_project(self.workspace.root, ts_intent=self.workspace.ts_intent)
        if not project.run_commands:
            return "(no suggested run command for this project type)"
        command = project.run_commands[0]
        # Strip any inline comment so the command is actually runnable
        # (project.run_commands sometimes includes "# then open …" hints).
        if "#" in command:
            command = command.split("#", 1)[0].strip()
        self.on_event("auto_run", {"command": command})
        from tinycoder.schemas import RunCommand
        return self.executor._handle_run_command(  # noqa: SLF001
            RunCommand(type="run_command", reasoning="auto-verification run", command=command)
        )

    def _gather_key_files(self, max_bytes_per_file: int = 8_000) -> dict[str, str]:
        out: dict[str, str] = {}
        for rel in KEY_FILE_GLOBS:
            path = self.workspace.root / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) > max_bytes_per_file:
                text = text[:max_bytes_per_file] + f"\n... [truncated {len(text) - max_bytes_per_file} bytes]"
            out[rel] = text
        return out

    def _summarize_action_log(self, executor_results) -> str:
        lines: list[str] = []
        for i, er in enumerate(executor_results, start=1):
            lines.append(f"-- executor pass {i} --")
            for j, step in enumerate(er.steps, start=1):
                short = step.result.splitlines()[0] if step.result else ""
                if len(short) > 160:
                    short = short[:160] + " …"
                lines.append(f"  {j}. {step.action_type}: {short}")
            if er.summary:
                lines.append(f"  → finish: {er.summary}")
            elif er.stopped_reason:
                lines.append(f"  → stopped: {er.stopped_reason}")
        return "\n".join(lines) if lines else "(no executor activity)"

    def _recent_errors(self, result: OrchestrationResult, max_chars: int = 3000) -> str:
        """Pull recent error-looking lines from the executor action log."""
        chunks: list[str] = []
        for er in result.executor_results[-3:]:
            for step in er.steps[-6:]:
                if not step.result:
                    continue
                if any(
                    marker in step.result
                    for marker in ("[error]", "[validation]", "[command FAILED", "[hint]", "[note]")
                ):
                    chunks.append(f"{step.action_type}: {step.result[:600]}")
            if er.stopped_reason:
                chunks.append(f"(executor stopped: {er.stopped_reason})")
        joined = "\n".join(chunks)
        if len(joined) > max_chars:
            joined = joined[:max_chars] + "\n... [truncated]"
        return joined or "(no obvious error markers in recent executor log)"


# -- Scaffold-fill-verify mode --------------------------------------------


_NO_RUN_CMD = "(no run command)"


_FILL_PROMPT_TEMPLATE = """You are in FILL mode.

The project skeleton has already been written by a smarter model. Every file
already has correct imports, exports, structure, and package.json deps. Your
ONLY job is to replace exactly ONE TODO marker with implementation code.

File:          {path}
Marker line:   {marker}
What to write: {instruction}

How to do this, in order:
  1. read_file({path}) so you can see the context around the marker.
  2. edit_file with:
       path:    {path}
       search:  {marker}
       replace: <your implementation — typically 1–10 lines>
     The `search` field MUST be the marker line EXACTLY as shown.
  3. finish with a one-sentence summary.

Rules:
  - DO NOT restructure other code or add imports — the scaffold already has them.
  - DO NOT run commands. Verification happens later.
  - DO NOT touch any other file or any other TODO.
  - If you cannot make the edit, finish with a one-sentence explanation."""


class ScaffoldOrchestrator:
    """Scaffold-fill-verify flow.

    Flow per user request:
      1. SCAFFOLD — big AI writes EVERY file complete and parseable, with
                    TODO(small-ai): markers where logic goes.
      2. WRITE    — orchestrator drops the scaffold files into the workspace
                    directly (no executor round-trip).
      3. FILL     — for each TODO, hand a tightly-scoped task to the small
                    executor: "replace this exact line with this implementation".
      4. VERIFY   — run the project's run command.
      5. RESCUE   — on failure, big AI inspects + takes over via the
                    existing intervention path (guide / takeover / abort).

    Differs from Orchestrator: the scaffold means the small AI never has to
    design files from scratch — it only fills in holes.
    """

    def __init__(
        self,
        planner: Planner,
        executor: Agent,
        max_review_rounds: int = 2,
        max_interventions: int = 3,
        fill_max_steps: int = 5,
        on_event=None,
    ):
        self.planner = planner
        self.executor = executor
        self.workspace: Workspace = executor.workspace
        self.max_review_rounds = max_review_rounds
        self.max_interventions = max_interventions
        # Per-fill budget — small AI should need 3 turns (read, edit, finish).
        # 5 leaves headroom for one self-correction. Anything more and we want
        # to bail and let the central verify/takeover loop rescue the project,
        # rather than burning tokens on a confused small model.
        self.fill_max_steps = fill_max_steps
        self.on_event = on_event or (lambda _kind, _payload: None)
        self._intervention_count = 0
        # Reuse the helpers from the regular Orchestrator without inheritance.
        self._helper = Orchestrator(
            planner=planner,
            executor=executor,
            max_review_rounds=max_review_rounds,
            max_interventions=max_interventions,
            on_event=lambda _k, _p: None,  # suppress duplicate events
        )

    def run(self, user_request: str) -> OrchestrationResult:
        result = OrchestrationResult()
        self._intervention_count = 0

        from tinycoder.workspace import _has_ts_intent  # local import
        if _has_ts_intent(user_request):
            self.workspace.ts_intent = True

        # 1. SCAFFOLD
        project = detect_project(self.workspace.root, ts_intent=self.workspace.ts_intent)
        self.on_event("scaffolding", {"workspace": project.to_context()})
        try:
            scaffold = self.planner.scaffold(user_request, project.to_context())
        except Exception as e:
            result.stopped_reason = f"scaffolder failed: {e}"
            self.on_event("error", {"message": result.stopped_reason})
            return result

        self.on_event(
            "scaffold",
            {
                "rationale": scaffold.rationale,
                "files": [f.path for f in scaffold.files],
                "fill_tasks": [t.model_dump() for t in scaffold.fill_tasks],
                "run_command": scaffold.run_command,
            },
        )

        # 2. WRITE SCAFFOLD
        scaffold_paths: set[str] = set()
        for fw in scaffold.files:
            try:
                self.workspace.write_file(fw.path, fw.content)
                scaffold_paths.add(fw.path)
                self.on_event("scaffold_write", {"path": fw.path})
            except WorkspaceError as e:
                self.on_event(
                    "scaffold_error", {"path": fw.path, "error": str(e)}
                )

        # 3. FILL — small executor handles one marker at a time.
        # Guardrails: fill_mode rejects write_file on scaffold paths + rejects
        # run_command; per-task step budget keeps a confused small model from
        # burning tokens (the central verify/takeover loop will rescue any
        # unfilled holes anyway).
        previous_fill_mode = self.executor.fill_mode
        previous_protected = self.executor.fill_protected_paths
        self.executor.fill_mode = True
        self.executor.fill_protected_paths = scaffold_paths
        try:
            for i, task in enumerate(scaffold.fill_tasks):
                self.on_event(
                    "fill_start",
                    {
                        "index": i + 1,
                        "total": len(scaffold.fill_tasks),
                        "path": task.path,
                        "marker": task.marker,
                        "instruction": task.instruction,
                    },
                )
                fill_prompt = _FILL_PROMPT_TEMPLATE.format(
                    path=task.path,
                    marker=task.marker,
                    instruction=task.instruction,
                )
                exec_result = self.executor.run(
                    fill_prompt, max_steps=self.fill_max_steps
                )
                result.executor_results.append(exec_result)
                self.on_event(
                    "fill_end",
                    {
                        "index": i + 1,
                        "finished": exec_result.finished,
                        "summary": exec_result.summary,
                    },
                )
                if exec_result.stopped_reason and "quit" in exec_result.stopped_reason:
                    result.stopped_reason = exec_result.stopped_reason
                    return result
        finally:
            self.executor.fill_mode = previous_fill_mode
            self.executor.fill_protected_paths = previous_protected

        # 4. VERIFY + 5. RESCUE LOOP
        run_command = scaffold.run_command or self._default_run_command()
        run_output = self._verify(run_command) if run_command else _NO_RUN_CMD

        review_round = 0
        while True:
            listing = self.workspace.list_files(".")
            key_files = self._helper._gather_key_files()  # noqa: SLF001
            action_log = self._helper._summarize_action_log(result.executor_results)  # noqa: SLF001

            self.on_event(
                "reviewing",
                {"round": review_round + 1, "max": self.max_review_rounds + 1},
            )
            try:
                review = self.planner.review(
                    user_request=user_request,
                    workspace_listing=listing,
                    run_output=run_output,
                    action_log=action_log,
                    key_files=key_files,
                )
            except Exception as e:
                result.stopped_reason = f"reviewer failed: {e}"
                self.on_event("error", {"message": result.stopped_reason})
                return result

            review_record = {"round": review_round + 1, "review": review.model_dump()}
            result.review_rounds.append(review_record)
            self.on_event("review", review_record)

            if isinstance(review, ReviewDone):
                result.finished = True
                result.summary = review.summary
                return result

            fix: ReviewFix = review  # type: ignore[assignment]

            if review_round >= self.max_review_rounds:
                result.stopped_reason = (
                    f"still failing after {self.max_review_rounds + 1} review rounds"
                )
                return result

            if self._intervention_count >= self.max_interventions:
                result.stopped_reason = "exhausted intervention budget"
                return result

            # Big AI takeover — this is the user-requested "final step":
            # planner reads everything and patches the project itself.
            self._intervention_count += 1
            self.on_event(
                "intervening",
                {
                    "attempt": self._intervention_count,
                    "max": self.max_interventions,
                    "situation": fix.issue,
                },
            )
            try:
                decision = self.planner.intervene(
                    user_request=user_request,
                    situation=(
                        f"verification failed in scaffold mode. "
                        f"Reviewer says: issue='{fix.issue}', "
                        f"instruction='{fix.instruction}'."
                    ),
                    recent_errors=(run_output or "")[:3000],
                    workspace_listing=listing,
                    key_files=key_files,
                    fix_attempts=review_round,
                )
            except Exception as e:
                result.stopped_reason = f"intervene failed: {e}"
                self.on_event("error", {"message": result.stopped_reason})
                return result

            result.interventions.append(
                {"attempt": self._intervention_count, "decision": decision.model_dump()}
            )
            self.on_event("intervention", decision.model_dump())

            run_output = self._apply_decision(decision, run_command, result)
            if isinstance(decision, AbortIntervention):
                return result
            review_round += 1

    # -- helpers ------------------------------------------------------

    def _apply_decision(
        self,
        decision: InterventionBody,
        default_run_command: str,
        result: OrchestrationResult,
    ) -> str:
        """Apply an intervention. Returns new run_output for the next review."""
        if isinstance(decision, AbortIntervention):
            result.stopped_reason = f"intervention: abort — {decision.rationale}"
            result.summary = decision.summary
            return "(aborted by planner)"

        if isinstance(decision, TakeoverIntervention):
            for fw in decision.files:
                try:
                    self.workspace.write_file(fw.path, fw.content)
                    self.on_event("takeover_write", {"path": fw.path})
                except WorkspaceError as e:
                    self.on_event(
                        "takeover_error", {"path": fw.path, "error": str(e)}
                    )
            cmd = decision.post_command or default_run_command
            return self._verify(cmd) if cmd else _NO_RUN_CMD

        if isinstance(decision, GuideIntervention):
            exec_result = self.executor.run(
                f"[planner guidance] {decision.instruction}\nWhy: {decision.rationale}"
            )
            result.executor_results.append(exec_result)
            return self._verify(default_run_command) if default_run_command else _NO_RUN_CMD

        if isinstance(decision, ReplanIntervention):
            # Best-effort: drive the small AI through the new first step.
            if decision.new_steps:
                first = decision.new_steps[0]
                exec_result = self.executor.run(
                    f"[planner replan] {first.description}\n"
                    f"Success: {first.success_criteria or '(none)'}"
                )
                result.executor_results.append(exec_result)
            return self._verify(default_run_command) if default_run_command else _NO_RUN_CMD

        return "(no-op intervention)"

    def _default_run_command(self) -> str:
        project = detect_project(self.workspace.root, ts_intent=self.workspace.ts_intent)
        if not project.run_commands:
            return ""
        cmd = project.run_commands[0]
        if "#" in cmd:
            cmd = cmd.split("#", 1)[0].strip()
        return cmd

    def _verify(self, command: str) -> str:
        if not command:
            return _NO_RUN_CMD
        from tinycoder.schemas import RunCommand

        self.on_event("verify", {"command": command})
        return self.executor._handle_run_command(  # noqa: SLF001
            RunCommand(
                type="run_command",
                reasoning="scaffold verification",
                command=command,
            )
        )
