"""Agent loop: call LLM → parse action → execute → feed result back → repeat."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import ValidationError

from tinycoder.backends.base import Backend, BackendError
from tinycoder.json_utils import extract_json
from tinycoder.project import ProjectInfo, detect_project
from tinycoder.prompts import FEW_SHOTS, SYSTEM_PROMPT
from tinycoder.schemas import (
    AgentResponse,
    DeleteFile,
    EditFile,
    Finish,
    ListFiles,
    ReadFile,
    RunCommand,
    WriteFile,
    response_json_schema,
)
from tinycoder.workspace import Workspace, WorkspaceError, _has_ts_intent


CommandPolicy = Literal["ask", "yes", "no"]
ApproveDecision = Literal["allow", "deny", "always", "quit"]


@dataclass
class CommandRequest:
    """Bundle of info passed to the approve_command callback."""

    command: str
    reasoning: str
    project: ProjectInfo


ApproveCommand = Callable[[CommandRequest], ApproveDecision]


def _default_approve(_req: CommandRequest) -> ApproveDecision:
    return "deny"


FAILURE_STRIKE_LIMIT = 3

# Markers that mean the previous step failed in a way the supervisor might want
# to act on. Kept in sync with orchestrator._FAILURE_MARKERS.
_STEP_FAILURE_MARKERS = (
    "[command FAILED",
    "[command TIMEOUT",
    "[error]",
    "[validation]",
)


def _is_step_failure(result: str) -> bool:
    return any(marker in result for marker in _STEP_FAILURE_MARKERS)


@dataclass
class StepFailureContext:
    """Info given to on_step_failure when a step result looks like a failure."""

    action_type: str  # e.g. "run_command", "edit_file", "write_file"
    action_payload: dict
    output: str  # the raw failure message
    consecutive_failures: int  # consecutive failing steps in this run
    recent_steps: list["Step"]


@dataclass
class StepFailureResolution:
    """What the failure callback returns."""

    feedback: str  # text appended to the step output (fed back to LLM)
    request_stop: bool = False  # if True, agent stops with stopped_reason


StepFailureCallback = Callable[[StepFailureContext], StepFailureResolution | None]


@dataclass
class Step:
    action_type: str
    reasoning: str
    result: str
    raw_action: dict


@dataclass
class RunResult:
    steps: list[Step] = field(default_factory=list)
    finished: bool = False
    summary: str = ""
    stopped_reason: str = ""


EventListener = Callable[[str, dict], None]


class Agent:
    """Drives the LLM through a task until it emits `finish` or hits a cap."""

    def __init__(
        self,
        backend: Backend,
        workspace: Workspace,
        max_steps: int = 20,
        command_policy: CommandPolicy = "ask",
        approve_command: ApproveCommand | None = None,
        on_event: EventListener | None = None,
        on_step_failure: StepFailureCallback | None = None,
        step_failure_threshold: int = 1,
    ):
        self.backend = backend
        self.workspace = workspace
        self.max_steps = max_steps
        self.command_policy: CommandPolicy = command_policy
        self.approve_command: ApproveCommand = approve_command or _default_approve
        self.on_event = on_event or (lambda _kind, _payload: None)
        self.on_step_failure = on_step_failure
        self.step_failure_threshold = step_failure_threshold
        self._schema = response_json_schema()
        self._history: list[dict[str, str]] = []
        self._quit_requested = False
        self._stop_reason: str | None = None
        self._command_failures: dict[str, int] = {}
        self._consecutive_step_failures = 0
        self._current_result: RunResult | None = None
        # Scaffold/fill-mode guardrails. When fill_mode is on:
        #   - write_file targeting paths the big AI already scaffolded is
        #     rejected (forces the small AI back to edit_file on the marker)
        #   - run_command is rejected (verify happens centrally, not here)
        self.fill_mode: bool = False
        self.fill_protected_paths: set[str] = set()
        # Track last-written content per path so we can detect "small AI just
        # rewrote the same broken file again" — a strong early-exit signal.
        self._last_write: dict[str, str] = {}
        self._reset_history()

    def _reset_history(self) -> None:
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOTS]

    def run(
        self,
        user_request: str,
        max_steps: int | None = None,
        reset_history: bool = False,
    ) -> RunResult:
        # Once we spot TypeScript intent (user said "typescript" / "tsx" / etc.)
        # the flag latches on for the lifetime of the workspace. This lets the
        # workspace's TS guardrails fire from turn 1 — before tsconfig.json or
        # any .tsx file exists — instead of waiting for big-AI rescue.
        if _has_ts_intent(user_request):
            self.workspace.ts_intent = True
        # Orchestrator-driven step execution should pass reset_history=True so
        # the LLM context doesn't carry forward irrelevant prior-step history.
        # Each plan step is independent — its step_prompt is self-contained —
        # so dragging step 1's full action log into step 8's call is pure
        # token waste. Iter-5 XO grew agent history to 30+ turns by step 8.
        if reset_history:
            self._reset_history()
        # Refresh workspace context at the start of each turn so the LLM knows
        # what kind of project it's working in.
        project = detect_project(self.workspace.root, ts_intent=self.workspace.ts_intent)
        context_block = (
            f"[workspace context]\n{project.to_context()}\n\n[user request]\n{user_request}"
        )
        self._history.append({"role": "user", "content": context_block})
        self._command_failures.clear()
        self._consecutive_step_failures = 0
        self._last_write.clear()
        self._stop_reason = None
        result = RunResult()
        self._current_result = result
        effective_max_steps = max_steps if max_steps is not None else self.max_steps

        for step_idx in range(effective_max_steps):
            self.on_event("thinking", {"step": step_idx + 1})
            try:
                raw = self.backend.chat_json(self._history, self._schema)
            except BackendError as e:
                result.stopped_reason = f"backend error: {e}"
                self.on_event("error", {"message": str(e)})
                return result

            self._history.append({"role": "assistant", "content": raw})

            try:
                parsed = AgentResponse.model_validate_json(extract_json(raw))
            except (ValidationError, json.JSONDecodeError) as e:
                # Constrained decoding should make this almost impossible, but
                # if it happens we tell the model and let it retry.
                # Critical: the model often follows up by calling `finish`
                # claiming the previous action succeeded — it did NOT, the
                # JSON didn't parse so nothing executed. Spell that out.
                feedback = (
                    f"[error] your previous output did not match the schema: {e}. "
                    "The action was NOT executed and the workspace is UNCHANGED. "
                    "Do NOT call finish claiming you wrote the file — you did not. "
                    "Emit a single valid JSON object now (typically a retry of the "
                    "same action with corrected JSON escaping)."
                )
                self._history.append({"role": "user", "content": feedback})
                self.on_event("parse_error", {"message": str(e)})
                continue

            action = parsed.action
            self.on_event(
                "action",
                {"type": action.type, "reasoning": action.reasoning, "payload": action.model_dump()},
            )

            exec_result = self._execute(action)

            # Supervisor intervention on failure — give the big model a chance
            # to fix things in-place before the small model keeps thrashing.
            stop_requested = False
            if not isinstance(action, Finish):
                if _is_step_failure(exec_result):
                    self._consecutive_step_failures += 1
                    if (
                        self.on_step_failure
                        and self._consecutive_step_failures >= self.step_failure_threshold
                    ):
                        ctx = StepFailureContext(
                            action_type=action.type,
                            action_payload=action.model_dump(),
                            output=exec_result,
                            consecutive_failures=self._consecutive_step_failures,
                            recent_steps=list(result.steps[-5:]),
                        )
                        try:
                            resolution = self.on_step_failure(ctx)
                        except Exception as e:
                            self.on_event(
                                "error", {"message": f"on_step_failure raised: {e}"}
                            )
                            resolution = None
                        if resolution is not None:
                            if resolution.feedback:
                                exec_result = (
                                    f"{exec_result}\n[supervisor] {resolution.feedback}"
                                )
                            stop_requested = bool(resolution.request_stop)
                            self._consecutive_step_failures = 0
                else:
                    self._consecutive_step_failures = 0

            step = Step(
                action_type=action.type,
                reasoning=action.reasoning,
                result=exec_result,
                raw_action=action.model_dump(),
            )
            result.steps.append(step)
            self.on_event("result", {"type": action.type, "output": exec_result})

            if isinstance(action, Finish):
                result.finished = True
                result.summary = action.summary
                return result

            self._history.append({"role": "user", "content": f"[result] {exec_result}"})

            if stop_requested:
                result.stopped_reason = "supervisor intervention requested stop"
                self.on_event("error", {"message": result.stopped_reason})
                return result

            if self._quit_requested:
                result.stopped_reason = "user quit during command confirmation"
                self.on_event("error", {"message": result.stopped_reason})
                return result

        result.stopped_reason = f"hit max_steps={effective_max_steps}"
        self.on_event("error", {"message": result.stopped_reason})
        return result

    def _execute(self, action) -> str:
        try:
            if isinstance(action, WriteFile):
                if self.fill_mode and action.path in self.fill_protected_paths:
                    return (
                        f"[error] write_file is disabled in FILL mode for {action.path} — "
                        "the scaffold for this file was already written by a smarter model. "
                        "Use edit_file with the TODO(small-ai) marker line as `search` to "
                        "replace just that one line. Read the file first if you need context."
                    )
                if self._last_write.get(action.path) == action.content:
                    return (
                        f"[error] same content was already written to {action.path} on a "
                        "previous turn and failed validation/run. Do not retry the same "
                        "bytes. Either change strategy (use edit_file, fix the broken "
                        "line) or call finish."
                    )
                msg = self.workspace.write_file(action.path, action.content)
                self._last_write[action.path] = action.content
                return msg
            if isinstance(action, ReadFile):
                content = self.workspace.read_file(action.path)
                return f"contents of {action.path}:\n{content}"
            if isinstance(action, ListFiles):
                listing = self.workspace.list_files(action.path)
                return f"contents of {action.path}:\n{listing}"
            if isinstance(action, EditFile):
                return self.workspace.edit_file(action.path, action.search, action.replace)
            if isinstance(action, DeleteFile):
                # Forget the last-written-content cache for this path so a
                # later write of the same bytes is allowed (we wanted it gone).
                self._last_write.pop(action.path, None)
                return self.workspace.delete_file(action.path)
            if isinstance(action, RunCommand):
                if self.fill_mode:
                    return (
                        "[error] run_command is disabled in FILL mode. Verification "
                        "happens centrally after all TODO markers are filled. Just do "
                        "the edit_file and finish."
                    )
                return self._handle_run_command(action)
            if isinstance(action, Finish):
                return action.summary
        except WorkspaceError as e:
            return f"[error] {e}"
        return f"[error] unknown action type: {type(action).__name__}"

    def _handle_run_command(self, action: RunCommand) -> str:
        if self.command_policy == "no":
            return (
                "[error] run_command is disabled by policy. "
                "Re-run with --commands ask or --commands yes if shell access is needed."
            )

        project = detect_project(self.workspace.root)
        request = CommandRequest(
            command=action.command, reasoning=action.reasoning, project=project
        )

        if self.command_policy == "yes":
            return self._execute_command(action.command)

        decision = self.approve_command(request)
        if decision == "allow":
            return self._execute_command(action.command)
        if decision == "always":
            self.command_policy = "yes"
            return self._execute_command(action.command)
        if decision == "quit":
            self._quit_requested = True
            return "[denied] user chose to quit; stop the task."
        return "[denied] user rejected this command. Try a different approach or finish."

    def _execute_command(self, command: str) -> str:
        output = self.workspace.run_command(command)
        success = output.startswith("[command ok,")
        if success:
            self._command_failures.pop(command, None)
            return output

        strikes = self._command_failures.get(command, 0) + 1
        self._command_failures[command] = strikes
        if strikes >= FAILURE_STRIKE_LIMIT:
            output += (
                f"\n[note] this exact command has failed {strikes} times in a row. "
                "Try a fundamentally different approach (different command, different file, "
                "or different strategy), or call finish if you cannot resolve it."
            )
        else:
            output += (
                "\n[hint] command failed. Read stderr above, fix the underlying file with "
                "edit_file or write_file, then retry the command."
            )
        return output
