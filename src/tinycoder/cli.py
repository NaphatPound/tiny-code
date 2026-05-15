from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from tinycoder.agent import Agent, ApproveDecision, CommandPolicy, CommandRequest
from tinycoder.backends import build_backend
from tinycoder.orchestrator import Orchestrator, ScaffoldOrchestrator
from tinycoder.planner import Planner
from tinycoder.session_log import SessionLogger
from tinycoder.workspace import Workspace

console = Console()

# Set by _run_once / _repl while a request is in flight so the event
# renderers also feed the on-disk session log. None when logging is off.
_current_session_logger: SessionLogger | None = None


def _render_event(kind: str, payload: dict) -> None:
    if _current_session_logger is not None:
        _current_session_logger.record(kind, payload)
    if kind == "thinking":
        console.print(f"[dim]· step {payload['step']} thinking…[/dim]")
        return
    if kind == "action":
        atype = payload["type"]
        reasoning = payload["reasoning"]
        body = payload["payload"]
        console.print(f"[bold cyan]→ {atype}[/bold cyan]  [dim]{reasoning}[/dim]")
        if atype == "write_file":
            console.print(
                Panel(
                    Syntax(body["content"], _guess_lang(body["path"]), line_numbers=False),
                    title=f"write_file {body['path']}",
                    border_style="cyan",
                )
            )
        elif atype == "edit_file":
            console.print(f"  [dim]path:[/dim] {body['path']}")
            console.print(f"  [red]- {body['search']!r}[/red]")
            console.print(f"  [green]+ {body['replace']!r}[/green]")
        elif atype == "read_file":
            console.print(f"  [dim]path:[/dim] {body['path']}")
        elif atype == "list_files":
            console.print(f"  [dim]path:[/dim] {body['path']}")
        elif atype == "delete_file":
            console.print(f"  [red]rm[/red] {body['path']}")
        elif atype == "run_command":
            console.print(f"  [yellow]$ {body['command']}[/yellow]")
        return
    if kind == "result":
        output = payload["output"]
        if output.startswith("[error]"):
            console.print(f"  [red]{output}[/red]")
        else:
            short = output if len(output) < 200 else output[:200] + " …"
            console.print(f"  [dim green]✓ {short}[/dim green]")
        return
    if kind == "parse_error":
        console.print(f"[yellow]· schema retry: {payload['message']}[/yellow]")
        return
    if kind == "error":
        console.print(f"[red]· {payload['message']}[/red]")


def _console_approve(req: CommandRequest) -> ApproveDecision:
    """Interactive y/N/a/q confirmation prompt for run_command.

    Shows the chosen command alongside the project type + suggested commands
    so the user can spot mismatches before approving.
    """
    suggested = (
        " OR ".join(req.project.run_commands) if req.project.run_commands else "(none)"
    )
    mismatch = req.project.run_commands and req.command not in req.project.run_commands
    mismatch_note = (
        "\n[red]⚠ chosen command differs from the detected suggestion[/red]"
        if mismatch
        else ""
    )
    body = (
        f"[bold]project:[/bold]   {req.project.type}\n"
        f"[bold]suggested:[/bold] {suggested}\n"
        f"[bold]chosen:[/bold]    [yellow]$ {req.command}[/yellow]\n"
        f"[dim]why:[/dim]      {req.reasoning}"
        f"{mismatch_note}"
    )
    console.print(Panel(body, title="run command?", border_style="yellow"))
    while True:
        try:
            choice = console.input(
                "[bold yellow]▶ [y]es / [N]o / [a]lways / [q]uit: [/bold yellow]"
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return "quit"
        if choice in ("", "n", "no"):
            return "deny"
        if choice in ("y", "yes"):
            return "allow"
        if choice in ("a", "always"):
            return "always"
        if choice in ("q", "quit", "exit"):
            return "quit"
        console.print("[dim](please type y, n, a, or q)[/dim]")


def _guess_lang(path: str) -> str:
    ext = Path(path).suffix.lstrip(".")
    return {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "tsx": "tsx",
        "jsx": "jsx",
        "rs": "rust",
        "go": "go",
        "rb": "ruby",
        "sh": "bash",
        "md": "markdown",
        "json": "json",
        "yml": "yaml",
        "yaml": "yaml",
        "toml": "toml",
        "html": "html",
        "css": "css",
        "sql": "sql",
    }.get(ext, "text")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tinycoder",
        description="Terminal coding assistant for small local LLMs.",
    )
    p.add_argument(
        "--backend",
        default=os.environ.get("TINYCODER_BACKEND", "ollama"),
        choices=["ollama", "llamacpp", "openai"],
        help="LLM backend to use (default: ollama).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("TINYCODER_MODEL", "qwen2.5-coder:1.5b"),
        help="Model name (Ollama tag, llama.cpp loaded model, or OpenAI-compat id).",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("TINYCODER_BASE_URL"),
        help="Override the backend base URL.",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("TINYCODER_API_KEY"),
        help="API key for openai-compat backends (optional).",
    )
    p.add_argument(
        "--workspace",
        default=os.environ.get("TINYCODER_WORKSPACE", "./workspace"),
        help="Directory the agent is allowed to read/write (default: ./workspace).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum agent iterations per request.",
    )
    p.add_argument(
        "--commands",
        choices=["ask", "yes", "no"],
        default=os.environ.get("TINYCODER_COMMANDS", "ask"),
        help=(
            "Shell command policy: ask (default — confirm each command), "
            "yes (auto-approve), no (block all)."
        ),
    )
    p.add_argument(
        "--allow-commands",
        action="store_true",
        help="Backward-compat alias for --commands yes.",
    )
    p.add_argument(
        "--planner-model",
        default=os.environ.get("TINYCODER_PLANNER_MODEL"),
        help=(
            "Enable planner-executor mode. The smarter planner plans+reviews; "
            "the small executor does the work. e.g., qwen3-coder-next:cloud."
        ),
    )
    p.add_argument(
        "--planner-backend",
        choices=["ollama", "llamacpp", "openai"],
        default=os.environ.get("TINYCODER_PLANNER_BACKEND", "ollama"),
        help="Backend for the planner (default: ollama — local daemon, supports :cloud models if signed in).",
    )
    p.add_argument(
        "--planner-base-url",
        default=os.environ.get("TINYCODER_PLANNER_BASE_URL"),
        help="Override planner base URL (e.g., https://ollama.com/v1 with --planner-backend openai).",
    )
    p.add_argument(
        "--planner-api-key",
        default=os.environ.get("OLLAMA_API_KEY") or os.environ.get("TINYCODER_PLANNER_API_KEY"),
        help="Planner API key (default: env OLLAMA_API_KEY). NEVER paste a key in argv; use env.",
    )
    p.add_argument(
        "--planner-rounds",
        type=int,
        default=3,
        help="Max review/fix rounds after the initial execution (default: 3).",
    )
    p.add_argument(
        "--scaffold",
        action="store_true",
        default=bool(os.environ.get("TINYCODER_SCAFFOLD")),
        help=(
            "Scaffold-first mode: planner writes complete skeleton files with "
            "TODO(small-ai): markers, executor fills them one at a time, "
            "planner runs verification and patches any failures. "
            "Requires --planner-model. Best for framework projects where "
            "small models otherwise produce non-runnable code."
        ),
    )
    p.add_argument(
        "--log-dir",
        default=os.environ.get("TINYCODER_LOG_DIR"),
        help=(
            "Write a JSON log of every session under this directory "
            "(e.g. --log-dir ./logs). One file per user request, named "
            "session-YYYYMMDD-HHMMSS-<id>.json. Designed for a larger "
            "reviewer model to analyze and improve workflow."
        ),
    )
    p.add_argument(
        "--intervention-after",
        type=int,
        default=2,
        help="After this many failed review rounds, planner takes over with guide/takeover/replan/abort (default: 2).",
    )
    p.add_argument(
        "--max-interventions",
        type=int,
        default=3,
        help="Hard cap on planner interventions per request to bound cloud token use (default: 3).",
    )
    p.add_argument(
        "prompt",
        nargs="*",
        help="Single-shot prompt. If omitted, starts an interactive REPL.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        backend = build_backend(
            args.backend, model=args.model, base_url=args.base_url, api_key=args.api_key
        )
    except ValueError as e:
        console.print(f"[red]error:[/red] {e}")
        return 2

    workspace = Workspace(args.workspace)
    command_policy: CommandPolicy = "yes" if args.allow_commands else args.commands
    agent = Agent(
        backend=backend,
        workspace=workspace,
        max_steps=args.max_steps,
        command_policy=command_policy,
        approve_command=_console_approve,
        on_event=_render_event,
    )

    orchestrator: Orchestrator | ScaffoldOrchestrator | None = None
    if args.scaffold and not args.planner_model:
        console.print(
            "[red]error:[/red] --scaffold requires --planner-model "
            "(scaffold mode needs the big AI)."
        )
        return 2
    if args.planner_model:
        try:
            planner_backend = build_backend(
                args.planner_backend,
                model=args.planner_model,
                base_url=args.planner_base_url,
                api_key=args.planner_api_key,
            )
        except ValueError as e:
            console.print(f"[red]error:[/red] {e}")
            return 2
        planner = Planner(planner_backend)
        if args.scaffold:
            orchestrator = ScaffoldOrchestrator(
                planner=planner,
                executor=agent,
                max_review_rounds=args.planner_rounds,
                max_interventions=args.max_interventions,
                on_event=_render_orchestrator_event,
            )
        else:
            orchestrator = Orchestrator(
                planner=planner,
                executor=agent,
                max_review_rounds=args.planner_rounds,
                intervention_after=args.intervention_after,
                max_interventions=args.max_interventions,
                on_event=_render_orchestrator_event,
            )

    policy_label = {
        "ask": "ask each time",
        "yes": "auto-approve",
        "no": "blocked",
    }[command_policy]
    mode_line = ""
    if orchestrator:
        flow = "scaffold-fill-verify" if isinstance(orchestrator, ScaffoldOrchestrator) else "plan-execute-review"
        mode_line = (
            f"\n[bold]planner:[/bold]   {args.planner_model} "
            f"(via {args.planner_backend}, flow={flow}, up to {args.planner_rounds} review rounds)"
        )
    if isinstance(orchestrator, ScaffoldOrchestrator):
        title_suffix = " (scaffold)"
    elif orchestrator:
        title_suffix = " (planned)"
    else:
        title_suffix = ""
    console.print(
        Panel(
            f"[bold]backend:[/bold]   {args.backend}\n"
            f"[bold]model:[/bold]     {args.model}\n"
            f"[bold]workspace:[/bold] {workspace.root}\n"
            f"[bold]commands:[/bold]  {policy_label}"
            f"{mode_line}",
            title="tinycoder" + title_suffix,
            border_style="green",
        )
    )

    if args.prompt:
        prompt = " ".join(args.prompt)
        return _run_once(agent, prompt, orchestrator, args)

    return _repl(agent, orchestrator, args)


def _render_orchestrator_event(kind: str, payload: dict) -> None:
    if _current_session_logger is not None:
        _current_session_logger.record(kind, payload)
    if kind == "scaffolding":
        console.print("[bold magenta]◆ scaffolder: writing skeleton…[/bold magenta]")
        return
    if kind == "scaffold":
        files = payload["files"]
        fills = payload["fill_tasks"]
        body = (
            f"[dim]{payload['rationale']}[/dim]\n\n"
            f"[bold]files ({len(files)}):[/bold]\n"
            + "\n".join(f"  • {p}" for p in files)
            + f"\n\n[bold]fill tasks ({len(fills)}):[/bold]\n"
            + "\n".join(f"  {i + 1}. {t['path']} ← {t['instruction']}" for i, t in enumerate(fills))
            + (f"\n\n[bold]verify with:[/bold] {payload['run_command']}" if payload.get("run_command") else "")
        )
        console.print(Panel(body, title="scaffold", border_style="magenta"))
        return
    if kind == "scaffold_write":
        console.print(f"  [magenta]✓ scaffold {payload['path']}[/magenta]")
        return
    if kind == "scaffold_error":
        console.print(f"  [red]✗ scaffold {payload['path']}: {payload['error']}[/red]")
        return
    if kind == "fill_start":
        console.print(
            f"\n[bold magenta]◆ fill {payload['index']}/{payload['total']}:"
            f"[/bold magenta] {payload['path']}"
        )
        console.print(f"  [dim]marker:[/dim] {payload['marker']}")
        console.print(f"  [dim]→[/dim] {payload['instruction']}")
        return
    if kind == "fill_end":
        marker = "✓" if payload["finished"] else "✗"
        color = "green" if payload["finished"] else "yellow"
        console.print(f"[{color}]{marker} fill done — {payload.get('summary') or '(no summary)'}[/{color}]")
        return
    if kind == "verify":
        console.print(f"[magenta]◆ verify:[/magenta] [yellow]$ {payload['command']}[/yellow]")
        return
    if kind == "planning":
        console.print("[bold magenta]◆ planner: thinking…[/bold magenta]")
        return
    if kind == "plan":
        steps = payload["steps"]
        body = f"[dim]{payload['rationale']}[/dim]\n\n" + "\n".join(
            f"  {i + 1}. {s['description']}" for i, s in enumerate(steps)
        )
        console.print(Panel(body, title=f"plan ({len(steps)} steps)", border_style="magenta"))
        return
    if kind == "step_start":
        console.print(
            f"\n[bold magenta]◆ step {payload['index']}/{payload['total']}:"
            f"[/bold magenta] {payload['description']}"
        )
        return
    if kind == "step_end":
        marker = "✓" if payload["finished"] else "✗"
        color = "green" if payload["finished"] else "yellow"
        console.print(f"[{color}]{marker} step done — {payload.get('summary') or '(no summary)'}[/{color}]")
        return
    if kind == "step_skipped":
        console.print(
            f"\n[dim]· step {payload['index']}/{payload['total']} SKIPPED — "
            f"{payload['reason']}[/dim]"
        )
        return
    if kind == "auto_run":
        console.print(f"[magenta]◆ auto-verify:[/magenta] [yellow]$ {payload['command']}[/yellow]")
        return
    if kind == "reviewing":
        console.print(
            f"\n[bold magenta]◆ reviewer: round {payload['round']}/{payload['max']}…[/bold magenta]"
        )
        return
    if kind == "review":
        r = payload["review"]
        if r["type"] == "done":
            console.print(Panel(r["summary"], title="reviewer: DONE", border_style="green"))
        else:
            console.print(
                Panel(
                    f"[bold]issue:[/bold] {r['issue']}\n[bold]fix:[/bold] {r['instruction']}",
                    title="reviewer: FIX NEEDED",
                    border_style="yellow",
                )
            )
        return
    if kind == "fixing":
        return  # the next step_start covers it
    if kind == "intervening":
        console.print(
            f"\n[bold red]⚑ intervention {payload['attempt']}/{payload['max']}:[/bold red] "
            f"{payload['situation']}"
        )
        return
    if kind == "intervention":
        itype = payload["type"]
        rationale = payload.get("rationale", "")
        if itype == "guide":
            body = f"[bold]guide[/bold] — {rationale}\n[dim]{payload['instruction']}[/dim]"
            console.print(Panel(body, title="planner: GUIDE", border_style="cyan"))
        elif itype == "takeover":
            files = payload.get("files", [])
            body = (
                f"[bold]takeover[/bold] — {rationale}\n"
                f"writing {len(files)} file(s) directly: "
                + ", ".join(f["path"] for f in files)
            )
            console.print(Panel(body, title="planner: TAKEOVER", border_style="magenta"))
        elif itype == "replan":
            steps = payload.get("new_steps", [])
            body = f"[bold]replan[/bold] — {rationale}\n" + "\n".join(
                f"  {i + 1}. {s['description']}" for i, s in enumerate(steps)
            )
            console.print(Panel(body, title="planner: REPLAN", border_style="yellow"))
        elif itype == "abort":
            body = f"[bold]abort[/bold] — {rationale}\n[dim]{payload.get('summary','')}[/dim]"
            console.print(Panel(body, title="planner: ABORT", border_style="red"))
        return
    if kind == "intervention_skipped":
        console.print(f"[yellow]· intervention skipped: {payload['reason']}[/yellow]")
        return
    if kind == "takeover_write":
        console.print(f"  [magenta]✓ takeover write {payload['path']}[/magenta]")
        return
    if kind == "takeover_error":
        console.print(f"  [red]✗ takeover failed {payload['path']}: {payload['error']}[/red]")
        return
    if kind == "takeover_run":
        console.print(f"  [magenta]◆ takeover verify:[/magenta] [yellow]$ {payload['command']}[/yellow]")
        return
    if kind == "error":
        console.print(f"[red]· {payload['message']}[/red]")


def _open_session_log(
    args: argparse.Namespace,
    prompt: str,
    workspace: Workspace,
    orchestrator: Orchestrator | ScaffoldOrchestrator | None,
) -> SessionLogger | None:
    """Create a session logger if --log-dir is set, install it as the current
    logger so event renderers feed it. Returns the logger (or None)."""
    global _current_session_logger
    if not args.log_dir:
        return None
    if isinstance(orchestrator, ScaffoldOrchestrator):
        mode = "scaffold"
    elif orchestrator is not None:
        mode = "planner"
    else:
        mode = "plain"
    logger = SessionLogger(
        log_dir=args.log_dir,
        user_request=prompt,
        mode=mode,
        executor={"backend": args.backend, "model": args.model},
        planner=(
            {"backend": args.planner_backend, "model": args.planner_model}
            if args.planner_model
            else None
        ),
        workspace=str(workspace.root),
    )
    _current_session_logger = logger
    return logger


def _close_session_log(
    logger: SessionLogger | None, result_obj
) -> None:
    """Write the JSON file and clear the active logger. result_obj is either
    a RunResult or OrchestrationResult — both expose finished/summary/
    stopped_reason."""
    global _current_session_logger
    if logger is None:
        return
    try:
        outcome = {
            "finished": getattr(result_obj, "finished", None),
            "summary": getattr(result_obj, "summary", ""),
            "stopped_reason": getattr(result_obj, "stopped_reason", ""),
        }
        path = logger.finalize(outcome)
        console.print(f"[dim]· session log → {path}[/dim]")
    finally:
        _current_session_logger = None


def _run_once(
    agent: Agent,
    prompt: str,
    orchestrator: Orchestrator | ScaffoldOrchestrator | None = None,
    args: argparse.Namespace | None = None,
) -> int:
    console.print(f"[bold]you:[/bold] {prompt}")
    logger = (
        _open_session_log(args, prompt, agent.workspace, orchestrator)
        if args is not None
        else None
    )
    if orchestrator:
        try:
            result = orchestrator.run(prompt)
        finally:
            _close_session_log(logger, locals().get("result"))
        if result.finished:
            console.print(Panel(result.summary, title="done", border_style="green"))
            return 0
        console.print(
            Panel(
                result.stopped_reason or "incomplete",
                title="incomplete",
                border_style="yellow",
            )
        )
        return 1

    try:
        result = agent.run(prompt)
    finally:
        _close_session_log(logger, locals().get("result"))
    if result.finished:
        console.print(Panel(result.summary, title="done", border_style="green"))
        return 0
    console.print(
        Panel(
            f"agent stopped without finishing: {result.stopped_reason or 'unknown reason'}",
            title="incomplete",
            border_style="yellow",
        )
    )
    return 1


def _repl(
    agent: Agent,
    orchestrator: Orchestrator | ScaffoldOrchestrator | None = None,
    args: argparse.Namespace | None = None,
) -> int:
    console.print("[dim]type your request, /reset to start a new session, /quit to exit[/dim]")
    while True:
        try:
            prompt = console.input("[bold]you> [/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            return 0
        if prompt == "/reset":
            agent._reset_history()
            console.print("[dim]conversation reset[/dim]")
            continue
        logger = (
            _open_session_log(args, prompt, agent.workspace, orchestrator)
            if args is not None
            else None
        )
        if orchestrator:
            try:
                result = orchestrator.run(prompt)
            finally:
                _close_session_log(logger, locals().get("result"))
            if result.finished:
                console.print(Panel(result.summary, title="done", border_style="green"))
            else:
                console.print(
                    Panel(
                        result.stopped_reason or "incomplete",
                        title="incomplete",
                        border_style="yellow",
                    )
                )
            continue
        try:
            result = agent.run(prompt)
        finally:
            _close_session_log(logger, locals().get("result"))
        if result.finished:
            console.print(Panel(result.summary, title="done", border_style="green"))
        else:
            console.print(
                Panel(
                    f"stopped: {result.stopped_reason or 'unknown reason'}",
                    title="incomplete",
                    border_style="yellow",
                )
            )


if __name__ == "__main__":
    sys.exit(main())
