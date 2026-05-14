"""Per-session JSON log for workflow analysis.

Purpose: capture everything that happened during one user request — every
agent action, every intervention, every review verdict — and write it to
a single JSON file in `--log-dir`. The intent is to feed those files to a
larger reviewer model that can spot patterns (where the small AI gets
stuck, which intervention kind unblocks it, where tokens are wasted) and
suggest changes to prompts/budgets/rules.

File shape (one per user request):

    {
      "session_id":  "20260515-143022-a1b2c3",
      "started_at":  "2026-05-15T14:30:22",
      "ended_at":    "2026-05-15T14:35:11",
      "user_request":"create xo game with react typescript",
      "mode":        "planner",
      "executor":    {"backend": "ollama", "model": "gemma4:e2b"},
      "planner":     {"backend": "ollama", "model": "qwen3-coder-next:cloud"},
      "workspace":   "/tmp/xo-test",
      "summary_header": {
        "finished": true,
        "stopped_reason": "",
        "event_counts": {"thinking": 14, "action": 12, "intervention": 3, …},
        "intervention_kinds": {"takeover": 2, "guide": 1},
        "review_rounds": 1
      },
      "events":      [ {t, kind, payload}, … ]
    }
"""
from __future__ import annotations

import json
import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe_json(value: Any) -> Any:
    """Make any payload JSON-serializable. Falls back to repr() if needed."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    # Pydantic models, dataclasses, paths, etc.
    if hasattr(value, "model_dump"):
        try:
            return _safe_json(value.model_dump())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


class SessionLogger:
    """Captures events for one user request, writes a JSON file on finalize."""

    def __init__(
        self,
        log_dir: str | Path,
        user_request: str,
        mode: str,
        executor: dict[str, str] | None = None,
        planner: dict[str, str] | None = None,
        workspace: str | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now()
        self.session_id = (
            self.started_at.strftime("%Y%m%d-%H%M%S")
            + "-"
            + secrets.token_hex(3)
        )
        self.path = self.log_dir / f"session-{self.session_id}.json"
        self.user_request = user_request
        self.mode = mode
        self.executor = executor or {}
        self.planner = planner
        self.workspace = workspace
        self.events: list[dict[str, Any]] = []
        self._event_counts: Counter[str] = Counter()
        self._intervention_kinds: Counter[str] = Counter()
        # Best-effort finalize so an interrupted run still leaves something.
        self._finalized = False

    def record(self, kind: str, payload: dict | None = None) -> None:
        """Capture one event. Safe to call from any thread context."""
        self._event_counts[kind] += 1
        if kind == "intervention" and isinstance(payload, dict):
            itype = payload.get("type")
            if itype:
                self._intervention_kinds[itype] += 1
        self.events.append(
            {
                "t": datetime.now().isoformat(timespec="seconds"),
                "kind": kind,
                "payload": _safe_json(payload) if payload is not None else None,
            }
        )

    def finalize(self, outcome: dict | None = None) -> Path:
        """Write the JSON file. Idempotent — second call is a no-op."""
        if self._finalized:
            return self.path
        self._finalized = True
        outcome = outcome or {}
        review_rounds = self._event_counts.get("review", 0)
        summary_header = {
            "finished": outcome.get("finished"),
            "stopped_reason": outcome.get("stopped_reason", ""),
            "summary": outcome.get("summary", ""),
            "event_counts": dict(self._event_counts),
            "intervention_kinds": dict(self._intervention_kinds),
            "review_rounds": review_rounds,
        }
        data = {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "user_request": self.user_request,
            "mode": self.mode,
            "executor": self.executor,
            "planner": self.planner,
            "workspace": self.workspace,
            "summary_header": summary_header,
            "events": self.events,
        }
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return self.path
