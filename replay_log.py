from __future__ import annotations

import json
import re
import warnings
from collections import Counter, OrderedDict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from arena_hero import (
    Accepted,
    CommandSource,
    Received,
    TransportError,
    Turn,
    __version__ as arena_hero_version,
)
from arena_tactic.log_projection import compact_logged_state, compact_strategy_trace
from arena_tactic.schema import REPLAY_LOG_SCHEMA_VERSION


LOG_SCHEMA_VERSION = REPLAY_LOG_SCHEMA_VERSION
DEFAULT_ENDPOINT = "https://api.arenahero.io"
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_PLAN_REFERENCE_LIMIT = 256


def _now() -> datetime:
    return datetime.now().astimezone()


def _model_dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _action_type(action: Any) -> str:
    data = _model_dump(action)
    return str(data.get("type", type(action).__name__))


def _safe_error(error: BaseException, secret: str | None) -> dict[str, str]:
    message = str(error)
    if secret:
        message = message.replace(secret, "[REDACTED]")
    message = _BEARER_TOKEN.sub("Bearer [REDACTED]", message)
    return {
        "type": type(error).__name__,
        "message": message[:2_000],
    }


class ReplayLogger:
    """Append-only, crash-tolerant JSON Lines log for one tactic session."""

    def __init__(
        self,
        directory: str | Path,
        *,
        tactic_name: str = "balanced_tactic",
        endpoint: str = DEFAULT_ENDPOINT,
        request_timeout: float | None = None,
        request_retries: int | None = None,
    ) -> None:
        started_at = _now()
        self.session_id = str(uuid4())
        self.started_at = started_at
        self._started_clock = perf_counter()
        self._closed = False
        self._io_error: OSError | None = None
        self._agent_plan_fingerprints: OrderedDict[int, bytes] = OrderedDict()

        log_directory = Path(directory)
        log_directory.mkdir(parents=True, exist_ok=True)
        filename = (
            f"arena_hero_{started_at:%Y%m%d_%H%M%S}_"
            f"{self.session_id[:8]}.jsonl"
        )
        self.path = log_directory / filename
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")

        if not self._write(
            {
                "record_type": "session_start",
                "schema_version": LOG_SCHEMA_VERSION,
                "recorded_at": started_at.isoformat(timespec="milliseconds"),
                "session_id": self.session_id,
                "tactic": tactic_name,
                "arena_hero_sdk": arena_hero_version,
                "endpoint": endpoint,
                "request_timeout": request_timeout,
                "request_retries": request_retries,
            }
        ):
            raise OSError(f"Could not initialize replay log: {self._io_error}")

    @property
    def io_error(self) -> OSError | None:
        return self._io_error

    def record_turn(
        self,
        turn: Turn,
        *,
        decision_ms: float,
        submission_ms: float | None,
        accepted: Accepted | None = None,
        error: BaseException | None = None,
        failure_stage: str | None = None,
        secret: str | None = None,
        strategy: dict[str, object] | None = None,
        recoverable: bool = False,
    ) -> bool:
        if (accepted is None) == (error is None):
            raise ValueError("Provide exactly one of accepted or error")

        plan = turn.plan
        plan_data = _model_dump(plan)
        state_data = compact_logged_state(_model_dump(turn.state))
        strategy_data = compact_strategy_trace(strategy)
        action_counts = Counter(
            _action_type(action) for action in plan.unit_actions.values()
        )
        core_action_type = None
        if plan.core_action is not None:
            core_action_type = _action_type(plan.core_action)
            action_counts[core_action_type] += 1

        unit_counts = Counter(unit.unit_type.value for unit in turn.units)
        event_counts = Counter(event.event_type for event in turn.events)
        core_summary = None
        if turn.core is not None:
            core_summary = {
                "id": str(turn.core.id),
                "position": list(turn.core.position),
                "hp": turn.core.hp,
                "shield": turn.core.shield,
                "state": turn.core.view.state.value,
            }

        if accepted is not None:
            submission: dict[str, Any] = {
                "status": "accepted",
                "receipt": _model_dump(accepted),
            }
        else:
            assert error is not None
            submission = {
                "status": "error",
                "outcome": (
                    "unknown"
                    if isinstance(error, TransportError)
                    else "rejected"
                ),
                "stage": failure_stage,
                "recoverable": recoverable,
                "error": _safe_error(error, secret),
            }

        written = self._write(
            {
                "record_type": "turn",
                "recorded_at": _now().isoformat(timespec="milliseconds"),
                "tick": turn.tick,
                "timing_ms": {
                    "decision": round(decision_ms, 3),
                    "submission": (
                        None if submission_ms is None else round(submission_ms, 3)
                    ),
                },
                "summary": {
                    "status": turn.state.status.value,
                    "resources": turn.resources,
                    "resource_capacity": turn.resource_capacity,
                    "population": turn.state.population,
                    "unit_counts": dict(sorted(unit_counts.items())),
                    "visible_enemy_count": len(turn.visible_enemies),
                    "visible_resource_cell_count": len(turn.resource_cells),
                    "event_counts": dict(sorted(event_counts.items())),
                    "core": core_summary,
                    "planned_unit_action_count": len(plan.unit_actions),
                    "planned_action_counts": dict(sorted(action_counts.items())),
                    "planned_core_action": core_action_type,
                },
                "strategy": strategy_data,
                "state": state_data,
                "plan": plan_data,
                "submission": submission,
            }
        )
        uncertain_agent_submission = bool(
            accepted is None
            and isinstance(error, TransportError)
            and recoverable
            and failure_stage == "submitting"
        )
        if written and (
            uncertain_agent_submission
            or (
                accepted is not None
                and accepted.source == CommandSource.AGENT
            )
        ):
            self._remember_agent_plan(turn.tick, plan_data)
        return written

    def record_turn_skip(self, *, tick: int, reason: str) -> bool:
        """Record a Turn deliberately ignored before planning or submission."""

        return self._write(
            {
                "record_type": "turn_skip",
                "recorded_at": _now().isoformat(timespec="milliseconds"),
                "tick": tick,
                "reason": reason,
            }
        )

    def record_error(
        self,
        *,
        stage: str,
        tick: int | None,
        error: BaseException,
        secret: str | None = None,
    ) -> bool:
        return self._write(
            {
                "record_type": "error",
                "recorded_at": _now().isoformat(timespec="milliseconds"),
                "stage": stage,
                "tick": tick,
                "error": _safe_error(error, secret),
            }
        )

    def record_receipt(self, receipt: Received) -> bool:
        """Record the canonical Agent or Manual plan broadcast by the server."""

        plan_data = _model_dump(receipt.plan)
        matches_turn_plan = (
            receipt.source == CommandSource.AGENT
            and self._agent_plan_fingerprints.get(receipt.tick)
            == _plan_fingerprint(plan_data)
        )
        record: dict[str, Any] = {
            "record_type": "canonical_receipt",
            "recorded_at": _now().isoformat(timespec="milliseconds"),
            "tick": receipt.tick,
            "source": receipt.source.value,
            "received_at": receipt.received_at.isoformat(timespec="milliseconds"),
            "matches_turn_plan": matches_turn_plan,
        }
        if matches_turn_plan:
            record["plan_ref"] = {"record_type": "turn", "tick": receipt.tick}
        else:
            record["plan"] = plan_data
        return self._write(record)

    def _remember_agent_plan(self, tick: int, plan: dict[str, Any]) -> None:
        self._agent_plan_fingerprints[tick] = _plan_fingerprint(plan)
        self._agent_plan_fingerprints.move_to_end(tick)
        while len(self._agent_plan_fingerprints) > _PLAN_REFERENCE_LIMIT:
            self._agent_plan_fingerprints.popitem(last=False)

    def close(self, *, status: str, last_tick: int | None) -> bool:
        if self._closed:
            return self._io_error is None

        written = self._write(
            {
                "record_type": "session_end",
                "recorded_at": _now().isoformat(timespec="milliseconds"),
                "status": status,
                "last_tick": last_tick,
                "duration_ms": round(
                    (perf_counter() - self._started_clock) * 1_000,
                    3,
                ),
            }
        )
        self._closed = True
        try:
            self._stream.close()
        except OSError as error:
            self._disable_after_io_error(error)
            return False
        return written

    def _write(self, record: dict[str, Any]) -> bool:
        if self._closed or self._io_error is not None:
            return False

        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            self._stream.write(line)
            self._stream.write("\n")
            self._stream.flush()
        except OSError as error:
            self._disable_after_io_error(error)
            return False
        return True

    def _disable_after_io_error(self, error: OSError) -> None:
        if self._io_error is not None:
            return
        self._io_error = error
        warnings.warn(
            f"Replay logging was disabled after an I/O error: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            self._stream.close()
        except OSError:
            pass


def _plan_fingerprint(plan: dict[str, Any]) -> bytes:
    payload = json.dumps(
        plan,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).digest()
