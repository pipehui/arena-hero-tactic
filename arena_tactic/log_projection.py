from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from .schema import REPLAY_LOG_SCHEMA_VERSION


_ROUTINE_MOVE_EVENT = "UNIT_MOVE_SUCCEEDED"
_CRITICAL_REJECTION_METADATA = frozenset(
    {
        "candidate_coverage",
        "candidate_coverage_after",
        "candidate_coverage_before",
        "continuation_reachable",
        "dead_end_rejected",
        "first_step_heat",
        "firing_stance",
        "forward_exits",
        "future_attackers",
        "hold_class",
        "intercept_path_after",
        "intercept_path_before",
        "lane_version",
        "route_distance",
        "route_reachable",
        "safe_horizon",
        "scheduled_deposit_tick",
        "target_id",
        "terminal_exception",
        "visible_candidates",
    }
)
_ABNORMAL_REASON_MARKERS = (
    "BLOCK",
    "CONFLICT",
    "NO_",
    "STALL",
    "WAIT",
)


def compact_logged_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached state snapshot with routine moves encoded compactly."""

    compact = {
        key: deepcopy(value)
        for key, value in state.items()
        if key != "events"
    }
    exceptional_events: list[Any] = []
    routine_moves: list[dict[str, Any]] = []
    events = state.get("events", ())
    if not isinstance(events, (list, tuple)):
        events = ()
    for event in events:
        if (
            isinstance(event, Mapping)
            and event.get("event_type") == _ROUTINE_MOVE_EVENT
        ):
            routine_moves.append(
                {
                    "actor_id": event.get("actor_id"),
                    "position": deepcopy(event.get("position")),
                }
            )
        else:
            exceptional_events.append(deepcopy(event))
    compact["events"] = exceptional_events
    if routine_moves:
        compact["routine_moves"] = routine_moves
    return compact


def compact_strategy_trace(
    strategy: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Project the full in-memory trace into the smaller on-disk schema."""

    if strategy is None:
        return None
    compact: dict[str, Any] = {
        key: deepcopy(value)
        for key, value in strategy.items()
        if key
        not in {
            "schema_version",
            "tasks",
            "decisions",
            "resolution",
            "world",
            "economy",
            "combat",
        }
    }
    compact["schema_version"] = REPLAY_LOG_SCHEMA_VERSION
    compact["source_trace_schema"] = strategy.get("schema_version")
    compact["decisions"] = _compact_decisions(strategy.get("decisions"))
    compact["resolution"] = _compact_resolution(strategy.get("resolution"))
    compact["world"] = _compact_world(strategy.get("world"))
    compact["economy"] = _compact_economy(strategy.get("economy"))
    compact["combat"] = _compact_combat(strategy.get("combat"))
    return compact


def _compact_decisions(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return []
    decisions: list[Any] = []
    for row in value:
        if not isinstance(row, Mapping):
            decisions.append(deepcopy(row))
            continue
        final = row.get("final")
        service = row.get("service")
        rejections = row.get("key_rejections")
        if not isinstance(rejections, (list, tuple)):
            rejections = ()
        rejection_counts = Counter(
            str(item.get("rejection_reason"))
            for item in rejections
            if isinstance(item, Mapping) and item.get("rejection_reason") is not None
        )
        include_details = _decision_needs_rejection_details(final, service, rejections)
        decisions.append(
            {
                "actor_id": row.get("actor_id"),
                "actor_type": row.get("actor_type"),
                "position": deepcopy(row.get("position")),
                "final": deepcopy(final),
                "rejection_reason_counts": dict(sorted(rejection_counts.items())),
                "key_rejections": (
                    [_compact_rejection(item) for item in rejections[:3]]
                    if include_details
                    else []
                ),
                "service": deepcopy(service),
            }
        )
    return decisions


def _decision_needs_rejection_details(
    final: Any,
    service: Any,
    rejections: tuple[Any, ...] | list[Any],
) -> bool:
    if isinstance(final, Mapping):
        if final.get("action") == "WAIT":
            return True
        reason = str(final.get("reason") or "")
        if any(marker in reason for marker in _ABNORMAL_REASON_MARKERS):
            return True
    if isinstance(service, Mapping):
        if int(service.get("stalled_ticks") or 0) > 0:
            return True
        if service.get("last_route_rejection"):
            return True
        feedback = service.get("resolver_feedback")
        if isinstance(feedback, Mapping) and feedback.get("rejection_reason"):
            return True
    return any(
        isinstance(item, Mapping)
        and item.get("rejection_reason") != "ACTOR_ALREADY_ASSIGNED"
        for item in rejections
    )


def _compact_rejection(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return deepcopy(value)
    intent = value.get("intent")
    if not isinstance(intent, Mapping):
        intent = {}
    metadata = intent.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "action": intent.get("action"),
        "mission": intent.get("mission"),
        "target": deepcopy(intent.get("target")),
        "target_id": intent.get("target_id"),
        "direction": intent.get("direction"),
        "priority": intent.get("priority"),
        "risk": intent.get("risk"),
        "reason": value.get("rejection_reason"),
        "blocking_actor_ids": deepcopy(value.get("blocking_actor_ids", [])),
        "metadata": {
            key: deepcopy(metadata[key])
            for key in sorted(metadata)
            if key in _CRITICAL_REJECTION_METADATA
        },
    }


def _compact_resolution(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    rejected = value.get("rejected")
    if not isinstance(rejected, (list, tuple)):
        rejected = ()
    reason_counts = Counter(
        str(item.get("reason"))
        for item in rejected
        if isinstance(item, Mapping) and item.get("reason") is not None
    )
    return {
        "selected_count": value.get("selected_count"),
        "rejected_count": value.get("rejected_count"),
        "rejected_reason_counts": dict(sorted(reason_counts.items())),
        "reserved_positions": deepcopy(value.get("reserved_positions", [])),
        "resource_spent": value.get("resource_spent"),
        "resource_gained": value.get("resource_gained"),
    }


def _compact_world(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact = deepcopy(dict(value))
    global_map = compact.get("global_map")
    if isinstance(global_map, dict):
        global_map.pop("friendlies", None)
        vision = global_map.get("vision")
        if isinstance(vision, dict):
            vision.pop("sources", None)
    return compact


def _compact_economy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact = {
        key: deepcopy(item)
        for key, item in value.items()
        if key not in {"service_queue", "worker_scouts"}
    }
    scouts = value.get("worker_scouts")
    compact["worker_scouts"] = []
    if isinstance(scouts, (list, tuple)):
        compact["worker_scouts"] = [
            {
                key: deepcopy(row.get(key))
                for key in (
                    "worker_id",
                    "mode",
                    "target",
                    "stalled_ticks",
                    "action",
                    "reason",
                )
            }
            for row in scouts
            if isinstance(row, Mapping)
        ]
    compact["service_queue"] = _compact_service_queue(value.get("service_queue"))
    return compact


def _compact_service_queue(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    jobs = value.get("jobs")
    reservations = value.get("return_reservations")
    if not isinstance(jobs, (list, tuple)):
        jobs = ()
    if not isinstance(reservations, (list, tuple)):
        reservations = ()
    by_actor: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        actor_id = job.get("actor_id")
        if actor_id not in by_actor:
            order.append(actor_id)
        by_actor[actor_id] = deepcopy(dict(job))
    for reservation in reservations:
        if not isinstance(reservation, Mapping):
            continue
        actor_id = reservation.get("worker_id")
        if actor_id not in by_actor:
            order.append(actor_id)
            by_actor[actor_id] = {"actor_id": actor_id}
        merged = by_actor[actor_id]
        for source, target in (
            ("route_target", "route_target"),
            ("route_distance", "route_distance"),
            ("first_direction", "first_direction"),
            ("first_position", "first_position"),
            ("earliest_deposit_tick", "earliest_service_tick"),
            ("scheduled_deposit_tick", "service_tick"),
            ("departure_tick", "departure_tick"),
            ("slack_ticks", "slack_ticks"),
            ("status", "return_status"),
            ("delay_reason", "delay_reason"),
            ("route_mode", "route_mode"),
            ("waypoint", "waypoint"),
            ("lane_version", "lane_version"),
            ("previous_scheduled_tick", "previous_service_tick"),
            ("schedule_change_reason", "schedule_change_reason"),
            ("schedule_drift", "schedule_drift"),
        ):
            if reservation.get(source) is not None or target not in merged:
                merged[target] = deepcopy(reservation.get(source))
    retained = (
        "service",
        "service_core_position",
        "admission_id",
        "previous_admission_id",
        "admission_reason",
        "release_reason",
        "entrance",
        "queue_cells",
        "overflow_slots",
        "exit",
        "patient_gateway",
        "slot_schedule",
        "patient_queue",
        "service_cell_leases",
        "blocking_units",
        "reschedule_reasons",
        "core_slot_reserved",
        "reserved_resources",
        "paused_reason",
        "lane_lease",
        "lane_replan_reason",
        "liveness_indicators",
    )
    compact = {key: deepcopy(value.get(key)) for key in retained if key in value}
    compact["jobs"] = [by_actor[actor_id] for actor_id in order]
    return compact


def _compact_combat(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact = deepcopy(dict(value))
    formation = compact.get("formation")
    if not isinstance(formation, dict):
        return compact
    leases = formation.get("leases")
    waits = formation.get("waits")
    blocked = isinstance(waits, Mapping) and int(waits.get("blocked_or_idle") or 0) > 0
    stalled = isinstance(leases, list) and any(
        isinstance(row, Mapping)
        and (int(row.get("stalled_ticks") or 0) > 0 or int(row.get("blocked_ticks") or 0) > 0)
        for row in leases
    )
    feedback = formation.get("move_feedback")
    if isinstance(feedback, list):
        formation["move_feedback"] = [
            row
            for row in feedback
            if isinstance(row, Mapping)
            and (
                row.get("rejection_reason") is not None
                or int(row.get("consecutive_blocked_ticks") or 0) > 0
            )
        ]
    assignment = formation.get("assignment")
    if isinstance(assignment, dict) and not (blocked or stalled):
        assignment.pop("rejected", None)
    return compact


__all__ = ("compact_logged_state", "compact_strategy_trace")
