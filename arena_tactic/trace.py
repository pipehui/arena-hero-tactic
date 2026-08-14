from __future__ import annotations

from collections import Counter
from math import ceil

from arena_hero import UnitType

from .config import TacticConfig
from .models import (
    ActionIntent,
    CoreEvacuationCampaign,
    CoreServiceQueue,
    FireMission,
    HomeCombatAssignment,
    HomeCounterSiegeDecision,
    IntentAction,
    IntentResolution,
    UnitMission,
    WorldModel,
)
from .projection import TacticalMap
from .geometry import manhattan
from .rules import UNIT_MAX_HP
from .schema import STRATEGY_LOG_SCHEMA_VERSION
from .state import TacticMemory


class DecisionTraceBuilder:
    def __init__(self, config: TacticConfig, memory: TacticMemory) -> None:
        self.config = config
        self.memory = memory

    def build(
        self,
        world: WorldModel,
        projection: TacticalMap,
        resolution: IntentResolution,
        service: CoreServiceQueue,
        fire_missions: tuple[FireMission, ...],
        legal_opportunities,
        evacuation: CoreEvacuationCampaign | None,
        production_candidates: tuple[dict[str, object], ...],
        home_combat_assignment: HomeCombatAssignment = HomeCombatAssignment(),
        counter_siege: HomeCounterSiegeDecision = HomeCounterSiegeDecision(),
        home_defense_active: bool = False,
    ) -> dict[str, object]:
        decisions = self._decision_dicts(world, resolution, service)
        return {
            "schema_version": STRATEGY_LOG_SCHEMA_VERSION,
            "mode": "GLOBAL_MAP_SURVIVAL_ECONOMY",
            "outcomes": self._outcomes(),
            "tasks": [self.intent_dict(intent) for intent in resolution.selected],
            "decisions": decisions,
            "decision_summary": {
                "final_reason_counts": dict(
                    sorted(Counter(intent.reason for intent in resolution.selected).items())
                ),
                "wait_reason_counts": dict(
                    sorted(
                        Counter(
                            intent.reason
                            for intent in resolution.selected
                            if intent.action is IntentAction.WAIT
                        ).items()
                    )
                ),
            },
            "resolution": self._resolution_dict(resolution),
            "capacity_policy": {
                "physical_capacity": 2,
                "core_unit_capacity": 1,
                "home_defense_active": home_defense_active,
                "wartime_worker_exclusive": home_defense_active,
                "wartime_combat_exclusive": home_defense_active,
                "mixed_worker_combat_allowed": True,
            },
            "world": self._world_dict(world, projection),
            "economy": self._economy_dict(
                world,
                projection,
                service,
                production_candidates,
                resolution,
            ),
            "combat": self._combat_dict(
                world,
                projection,
                resolution,
                fire_missions,
                legal_opportunities,
                home_combat_assignment,
                counter_siege,
            ),
            "core_safety": self._core_safety_dict(evacuation),
        }

    def _decision_dicts(
        self,
        world: WorldModel,
        resolution: IntentResolution,
        service: CoreServiceQueue,
    ) -> list[dict[str, object]]:
        units = {unit.id: unit for unit in world.friendlies}
        rejected_by_actor: dict[object, list] = {}
        for row in resolution.rejected:
            rejected_by_actor.setdefault(row.intent.actor_id, []).append(row)
        reservations = {row.worker_id: row for row in service.return_reservations}
        progress = {
            worker_id: (position, stalled)
            for worker_id, position, stalled in service.worker_progress
        }
        rows: list[dict[str, object]] = []
        for final in resolution.selected:
            actor = None if final.actor_id is None else units.get(final.actor_id)
            rejected = sorted(
                rejected_by_actor.get(final.actor_id, ()),
                key=lambda row: (
                    row.reason == "ACTOR_ALREADY_ASSIGNED",
                    row.intent.sort_key(),
                ),
            )[:3]
            critical = []
            for row in rejected:
                target = row.intent.target_position
                blockers = []
                if target is not None:
                    blockers = [
                        str(unit.id)
                        for unit in world.friendlies
                        if unit.id != row.intent.actor_id and unit.position == target
                    ]
                    blockers.extend(
                        str(unit.id)
                        for unit in world.enemies
                        if unit.position == target
                    )
                    blockers.extend(
                        str(core.id)
                        for core in world.enemy_cores
                        if core.position == target
                    )
                critical.append(
                    {
                        "intent": self.intent_dict(row.intent),
                        "rejection_reason": row.reason,
                        "blocking_actor_ids": sorted(blockers),
                    }
                )
            service_state = None
            if actor is not None and actor.unit_type is UnitType.WORKER:
                reservation = reservations.get(actor.id)
                position, stalled = progress.get(actor.id, (actor.position, 0))
                feedback = self.memory.service_move_feedback.get(actor.id)
                cargo_progress = self.memory.service_cargo_route_progress.get(actor.id)
                service_state = {
                    "scheduled_deposit_tick": (
                        None
                        if reservation is None
                        else reservation.scheduled_deposit_tick
                    ),
                    "departure_tick": (
                        None if reservation is None else reservation.departure_tick
                    ),
                    "stage": None if reservation is None else reservation.status,
                    "route_target": (
                        None
                        if reservation is None or reservation.route_target is None
                        else list(reservation.route_target)
                    ),
                    "remaining_distance": (
                        None if reservation is None else reservation.route_distance
                    ),
                    "route_mode": (
                        None if reservation is None else reservation.route_mode
                    ),
                    "waypoint": (
                        None
                        if reservation is None or reservation.waypoint is None
                        else list(reservation.waypoint)
                    ),
                    "lane_version": (
                        None if reservation is None else reservation.lane_version
                    ),
                    "transit_hold": (
                        None
                        if actor.id not in dict(service.overflow_slots)
                        else list(dict(service.overflow_slots)[actor.id])
                    ),
                    "progress_position": list(position),
                    "stalled_ticks": stalled,
                    "ping_pong_ticks": (
                        0 if cargo_progress is None else cargo_progress.ping_pong_ticks
                    ),
                    "last_route_rejection": (
                        None
                        if cargo_progress is None
                        else cargo_progress.last_rejection_reason
                    ),
                    "resolver_feedback": (
                        None
                        if feedback is None
                        else {
                            "selected": feedback.selected,
                            "destination": (
                                None
                                if feedback.destination is None
                                else list(feedback.destination)
                            ),
                            "rejection_reason": feedback.rejection_reason,
                            "stalled_ticks": feedback.stalled_ticks,
                        }
                    ),
                }
            if actor is not None and actor.id in self.memory.service_transit_progress:
                transit = self.memory.service_transit_progress[actor.id]
                if service_state is None:
                    service_state = {}
                service_state["transit"] = {
                    "kind": transit.kind.value,
                    "destination": (
                        None
                        if transit.destination is None
                        else list(transit.destination)
                    ),
                    "remaining_distance": transit.remaining_distance,
                    "selected": transit.selected,
                    "stalled_ticks": transit.stalled_ticks,
                    "rejection_reason": transit.rejection_reason,
                    "shared_with_id": (
                        None
                        if transit.shared_with_id is None
                        else str(transit.shared_with_id)
                    ),
                }
            if actor is not None and actor.id in self.memory.service_transit_routes:
                route = self.memory.service_transit_routes[actor.id]
                if service_state is None:
                    service_state = {}
                service_state["transit_route"] = {
                    "kind": route.kind.value,
                    "target": list(route.target),
                    "route_distance": route.route_distance,
                    "service_tick": route.service_tick,
                    "exit_tick": route.exit_tick,
                    "options": [
                        {
                            "direction": direction.value,
                            "destination": list(destination),
                            "remaining_distance": distance,
                        }
                        for direction, destination, distance in route.options
                    ],
                }
            rows.append(
                {
                    "actor_id": None if final.actor_id is None else str(final.actor_id),
                    "actor_type": (
                        "CORE" if actor is None else actor.unit_type.value
                    ),
                    "position": (
                        list(world.core.position)
                        if actor is None and world.core is not None
                        else None if actor is None else list(actor.position)
                    ),
                    "final": self.intent_dict(final),
                    "final_reason": final.reason,
                    "key_rejections": critical,
                    "service": service_state,
                }
            )
        return rows

    def _outcomes(self) -> dict[str, object]:
        shot_hits = self.memory.event_counts["SHOT_HIT"]
        shot_misses = self.memory.event_counts["SHOT_MISSED"]
        shots_resolved = shot_hits + shot_misses
        return {
            "events": dict(sorted(self.memory.event_counts.items())),
            "ranger_accuracy_percent": (
                None
                if not shots_resolved
                else round(100 * shot_hits / shots_resolved, 1)
            ),
            "sweep_target_hits": self.memory.event_counts["SWEEP_TARGET_HITS"],
            "sweep_resolved": self.memory.event_counts["SWEEP_RESOLVED"],
            "deposit_successes": self.memory.event_counts["DEPOSIT_SUCCEEDED"],
            "move_failures": (
                self.memory.event_counts["UNIT_MOVE_FAILED"]
                + self.memory.event_counts["CORE_MOVE_FAILED"]
                + self.memory.event_counts["CORE_MOVE_START_FAILED"]
            ),
        }

    def _resolution_dict(self, resolution: IntentResolution) -> dict[str, object]:
        return {
            "selected_count": len(resolution.selected),
            "rejected_count": len(resolution.rejected),
            "rejected": [
                {"intent": self.intent_dict(item.intent), "reason": item.reason}
                for item in resolution.rejected[:32]
            ],
            "reserved_positions": [list(cell) for cell in resolution.reserved_positions],
            "resource_spent": resolution.resource_spent,
            "resource_gained": resolution.resource_gained,
        }

    def _world_dict(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> dict[str, object]:
        source_counts = Counter(source.actor_kind for source in projection.vision_sources)
        coverage_sizes = [
            len(observer_ids)
            for observer_ids in projection.visibility_coverage.values()
        ]
        return {
            "tick": world.tick,
            "known_obstacles": len(world.known_obstacles),
            "known_passable": len(world.known_passable),
            "remembered_resources": len(world.remembered_resources),
            "danger_cells": len(world.danger_cells),
            "threat_heat_cells": len(world.threat_heat),
            "maximum_threat_heat": max(
                (risk for _, risk in world.threat_heat), default=0
            ),
            "immediate_attack_cells": len(projection.immediate_damage),
            "future_attack_cells": len(projection.future_damage),
            "visible_enemy_count": len(world.enemies) + len(world.enemy_cores),
            "global_map": {
                "terrain": {
                    "obstacles": len(projection.known_obstacles),
                    "passable": len(projection.known_passable),
                },
                "vision": {
                    "visible_cells": len(projection.visible_cells),
                    "remembered_cells": len(projection.last_visible_ticks),
                    "source_counts": dict(sorted(source_counts.items())),
                    "max_observers_per_cell": max(coverage_sizes, default=0),
                    "sources": [
                        {
                            "actor_id": str(source.actor_id),
                            "kind": source.actor_kind,
                            "unit_type": (
                                None
                                if source.unit_type is None
                                else source.unit_type.value
                            ),
                            "position": list(source.position),
                            "radius": source.radius,
                            "visible_cells": len(source.visible_cells),
                        }
                        for source in projection.vision_sources
                    ],
                },
                "resources": [
                    {
                        "position": list(resource.position),
                        "visible_now": resource.visible_now,
                        "last_seen_tick": resource.last_seen_tick,
                        "assigned_workers": [
                            str(worker_id)
                            for worker_id in resource.assigned_worker_ids
                        ],
                    }
                    for resource in projection.resources
                ],
                "enemies": [
                    {
                        "enemy_id": str(enemy.enemy_id),
                        "unit_type": enemy.unit_type.value,
                        "position": list(enemy.observed_position),
                        "visible_now": enemy.visible_now,
                        "last_seen_tick": enemy.last_seen_tick,
                        "age": enemy.age,
                        "confidence": enemy.confidence,
                        "observer_ids": [
                            str(observer_id) for observer_id in enemy.observer_ids
                        ],
                        "possible_positions": [
                            list(cell) for cell in enemy.possible_positions
                        ],
                        "movement_corridor": [
                            list(cell) for cell in enemy.movement_corridor
                        ],
                    }
                    for enemy in projection.enemies
                ],
                "threat": {
                    "cells": len(projection.threat_cells),
                    "immediate_cells": len(projection.immediate_damage),
                    "future_cells": len(projection.future_damage),
                    "worker_route_risk_cells": len(projection.worker_route_costs),
                    "remembered_cells": len(projection.remembered_danger),
                },
                "friendlies": [
                    {
                        "actor_id": str(actor_id),
                        "unit_type": (
                            None
                            if projection.friendly_types[actor_id] is None
                            else projection.friendly_types[actor_id].value
                        ),
                        "position": list(position),
                        "planned_position": (
                            None
                            if actor_id not in projection.planned_positions
                            else list(projection.planned_positions[actor_id])
                        ),
                    }
                    for actor_id, position in sorted(
                        projection.friendly_positions.items(),
                        key=lambda item: item[0].bytes,
                    )
                ],
                "operations": {
                    "projected_core_position": (
                        None
                        if projection.projected_core_position is None
                        else list(projection.projected_core_position)
                    ),
                    "service_positions": [
                        list(cell) for cell in sorted(projection.service_positions)
                    ],
                    "reserved_positions": [
                        list(cell) for cell in sorted(projection.reserved_positions)
                    ],
                    "congestion_cells": len(projection.congestion),
                },
            },
            "recent_move_failures": [
                {
                    "actor_id": str(actor_id),
                    "destination": list(failure.destination),
                    "expires_tick": failure.expires_tick,
                    "reason": failure.reason,
                }
                for actor_id, failure in sorted(
                    self.memory.failed_unit_moves.items(),
                    key=lambda item: item[0].bytes,
                )
            ],
            "beacon": {
                "position": list(world.beacon.position),
                "status": (
                    None if world.beacon.status is None else world.beacon.status.value
                ),
                "carrier_id": (
                    None
                    if world.beacon.carrier_id is None
                    else str(world.beacon.carrier_id)
                ),
            },
        }

    def _economy_dict(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        production_candidates: tuple[dict[str, object], ...],
        resolution: IntentResolution,
    ) -> dict[str, object]:
        workers = sum(unit.unit_type is UnitType.WORKER for unit in world.friendlies)
        combat_units = sum(
            unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            for unit in world.friendlies
        )
        selected_by_actor = {
            intent.actor_id: intent
            for intent in resolution.selected
            if intent.actor_id is not None
        }
        heat = dict(world.threat_heat)
        production_mode = next(
            (
                str(item.get("production_mode"))
                for item in production_candidates
                if item.get("production_mode") is not None
            ),
            "NONE",
        )
        crisis = self.memory.crisis_force_baseline
        core_position = None if world.core is None else world.core.position
        scout_limit = self.config.exploration_sector_radii[-1]
        staging_limit = self.config.worker_full_storage_parking_max_radius

        def cycle_period(actor_id) -> int | None:
            history = self.memory.position_history.get(actor_id, ())
            for period in range(1, 5):
                if (
                    len(history) >= period * 2
                    and history[-period * 2 : -period] == history[-period:]
                ):
                    if len(set(history[-period:])) > 1:
                        return period
            return None

        empty_scouts_beyond = sum(
            unit.unit_type is UnitType.WORKER
            and unit.cargo == 0
            and core_position is not None
            and manhattan(unit.position, core_position) > scout_limit
            for unit in world.friendlies
        )
        cargo_beyond_staging = sum(
            unit.unit_type is UnitType.WORKER
            and unit.cargo > 0
            and core_position is not None
            and manhattan(unit.position, core_position) > staging_limit
            for unit in world.friendlies
        )
        worker_cycle_count = sum(
            cycle_period(unit.id) is not None
            for unit in world.friendlies
            if unit.unit_type is UnitType.WORKER
        )
        cargo_stall_count = sum(
            progress.stalled_ticks >= 2
            for progress in self.memory.service_cargo_route_progress.values()
        )
        return {
            "workers": workers,
            "worker_target": ceil(
                world.population * self.config.worker_ratio_percent / 100
            ),
            "stockpile_active": production_mode == "HIGH_POP_STOCKPILE",
            "stockpile_population_threshold": self.config.population_stockpile_threshold,
            "stockpile_worker_target": self.config.stockpile_worker_target,
            "stockpile_combat_target": self.config.stockpile_combat_target,
            "stockpile_worker_gap": max(
                0,
                self.config.stockpile_worker_target - workers,
            ),
            "stockpile_combat_gap": max(
                0,
                max(
                    self.config.stockpile_combat_target,
                    self.config.home_force_floor,
                    self.memory.home_force_high_water,
                )
                - combat_units,
            ),
            "production_mode": production_mode,
            "full_storage_gate": world.resources == world.resource_capacity,
            "crisis_force_baseline": (
                None
                if crisis is None
                else {
                    "vanguards": crisis.vanguards,
                    "rangers": crisis.rangers,
                    "started_tick": crisis.started_tick,
                    "phase": crisis.phase,
                    "safe_ticks": crisis.safe_ticks,
                    "vanguard_gap": max(
                        0,
                        crisis.vanguards
                        - sum(
                            unit.unit_type is UnitType.VANGUARD
                            for unit in world.friendlies
                        ),
                    ),
                    "ranger_gap": max(
                        0,
                        crisis.rangers
                        - sum(
                            unit.unit_type is UnitType.RANGER
                            for unit in world.friendlies
                        ),
                    ),
                }
            ),
            "combat_losses": [
                {
                    "actor_id": str(record.actor_id),
                    "unit_type": record.unit_type.value,
                    "tick": record.tick,
                    "provenance": record.provenance.value,
                }
                for record in self.memory.recent_combat_losses
            ],
            "storage_saturated": self.memory.storage_saturated,
            "storage_headroom": max(0, world.resource_capacity - world.resources),
            "worker_bounds": {
                "scout_radii": list(self.config.exploration_sector_radii),
                "max_scout_radius": scout_limit,
                "full_storage_staging": [
                    self.config.worker_full_storage_parking_min_radius,
                    staging_limit,
                ],
                "fallback_staging_max_radius": 14,
            },
            "worker_activity_metrics": {
                "empty_scouts_beyond_30": empty_scouts_beyond,
                "cargo_workers_beyond_12": cargo_beyond_staging,
                "worker_periodic_cycles": worker_cycle_count,
                "cargo_return_stalls": cargo_stall_count,
            },
            "worker_home_guard_radii": list(
                sorted(
                    {
                        manhattan(world.core.position, post)
                        for post in self.memory.worker_home_guard_targets.values()
                    }
                )
                if world.core is not None and self.memory.worker_home_guard_targets
                else self.config.worker_full_storage_guard_radii
            ),
            "worker_home_guard": [
                {
                    "worker_id": str(worker_id),
                    "post": list(post),
                    "worker_core_distance": (
                        None
                        if world.core is None
                        or (unit := world.friendly(worker_id)) is None
                        else manhattan(unit.position, world.core.position)
                    ),
                    "post_core_distance": (
                        None
                        if world.core is None
                        else manhattan(post, world.core.position)
                    ),
                    "zone": (
                        None
                        if (
                            parking := self.memory.worker_parking_assignments.get(
                                worker_id
                            )
                        ) is None
                        else parking.zone
                    ),
                    "action": (
                        None
                        if worker_id not in selected_by_actor
                        else selected_by_actor[worker_id].action.value
                    ),
                    "reason": (
                        None
                        if worker_id not in selected_by_actor
                        else selected_by_actor[worker_id].reason
                    ),
                }
                for worker_id, post in sorted(
                    self.memory.worker_home_guard_targets.items(),
                    key=lambda item: item[0].bytes,
                )
                if world.friendly(worker_id) is not None
            ],
            "opening_complete": self.memory.opening_complete,
            "service_queue": self.service_dict(service),
            "worker_path_hard_blocks": [
                list(cell)
                for cell in sorted(
                    projection.hostile_occupied
                    | {
                        cell
                        for cell in (
                            service.service_core_position,
                            service.entrance,
                            service.exit_cell,
                            *service.queue_cells,
                            *dict(service.overflow_slots).values(),
                        )
                        if cell is not None
                    }
                )
            ],
            "production_candidates": list(production_candidates),
            "recovery_reserved_resources": service.reserved_resources,
            "worker_recoveries": [
                {
                    "worker_id": str(unit.id),
                    "hp": unit.hp,
                    "stage": UnitMission.RECOVER.value,
                    "action": (
                        None
                        if unit.id not in selected_by_actor
                        else selected_by_actor[unit.id].action.value
                    ),
                    "reason": (
                        None
                        if unit.id not in selected_by_actor
                        else selected_by_actor[unit.id].reason
                    ),
                    "destination": (
                        None
                        if unit.id not in selected_by_actor
                        or selected_by_actor[unit.id].target_position is None
                        else list(selected_by_actor[unit.id].target_position)
                    ),
                    "route": (
                        {}
                        if unit.id not in selected_by_actor
                        else {
                            key: value
                            for key, value in selected_by_actor[unit.id].metadata
                            if key
                            in {
                                "forward_exits",
                                "survival_terminals",
                                "first_step_heat",
                                "future_attackers",
                                "route_reachable",
                            }
                        }
                    ),
                    "nearby_heat": [
                        {"cell": list(cell), "risk": risk}
                        for cell, risk in sorted(heat.items())
                        if manhattan(cell, unit.position) <= 3
                    ][:16],
                }
                for unit in world.friendlies
                if unit.unit_type is UnitType.WORKER
                and unit.hp < UNIT_MAX_HP[UnitType.WORKER]
            ],
            "worker_scouts": [
                {
                    "worker_id": str(worker_id),
                    "slot": state.slot,
                    "sector": state.sector_index,
                    "stage": state.stage,
                    "patrol_radius": self.config.exploration_sector_radii[
                        state.stage % len(self.config.exploration_sector_radii)
                    ],
                    "mode": state.phase.value,
                    "target": None if state.target is None else list(state.target),
                    "worker_core_distance": (
                        None
                        if world.core is None
                        or (unit := world.friendly(worker_id)) is None
                        else manhattan(unit.position, world.core.position)
                    ),
                    "target_core_distance": (
                        None
                        if world.core is None or state.target is None
                        else manhattan(state.target, world.core.position)
                    ),
                    "return_to_band": (
                        state.phase.value == "RETURN_TO_BAND"
                    ),
                    "assigned_tick": state.assigned_tick,
                    "best_route_cost": state.best_route_cost,
                    "stalled_ticks": state.stalled_ticks,
                    "backoff_until": state.backoff_until,
                    "reachable_candidates": state.reachable_candidates,
                    "scan_budget": (
                        "GRANTED"
                        if state.last_scan_tick == world.tick
                        else (
                            "CHEAP_FALLBACK"
                            if state.assigned_tick == world.tick
                            and state.phase.value
                            in {"SECTOR_SCOUT", "LOCAL_DISPERSAL"}
                            else "STICKY_TARGET"
                        )
                    ),
                    "action": (
                        None
                        if worker_id not in selected_by_actor
                        else selected_by_actor[worker_id].action.value
                    ),
                    "reason": (
                        None
                        if worker_id not in selected_by_actor
                        else selected_by_actor[worker_id].reason
                    ),
                }
                for worker_id, state in sorted(
                    self.memory.worker_scout_states.items(),
                    key=lambda item: item[0].bytes,
                )
                if world.friendly(worker_id) is not None
            ],
            "resource_assignments": [
                {
                    "worker_id": str(unit_id),
                    "target": None if mission.target is None else list(mission.target),
                    "mission": mission.mission.value,
                }
                for unit_id, mission in sorted(
                    self.memory.unit_missions.items(), key=lambda item: item[0].bytes
                )
                if mission.mission in {UnitMission.HARVEST, UnitMission.EXPLORE}
            ],
            "worker_task_progress": [
                {
                    "worker_id": str(worker_id),
                    "target": list(progress.target),
                    "route_distance": progress.route_distance,
                    "last_progress_tick": progress.last_progress_tick,
                    "stalled_ticks": progress.stalled_ticks,
                    "rejection_reason": progress.rejection_reason,
                    "backoff_until": progress.backoff_until,
                    "selected_action": (
                        None
                        if worker_id not in selected_by_actor
                        else selected_by_actor[worker_id].action.value
                    ),
                    "selected_reason": (
                        None
                        if worker_id not in selected_by_actor
                        else selected_by_actor[worker_id].reason
                    ),
                }
                for worker_id, progress in sorted(
                    self.memory.worker_task_progress.items(),
                    key=lambda item: item[0].bytes,
                )
                if world.friendly(worker_id) is not None
            ],
            "worker_escapes": [
                {
                    "worker_id": str(worker_id),
                    "phase": state.phase,
                    "threat_ids": [str(item) for item in state.threat_ids],
                    "last_threat_tick": state.last_threat_tick,
                    "safe_ticks": state.safe_ticks,
                    "waypoint": (
                        None if state.waypoint is None else list(state.waypoint)
                    ),
                    "last_min_enemy_distance": state.last_min_enemy_distance,
                    "stalled_ticks": state.stalled_ticks,
                    "loop_period": state.loop_period,
                    "route_version": state.route_version,
                    "waypoint_assigned_tick": state.waypoint_assigned_tick,
                    "waypoint_expires_tick": state.waypoint_expires_tick,
                    "waypoint_invalid_reason": state.waypoint_invalid_reason,
                    "last_waypoint_distance": state.last_waypoint_distance,
                    "control_core_ids": [
                        str(item) for item in state.control_core_ids
                    ],
                    "control_centers": [
                        list(item) for item in state.control_centers
                    ],
                }
                for worker_id, state in sorted(
                    self.memory.worker_escape_states.items(),
                    key=lambda item: item[0].bytes,
                )
            ],
            "enemy_core_control_zones": [
                {
                    "core_id": str(zone.core_id),
                    "center": list(zone.center),
                    "exclusion_radius": zone.exclusion_radius,
                    "clear_radius": zone.clear_radius,
                    "last_seen_tick": zone.last_seen_tick,
                    "visible_now": zone.visible_now,
                    "expires_tick": zone.expires_tick,
                }
                for zone in sorted(
                    self.memory.enemy_core_control_zones.values(),
                    key=lambda item: item.core_id.bytes,
                )
            ],
            "worker_disengagements": [
                {
                    "worker_id": str(lease.worker_id),
                    "core_id": str(lease.core_id),
                    "center": list(lease.center),
                    "waypoint": None if lease.waypoint is None else list(lease.waypoint),
                    "assigned_tick": lease.assigned_tick,
                    "safe_ticks": lease.safe_ticks,
                    "distance": lease.last_distance,
                    "last_position": (
                        None
                        if lease.last_position is None
                        else list(lease.last_position)
                    ),
                    "stalled_ticks": lease.stalled_ticks,
                    "abandoned_target": (
                        None
                        if lease.abandoned_target is None
                        else list(lease.abandoned_target)
                    ),
                }
                for lease in sorted(
                    self.memory.worker_disengage_leases.values(),
                    key=lambda item: item.worker_id.bytes,
                )
            ],
        }

    def _combat_dict(
        self,
        world: WorldModel,
        projection: TacticalMap,
        resolution: IntentResolution,
        fire_missions: tuple[FireMission, ...],
        legal_opportunities,
        home_combat_assignment: HomeCombatAssignment,
        counter_siege: HomeCounterSiegeDecision,
    ) -> dict[str, object]:
        selected_attackers = {
            intent.actor_id
            for intent in resolution.selected
            if intent.action in {IntentAction.SHOOT, IntentAction.SHOOT_CELL}
        }
        legal_shooters = {shooter for shooter, _, _ in legal_opportunities}
        utilized = len(legal_shooters & selected_attackers)
        dynamic_fire_lines = [
            intent
            for intent in resolution.selected
            if intent.reason in {
                "ADVANCE_TO_DYNAMIC_FIRE_LINE",
                "LOW_VALUE_FIRE_REPOSITION",
            }
        ]
        vanguard_intercepts = [
            intent
            for intent in resolution.selected
            if intent.reason == "ROUTE_INTERCEPT_ADVANCE"
        ]
        return {
            "counter_siege": {
                "phase": counter_siege.phase,
                "target_id": (
                    None
                    if counter_siege.target_id is None
                    else str(counter_siege.target_id)
                ),
                "target_position": (
                    None
                    if counter_siege.target_position is None
                    else list(counter_siege.target_position)
                ),
                "members": [str(item) for item in counter_siege.member_ids],
                "home_reserve": [str(item) for item in counter_siege.reserve_ids],
                "last_seen_tick": counter_siege.last_seen_tick,
                "reason": counter_siege.reason,
                "actions": [
                    self.intent_dict(intent)
                    for intent in resolution.selected
                    if intent.mission is UnitMission.COUNTER_SIEGE
                ],
            },
            "fire_missions": [self.fire_mission_dict(item) for item in fire_missions],
            "low_value_shots_deferred": [
                {
                    "target_id": str(mission.target_id),
                    "target_type": mission.target_type.value,
                    "confidence": mission.confidence,
                    "candidate_cells": [list(cell) for cell in mission.candidate_cells],
                    "reason": "SINGLE_RANGER_MOVING_WORKER_LOW_CONFIDENCE",
                }
                for mission in fire_missions
                if mission.target_type is UnitType.WORKER
                and not mission.urgent
                and mission.confidence != "HIGH"
                and len(mission.candidate_cells) > 1
                and not mission.assignments
                and len(
                    [
                        unit
                        for unit in world.friendlies
                        if unit.unit_type is UnitType.RANGER
                        and unit.hp * 2 > UNIT_MAX_HP[UnitType.RANGER]
                    ]
                )
                == 1
            ],
            "enemy_action_candidates": [
                {
                    "target_id": str(mission.target_id),
                    "target_type": (
                        mission.target_kind
                        if mission.target_type is None
                        else mission.target_type.value
                    ),
                    "current": (
                        None
                        if world.enemy(mission.target_id) is None
                        else list(world.enemy(mission.target_id).position)
                    ),
                    "current_attack_targets": (
                        0
                        if projection.enemy(mission.target_id) is None
                        else sum(
                            unit.position
                            in projection.enemy(mission.target_id).immediate_attack_cells
                            for unit in world.friendlies
                        )
                        + int(
                            world.core is not None
                            and world.core.position
                            in projection.enemy(mission.target_id).immediate_attack_cells
                        )
                    ),
                    "ranked_cells": [list(cell) for cell in mission.candidate_cells],
                    "prediction_mode": mission.prediction_mode,
                    "candidate_roles": list(mission.candidate_roles),
                    "evidence": list(mission.evidence),
                    "split_fire": mission.split_fire,
                }
                for mission in fire_missions
                if mission.target_type is not None
            ],
            "legal_attack_opportunities": len(legal_shooters),
            "utilized_attack_opportunities": utilized,
            "utilization_percent": (
                100
                if not legal_shooters
                else round(100 * utilized / len(legal_shooters), 1)
            ),
            "dynamic_fire_lines": [
                {
                    "ranger_id": str(intent.actor_id),
                    "next_cell": list(intent.target_position),
                    "target_id": dict(intent.metadata).get("target_id"),
                    "firing_stance": list(
                        dict(intent.metadata).get("firing_stance")
                    ),
                    "candidate_coverage": dict(intent.metadata).get(
                        "candidate_coverage"
                    ),
                    "candidate_rank_limit": dict(intent.metadata).get(
                        "candidate_rank_limit"
                    ),
                    "first_step_high_value_coverage": dict(intent.metadata).get(
                        "first_step_high_value_coverage"
                    ),
                    "low_value_fire_rejected": dict(intent.metadata).get(
                        "rejected_low_rank_fire"
                    ),
                    "screening_role": dict(intent.metadata).get(
                        "screening_role"
                    ),
                    "visible_candidates": [
                        list(cell)
                        for cell in dict(intent.metadata).get(
                            "visible_candidates", ()
                        )
                    ],
                    "route_distance": dict(intent.metadata).get(
                        "route_distance"
                    ),
                }
                for intent in dynamic_fire_lines
            ],
            "screening_contact": [
                {
                    "target_id": str(decision.target_id),
                    "target_visible": decision.target_visible,
                    "candidate_cells": [
                        list(cell) for cell in decision.candidate_cells
                    ],
                    "contact_ranger_id": (
                        None
                        if decision.contact_ranger_id is None
                        else str(decision.contact_ranger_id)
                    ),
                    "fire_support_ranger_id": (
                        None
                        if decision.fire_support_ranger_id is None
                        else str(decision.fire_support_ranger_id)
                    ),
                    "visible_before": decision.visible_before,
                    "visible_after": decision.visible_after,
                    "reason": decision.reason,
                    "options": [
                        {
                            "ranger_id": str(option.ranger_id),
                            "role": option.role,
                            "theoretical_stance": list(option.stance),
                            "first_direction": (
                                None
                                if option.first_direction is None
                                else option.first_direction.value
                            ),
                            "first_position": (
                                None
                                if option.first_position is None
                                else list(option.first_position)
                            ),
                            "route_distance": option.route_distance,
                            "visible_candidates": [
                                list(cell) for cell in option.visible_candidates
                            ],
                            "firing_candidates": [
                                list(cell) for cell in option.firing_candidates
                            ],
                            "risk": option.risk,
                            "viable": option.viable,
                            "rejection_reason": option.rejection_reason,
                        }
                        for option in decision.options
                    ],
                }
                for decision in sorted(
                    self.memory.screening_contact_decisions.values(),
                    key=lambda item: item.target_id.bytes,
                )
            ],
            "vanguard_intercepts": [
                {
                    "vanguard_id": str(intent.actor_id),
                    "next_cell": list(intent.target_position),
                    "target_id": dict(intent.metadata).get("target_id"),
                    "path_before": dict(intent.metadata).get(
                        "intercept_path_before"
                    ),
                    "path_after": dict(intent.metadata).get(
                        "intercept_path_after"
                    ),
                    "coverage_before": dict(intent.metadata).get(
                        "candidate_coverage_before"
                    ),
                    "coverage_after": dict(intent.metadata).get(
                        "candidate_coverage_after"
                    ),
                }
                for intent in vanguard_intercepts
            ],
            "home_vanguard_assignment": {
                "tasks": [
                    {
                        "vanguard_id": str(task.vanguard_id),
                        "target_id": str(task.target_id),
                        "sector": task.sector.value,
                        "phase": task.phase,
                        "intercept_cell": list(task.intercept_cell),
                        "candidate_cells": [
                            list(cell) for cell in task.candidate_cells
                        ],
                        "cost": list(task.cost),
                        "lease": (
                            None
                            if (
                                lease := self.memory.vanguard_intercept_leases.get(
                                    task.vanguard_id
                                )
                            )
                            is None
                            else {
                                "target_id": str(lease.target_id),
                                "intercept_cell": list(lease.intercept_cell),
                                "assigned_tick": lease.assigned_tick,
                                "last_route_distance": lease.last_route_distance,
                                "no_progress_ticks": lease.no_progress_ticks,
                                "invalidation_reason": lease.invalidation_reason,
                            }
                        ),
                    }
                    for task in home_combat_assignment.tasks
                ],
                "candidates": [
                    {
                        "vanguard_id": str(candidate.vanguard_id),
                        "target_id": str(candidate.target_id),
                        "cost": (
                            None
                            if candidate.cost is None
                            else list(candidate.cost)
                        ),
                        "selected": candidate.selected,
                        "reason": candidate.reason,
                    }
                    for candidate in home_combat_assignment.candidates
                ],
                "unassigned_vanguards": [
                    str(item) for item in home_combat_assignment.unassigned_vanguards
                ],
                "uncovered_targets": [
                    str(item) for item in home_combat_assignment.uncovered_targets
                ],
            },
            "joint_sweeps": [
                {
                    "vanguard_id": str(intent.actor_id),
                    "cell": list(intent.target_position),
                    "metadata": {
                        key: list(value) if isinstance(value, tuple) else value
                        for key, value in intent.metadata
                    },
                }
                for intent in resolution.selected
                if intent.reason == "MULTI_ENEMY_CONVERGENCE_SWEEP"
            ],
            "home_force_target": max(
                self.config.home_force_floor,
                self.memory.home_force_high_water,
            ),
            "home_defense_alert_until": self.memory.home_defense_alert_until,
            "home_defense_alert_active": (
                self.memory.home_defense_alert_until >= world.tick
            ),
            "defense_trigger": {
                "global_home_pool": [
                    {
                        "enemy_id": str(enemy.enemy_id),
                        "distance": manhattan(
                            enemy.observed_position, world.core.position
                        ),
                        "reason": (
                            "INSIDE_HOME_ENGAGE"
                            if manhattan(
                                enemy.observed_position, world.core.position
                            ) <= self.config.home_engage_radius
                            else "FOUR_TICK_APPROACH"
                        ),
                    }
                    for enemy in projection.enemies
                    if world.core is not None
                    and enemy.visible_now
                    and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                    and manhattan(enemy.observed_position, world.core.position)
                    <= self.config.home_engage_radius + 4
                ],
                "outer_screen_only": [
                    {
                        "enemy_id": str(enemy.enemy_id),
                        "distance": manhattan(
                            enemy.observed_position, world.core.position
                        ),
                        "reason": "REMOTE_LOCAL_SCREEN",
                    }
                    for enemy in projection.enemies
                    if world.core is not None
                    and enemy.visible_now
                    and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                    and self.config.home_engage_radius + 4
                    < manhattan(enemy.observed_position, world.core.position)
                    <= self.config.home_warning_radius
                ],
            },
            "formation": self._formation_dict(world, resolution),
            "squads": [
                {
                    "vanguard_id": str(squad.vanguard_id),
                    "ranger_id": str(squad.ranger_id),
                    "radius": squad.radius,
                    "sector_index": squad.sector_index,
                    "patrol_anchor": (
                        None
                        if squad.patrol_anchor is None
                        else list(squad.patrol_anchor)
                    ),
                    "support_target": (
                        None
                        if squad.support_target is None
                        else list(squad.support_target)
                    ),
                    "target_assigned_tick": squad.target_assigned_tick,
                    "rendezvous": (
                        None
                        if (
                            lease := self.memory.squad_rendezvous_leases.get(
                                (squad.vanguard_id, squad.ranger_id)
                            )
                        ) is None
                        else {
                            "cell": list(lease.rendezvous),
                            "assigned_tick": lease.assigned_tick,
                            "best_separation": lease.best_separation,
                            "best_route_distance": lease.best_route_distance,
                            "stalled_ticks": lease.stalled_ticks,
                        }
                    ),
                }
                for squad in sorted(
                    self.memory.squad_states.values(),
                    key=lambda item: (
                        item.radius,
                        item.sector_index,
                        item.vanguard_id.bytes,
                    ),
                )
            ],
            "screening_groups": [
                {
                    "target_id": str(group.target_id),
                    "vanguards": [str(item) for item in group.vanguard_ids],
                    "rangers": [str(item) for item in group.ranger_ids],
                    "started_tick": group.started_tick,
                    "last_seen_tick": group.last_seen_tick,
                    "last_distance": group.last_distance,
                    "outward_ticks": group.outward_ticks,
                    "phase": group.phase,
                    "actions": [
                        {
                            "actor_id": str(intent.actor_id),
                            "action": intent.action.value,
                            "reason": intent.reason,
                            "target": (
                                None
                                if intent.target_position is None
                                else list(intent.target_position)
                            ),
                        }
                        for intent in resolution.selected
                        if intent.actor_id
                        in {*group.vanguard_ids, *group.ranger_ids}
                    ],
                }
                for group in sorted(
                    self.memory.screening_groups.values(),
                    key=lambda item: (item.started_tick, item.target_id.bytes),
                )
            ],
            "defense_sector_anchors": {
                key: {"cell": list(value[0]), "assigned_tick": value[1]}
                for key, value in sorted(self.memory.defense_sector_anchors.items())
            },
            "ranger_suppressed_cells": sum(
                feedback.misses >= self.config.ranger_repeat_miss_limit
                for feedback in self.memory.ranger_shot_feedback.values()
            ),
            "ranger_shot_feedback": [
                {
                    "target_id": str(feedback.target_id),
                    "cell": list(feedback.expected_cell),
                    "consecutive_misses": feedback.misses,
                    "suppressed_until": feedback.suppressed_until,
                    "last_evidence_tick": feedback.last_evidence_tick,
                    "release_reason": feedback.release_reason,
                    "last_attempt_tick": feedback.last_attempt_tick,
                }
                for feedback in sorted(
                    self.memory.ranger_shot_feedback.values(),
                    key=lambda item: (item.target_id.bytes, item.expected_cell),
                )
            ],
            "vanguard_suppressed_cells": len(self.memory.vanguard_sweep_feedback),
            "raid": {
                "phase": self.memory.raid_phase,
                "target_id": (
                    None
                    if self.memory.raid_target_id is None
                    else str(self.memory.raid_target_id)
                ),
                "last_position": (
                    None
                    if self.memory.raid_last_position is None
                    else list(self.memory.raid_last_position)
                ),
                "members": [str(item) for item in self.memory.raid_member_ids],
                "return_reason": self.memory.raid_return_reason,
                "handoff_targets": [
                    {
                        "actor_id": str(actor_id),
                        "position": list(position),
                    }
                    for actor_id, position in sorted(
                        self.memory.raid_handoff_targets.items(),
                        key=lambda item: item[0].bytes,
                    )
                ],
                "attempts": [
                    {
                        "target_id": str(target_id),
                        "failed_attempts": attempt.failed_attempts,
                        "last_failure_tick": attempt.last_failure_tick,
                        "last_failure_reason": attempt.last_failure_reason,
                        "last_failure_sighting_tick": (
                            attempt.last_failure_sighting_tick
                        ),
                        "next_pair_count": (
                            self.config.raid_initial_pair_count
                            + attempt.failed_attempts
                            * self.config.raid_escalation_pair_step
                        ),
                    }
                    for target_id, attempt in sorted(
                        self.memory.raid_attempts.items(),
                        key=lambda item: item[0].bytes,
                    )
                ],
                "return_handoffs": [
                    {
                        "actor_id": str(intent.actor_id),
                        "action": intent.action.value,
                        "destination": (
                            None
                            if intent.target_position is None
                            else list(intent.target_position)
                        ),
                        "reason": intent.reason,
                        "released": dict(intent.metadata).get(
                            "released_from_raid", False
                        ),
                    }
                    for intent in resolution.selected
                    if intent.reason.startswith("RAID_RETURN")
                ],
                "containment_mode": self.memory.raid_containment_mode,
                "containment_radius": self.config.raid_containment_radius,
                "peace_home_reserve": self.config.raid_peace_home_reserve,
                "home_force_target": max(
                    self.config.home_force_floor,
                    self.memory.home_force_high_water,
                ),
                "long_range": (
                    None
                    if self.memory.raid_long_range_campaign is None
                    else {
                        "target_id": str(self.memory.raid_long_range_campaign.target_id),
                        "members": [
                            str(item)
                            for item in self.memory.raid_long_range_campaign.member_ids
                        ],
                        "phase": self.memory.raid_long_range_campaign.phase,
                        "started_tick": self.memory.raid_long_range_campaign.started_tick,
                        "route_eta": self.memory.raid_long_range_campaign.route_eta,
                        "search_deadline_tick": (
                            self.memory.raid_long_range_campaign.search_deadline_tick
                        ),
                        "last_position": list(
                            self.memory.raid_long_range_campaign.last_position
                        ),
                        "last_group_distance": (
                            self.memory.raid_long_range_campaign.last_group_distance
                        ),
                        "no_progress_ticks": (
                            self.memory.raid_long_range_campaign.no_progress_ticks
                        ),
                    }
                ),
                "confirmed_nearby_cores": sum(
                    world.tick - intel.last_seen_tick <= self.config.raid_intel_ttl
                    and intel.sighting_count >= self.config.raid_confirmed_sightings
                    and world.core is not None
                    and manhattan(intel.position, world.core.position)
                    <= self.config.raid_containment_radius
                    for intel in self.memory.enemy_core_intel.values()
                ),
            },
        }

    def _formation_dict(
        self,
        world: WorldModel,
        resolution: IntentResolution,
    ) -> dict[str, object]:
        assignment = self.memory.peaceful_formation_assignment
        if assignment is not None and assignment.tick != world.tick:
            assignment = None
        combat_ids = {
            unit.id
            for unit in world.friendlies
            if unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
        }
        waits = [
            intent
            for intent in resolution.selected
            if intent.actor_id in combat_ids and intent.action is IntentAction.WAIT
        ]
        valid_classes = {"TACTICAL_HOLD", "PARTNER_PROGRESS_HOLD"}
        valid_holds = [
            intent
            for intent in waits
            if dict(intent.metadata).get("hold_class") in valid_classes
        ]
        blocked_holds = [
            intent
            for intent in waits
            if dict(intent.metadata).get("hold_class")
            in {"BLOCKED_WAIT", "NO_VIABLE_MOVE"}
            or "ROUTE_BLOCKED" in intent.reason
            or intent.reason == "NO_LEGAL_TASK"
        ]
        denominator = max(1, len(combat_ids))
        return {
            "assignment": (
                None
                if assignment is None
                else {
                    "tick": assignment.tick,
                    "reserved_positions": [
                        list(cell) for cell in assignment.reserved_positions
                    ],
                    "unassigned_squads": [
                        [str(vanguard_id), str(ranger_id)]
                        for vanguard_id, ranger_id in assignment.unassigned_squads
                    ],
                    "bundles": [
                        {
                            "vanguard_id": str(bundle.vanguard_id),
                            "ranger_id": str(bundle.ranger_id),
                            "vanguard_origin": list(bundle.vanguard_origin),
                            "ranger_origin": list(bundle.ranger_origin),
                            "anchor": list(bundle.anchor),
                            "support": list(bundle.support),
                            "vanguard_route_distance": bundle.vanguard_route_distance,
                            "ranger_route_distance": bundle.ranger_route_distance,
                            "vanguard_first_position": (
                                None
                                if bundle.vanguard_first_position is None
                                else list(bundle.vanguard_first_position)
                            ),
                            "ranger_first_position": (
                                None
                                if bundle.ranger_first_position is None
                                else list(bundle.ranger_first_position)
                            ),
                            "score": list(bundle.score),
                        }
                        for bundle in assignment.bundles
                    ],
                    "rejected": [
                        {
                            "vanguard_id": str(vanguard_id),
                            "ranger_id": str(ranger_id),
                            "anchor": list(anchor),
                            "support": list(support),
                            "reason": reason,
                        }
                        for vanguard_id, ranger_id, anchor, support, reason
                        in assignment.rejected
                    ],
                }
            ),
            "leases": [
                {
                    "vanguard_id": str(lease.vanguard_id),
                    "ranger_id": str(lease.ranger_id),
                    "anchor": list(lease.anchor),
                    "support": list(lease.support),
                    "assigned_tick": lease.assigned_tick,
                    "last_evaluated_tick": lease.last_evaluated_tick,
                    "vanguard_best_distance": lease.vanguard_best_distance,
                    "ranger_best_distance": lease.ranger_best_distance,
                    "vanguard_arrived": lease.vanguard_arrived,
                    "ranger_arrived": lease.ranger_arrived,
                    "stalled_ticks": lease.stalled_ticks,
                    "blocked_ticks": lease.blocked_ticks,
                    "partner_hold_ticks": lease.partner_hold_ticks,
                    "partner_progressing": lease.partner_progressing,
                    "last_rejection_reason": lease.last_rejection_reason,
                }
                for lease in sorted(
                    self.memory.squad_formation_leases.values(),
                    key=lambda item: (
                        item.vanguard_id.bytes,
                        item.ranger_id.bytes,
                    ),
                )
            ],
            "move_feedback": [
                {
                    "actor_id": str(feedback.actor_id),
                    "tick": feedback.tick,
                    "action": feedback.action,
                    "reason": feedback.reason,
                    "target": (
                        None
                        if feedback.target_position is None
                        else list(feedback.target_position)
                    ),
                    "rejection_reason": feedback.rejection_reason,
                    "consecutive_blocked_ticks": feedback.consecutive_blocked_ticks,
                    "consecutive_partner_wait_ticks": (
                        feedback.consecutive_partner_wait_ticks
                    ),
                }
                for feedback in sorted(
                    self.memory.formation_move_feedback.values(),
                    key=lambda item: item.actor_id.bytes,
                )
            ],
            "partner_dependencies": [
                {
                    "actor_id": str(feedback.actor_id),
                    "partner_id": str(feedback.partner_id),
                    "tick": feedback.tick,
                    "reason": feedback.reason,
                    "remaining_route_distance": feedback.remaining_route_distance,
                    "resolver_accepted": feedback.resolver_accepted,
                    "wait_ticks": feedback.wait_ticks,
                }
                for feedback in sorted(
                    self.memory.partner_dependency_feedback.values(),
                    key=lambda item: item.actor_id.bytes,
                )
            ],
            "pairing_cooldowns": [
                {
                    "vanguard_id": str(cooldown.vanguard_id),
                    "ranger_id": str(cooldown.ranger_id),
                    "expires_tick": cooldown.expires_tick,
                    "reason": cooldown.reason,
                }
                for cooldown in sorted(
                    self.memory.squad_pairing_cooldowns.values(),
                    key=lambda item: (
                        item.expires_tick,
                        item.vanguard_id.bytes,
                        item.ranger_id.bytes,
                    ),
                )
            ],
            "waits": {
                "combat_units": len(combat_ids),
                "total": len(waits),
                "valid_tactical": len(valid_holds),
                "blocked_or_idle": len(blocked_holds),
                "valid_tactical_percent": round(
                    100 * len(valid_holds) / denominator, 1
                ),
                "blocked_or_idle_percent": round(
                    100 * len(blocked_holds) / denominator, 1
                ),
            },
            "defense_reserve_leases": [
                {
                    "actor_id": str(actor_id),
                    "position": list(value[0]),
                    "assigned_tick": value[1],
                    "role": value[2],
                }
                for actor_id, value in sorted(
                    self.memory.defense_reserve_leases.items(),
                    key=lambda item: item[0].bytes,
                )
            ],
        }

    def _core_safety_dict(
        self,
        evacuation: CoreEvacuationCampaign | None,
    ) -> dict[str, object] | None:
        if evacuation is None:
            return None
        return {
            "evacuation_active": evacuation.active,
            "started_tick": evacuation.started_tick,
            "safe_ticks": evacuation.safe_ticks,
            "last_destination": (
                None
                if evacuation.last_destination is None
                else list(evacuation.last_destination)
            ),
            "reason": evacuation.reason,
            "no_escape_route": evacuation.no_escape_route,
            "move_candidates": [
                {
                    "direction": candidate.direction.value,
                    "destination": list(candidate.destination),
                    "forward_exits": candidate.forward_exits,
                    "local_open": candidate.local_open,
                    "unknown_frontier": candidate.unknown_frontier,
                    "service_exits": candidate.service_exits,
                    "viable": candidate.viable,
                    "rejection_reason": candidate.rejection_reason,
                }
                for candidate in evacuation.candidate_evaluations
            ],
            "recent_combat_loss_ticks": list(self.memory.recent_combat_loss_ticks),
            "recent_combat_losses": [
                {
                    "actor_id": str(record.actor_id),
                    "unit_type": record.unit_type.value,
                    "tick": record.tick,
                    "provenance": record.provenance.value,
                }
                for record in self.memory.recent_combat_losses
            ],
            "strategic_relocation_pending": self.memory.strategic_relocation_pending,
            "strategic_relocation_safe_ticks": self.memory.strategic_relocation_safe_ticks,
            "strategic_relocation_goal": (
                None
                if self.memory.strategic_relocation_goal is None
                else list(self.memory.strategic_relocation_goal)
            ),
        }

    @staticmethod
    def intent_dict(intent: ActionIntent) -> dict[str, object]:
        return {
            "actor_id": None if intent.actor_id is None else str(intent.actor_id),
            "action": intent.action.value,
            "mission": intent.mission.value,
            "priority": intent.priority,
            "target": None if intent.target_position is None else list(intent.target_position),
            "target_id": None if intent.target_id is None else str(intent.target_id),
            "expected_cell": None if intent.expected_cell is None else list(intent.expected_cell),
            "direction": None if intent.direction is None else intent.direction.value,
            "reason": intent.reason,
            "risk": intent.risk,
            "destination_exclusivity": intent.destination_exclusivity.value,
            "metadata": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in intent.metadata
            },
        }

    @staticmethod
    def service_dict(service: CoreServiceQueue) -> dict[str, object]:
        return {
            "service": service.service,
            "service_core_position": (
                None
                if service.service_core_position is None
                else list(service.service_core_position)
            ),
            "admission_id": None if service.admission_id is None else str(service.admission_id),
            "previous_admission_id": (
                None
                if service.previous_admission_id is None
                else str(service.previous_admission_id)
            ),
            "admission_reason": service.admission_reason,
            "release_reason": service.release_reason,
            "depositors": [str(item) for item in service.depositors],
            "ready_depositors": [str(item) for item in service.ready_depositors],
            "approaching_depositors": [str(item) for item in service.approaching_depositors],
            "holding_depositors": [str(item) for item in service.holding_depositors],
            "ready_ticks": [
                {"worker_id": str(worker_id), "tick": tick}
                for worker_id, tick in service.ready_ticks
            ],
            "queue_slots": [
                {"worker_id": str(worker_id), "cell": list(cell)}
                for worker_id, cell in service.queue_slots
            ],
            "overflow_slots": [
                {"worker_id": str(worker_id), "cell": list(cell)}
                for worker_id, cell in service.overflow_slots
            ],
            "scheduled_deposits": [
                {"worker_id": str(worker_id), "tick": tick}
                for worker_id, tick in service.scheduled_deposits
            ],
            "return_reservations": [
                {
                    "worker_id": str(row.worker_id),
                    "route_target": (
                        None if row.route_target is None else list(row.route_target)
                    ),
                    "route_distance": row.route_distance,
                    "first_direction": (
                        None if row.first_direction is None else row.first_direction.value
                    ),
                    "first_position": (
                        None if row.first_position is None else list(row.first_position)
                    ),
                    "earliest_deposit_tick": row.earliest_deposit_tick,
                    "scheduled_deposit_tick": row.scheduled_deposit_tick,
                    "departure_tick": row.departure_tick,
                    "slack_ticks": row.slack_ticks,
                    "status": row.status,
                    "delay_reason": row.delay_reason,
                    "route_mode": row.route_mode,
                    "waypoint": (
                        None if row.waypoint is None else list(row.waypoint)
                    ),
                    "lane_version": row.lane_version,
                    "previous_scheduled_tick": row.previous_scheduled_tick,
                    "schedule_change_reason": row.schedule_change_reason,
                    "schedule_drift": (
                        None
                        if row.previous_scheduled_tick is None
                        or row.scheduled_deposit_tick is None
                        else row.scheduled_deposit_tick
                        - row.previous_scheduled_tick
                    ),
                }
                for row in service.return_reservations
            ],
            "worker_progress": [
                {
                    "worker_id": str(worker_id),
                    "position": list(position),
                    "stalled_ticks": stalled_ticks,
                }
                for worker_id, position, stalled_ticks in service.worker_progress
            ],
            "wounded": [str(item) for item in service.wounded],
            "entrance": None if service.entrance is None else list(service.entrance),
            "queue_cells": [list(cell) for cell in service.queue_cells],
            "exit": None if service.exit_cell is None else list(service.exit_cell),
            "patient_gateway": (
                None
                if service.patient_gateway is None
                else list(service.patient_gateway)
            ),
            "service_windows": [
                {
                    "actor_id": str(window.actor_id),
                    "operation": window.operation,
                    "enter_tick": window.enter_tick,
                    "service_tick": window.service_tick,
                    "exit_tick": window.exit_tick,
                    "gateway": (
                        None if window.gateway is None else list(window.gateway)
                    ),
                    "status": window.status,
                }
                for window in service.service_windows
            ],
            "jobs": [
                {
                    "actor_id": None if job.actor_id is None else str(job.actor_id),
                    "operations": list(job.operations),
                    "phase": job.phase.value,
                    "route_distance": job.route_distance,
                    "first_direction": (
                        None if job.first_direction is None else job.first_direction.value
                    ),
                    "first_position": (
                        None if job.first_position is None else list(job.first_position)
                    ),
                    "gateway": None if job.gateway is None else list(job.gateway),
                    "earliest_service_tick": job.earliest_service_tick,
                    "service_tick": job.service_tick,
                    "exit_tick": job.exit_tick,
                    "priority": job.priority,
                    "ready_since_tick": job.ready_since_tick,
                    "resource_cost": job.resource_cost,
                    "resource_gain": job.resource_gain,
                    "reason": job.reason,
                }
                for job in service.jobs
            ],
            "slot_schedule": (
                None
                if service.slot_schedule is None
                else {
                    "tick": service.slot_schedule.tick,
                    "current_job_id": (
                        None
                        if service.slot_schedule.current_job_id is None
                        else str(service.slot_schedule.current_job_id)
                    ),
                    "next_job_id": (
                        None
                        if service.slot_schedule.next_job_id is None
                        else str(service.slot_schedule.next_job_id)
                    ),
                    "slot_owner_id": (
                        None
                        if service.slot_schedule.slot_owner_id is None
                        else str(service.slot_schedule.slot_owner_id)
                    ),
                    "slot_reserved": service.slot_schedule.slot_reserved,
                    "production_allowed": service.slot_schedule.production_allowed,
                    "spawn_egress": (
                        None
                        if service.slot_schedule.spawn_egress_cell is None
                        else list(service.slot_schedule.spawn_egress_cell)
                    ),
                    "reason": service.slot_schedule.reason,
                }
            ),
            "patient_queue": [
                {
                    "patient_id": str(patient.patient_id),
                    "urgent": patient.urgent,
                    "hp_percent": patient.hp_percent,
                    "eta": patient.eta,
                    "gateway": (
                        None if patient.gateway is None else list(patient.gateway)
                    ),
                    "stalled_ticks": patient.stalled_ticks,
                    "resource_cost": patient.resource_cost,
                    "status": patient.status,
                }
                for patient in service.patient_queue
            ],
            "service_cell_leases": [
                {
                    "cell": list(lease.cell),
                    "purpose": lease.purpose,
                    "owner_id": (
                        None if lease.owner_id is None else str(lease.owner_id)
                    ),
                    "start_tick": lease.start_tick,
                    "end_tick": lease.end_tick,
                    "active": lease.active,
                }
                for lease in service.service_cell_leases
            ],
            "blocking_units": [
                {
                    "unit_id": str(unit_id),
                    "cell": list(cell),
                    "reason": reason,
                }
                for unit_id, cell, reason in service.blocking_units
            ],
            "reschedule_reasons": list(service.reschedule_reasons),
            "core_slot_reserved": service.core_slot_reserved,
            "timeline": (
                None
                if service.timeline is None
                else {
                    "tick": service.timeline.tick,
                    "current_slot_owner": (
                        None
                        if service.timeline.current_slot_owner is None
                        else str(service.timeline.current_slot_owner)
                    ),
                    "current_slot_reserved": service.timeline.current_slot_reserved,
                    "next_service_eta": service.timeline.next_service_eta,
                    "next_service_tick": service.timeline.next_service_tick,
                    "next_release_tick": service.timeline.next_release_tick,
                    "production_allowed": service.timeline.production_allowed,
                    "spawn_egress": (
                        None
                        if service.timeline.spawn_egress_cell is None
                        else list(service.timeline.spawn_egress_cell)
                    ),
                    "reason": service.timeline.reason,
                    "requests": [
                        {
                            "actor_id": (
                                None
                                if request.actor_id is None
                                else str(request.actor_id)
                            ),
                            "operation": request.operation,
                            "eta": request.eta,
                            "occupy_tick": request.occupy_tick,
                            "release_tick": request.release_tick,
                            "priority": request.priority,
                            "resource_cost": request.resource_cost,
                            "resource_gain": request.resource_gain,
                            "gateway": (
                                None
                                if request.gateway is None
                                else list(request.gateway)
                            ),
                        }
                        for request in service.timeline.requests
                    ],
                }
            ),
            "patient_progress": (
                None
                if service.patient_progress is None
                else {
                    "patient_id": str(service.patient_progress.patient_id),
                    "gateway": (
                        None
                        if service.patient_progress.gateway is None
                        else list(service.patient_progress.gateway)
                    ),
                    "started_tick": service.patient_progress.started_tick,
                    "last_position": list(service.patient_progress.last_position),
                    "stalled_ticks": service.patient_progress.stalled_ticks,
                    "entry_distance": service.patient_progress.entry_distance,
                }
            ),
            "reserved_resources": service.reserved_resources,
            "paused_reason": service.paused_reason,
            "lane_lease": (
                None
                if service.lane_lease is None
                else {
                    "core_id": str(service.lane_lease.core_id),
                    "core_position": list(service.lane_lease.core_position),
                    "entrance": (
                        None
                        if service.lane_lease.entrance is None
                        else list(service.lane_lease.entrance)
                    ),
                    "queue_cells": [
                        list(cell) for cell in service.lane_lease.queue_cells
                    ],
                    "exit": (
                        None
                        if service.lane_lease.exit_cell is None
                        else list(service.lane_lease.exit_cell)
                    ),
                    "established_tick": service.lane_lease.established_tick,
                    "version": service.lane_lease.version,
                    "invalidation_reason": service.lane_lease.invalidation_reason,
                }
            ),
            "lane_replan_reason": service.lane_replan_reason,
            "liveness_indicators": list(service.liveness_indicators),
        }

    @staticmethod
    def fire_mission_dict(mission: FireMission) -> dict[str, object]:
        return {
            "target_id": str(mission.target_id),
            "target_type": (
                mission.target_kind if mission.target_type is None else mission.target_type.value
            ),
            "target_kind": mission.target_kind,
            "urgent": mission.urgent,
            "confidence": mission.confidence,
            "candidate_cells": [list(cell) for cell in mission.candidate_cells],
            "required_hits": mission.required_hits,
            "prediction_mode": mission.prediction_mode,
            "candidate_roles": list(mission.candidate_roles),
            "evidence": list(mission.evidence),
            "stationary_ticks": (
                next(
                    (
                        int(item.removeprefix("STATIONARY_TICKS="))
                        for item in mission.evidence
                        if item.startswith("STATIONARY_TICKS=")
                    ),
                    0,
                )
            ),
            "split_fire": mission.split_fire,
            "shooters": [str(shooter) for shooter in mission.assigned_shooters],
            "assignments": [
                {"shooter_id": str(shooter), "cell": list(cell)}
                for shooter, cell in mission.assignments
            ],
        }
