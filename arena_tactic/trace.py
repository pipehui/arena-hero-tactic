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
    ) -> dict[str, object]:
        return {
            "schema_version": STRATEGY_LOG_SCHEMA_VERSION,
            "mode": "GLOBAL_MAP_SURVIVAL_ECONOMY",
            "outcomes": self._outcomes(),
            "tasks": [self.intent_dict(intent) for intent in resolution.selected],
            "resolution": self._resolution_dict(resolution),
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
        return {
            "workers": workers,
            "worker_target": ceil(
                world.population * self.config.worker_ratio_percent / 100
            ),
            "stockpile_active": (
                world.population >= self.config.population_stockpile_threshold
            ),
            "stockpile_population_threshold": self.config.population_stockpile_threshold,
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
            "storage_saturated": self.memory.storage_saturated,
            "storage_headroom": max(0, world.resource_capacity - world.resources),
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
                    "mode": state.phase.value,
                    "target": None if state.target is None else list(state.target),
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
                }
                for worker_id, state in sorted(
                    self.memory.worker_escape_states.items(),
                    key=lambda item: item[0].bytes,
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
            if intent.reason == "ADVANCE_TO_DYNAMIC_FIRE_LINE"
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
                }
                for intent in dynamic_fire_lines
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
                        "candidate_cells": [list(cell) for cell in task.candidate_cells],
                        "cost": list(task.cost),
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
            "ranger_suppressed_cells": len(self.memory.ranger_shot_feedback),
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
                "containment_mode": self.memory.raid_containment_mode,
                "containment_radius": self.config.raid_containment_radius,
                "peace_home_reserve": self.config.raid_peace_home_reserve,
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
            "recent_combat_loss_ticks": list(self.memory.recent_combat_loss_ticks),
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
            "split_fire": mission.split_fire,
            "shooters": [str(shooter) for shooter in mission.assigned_shooters],
            "assignments": [
                {"shooter_id": str(shooter), "cell": list(cell)}
                for shooter, cell in mission.assignments
            ],
        }
