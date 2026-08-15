from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from uuid import UUID

from arena_hero import CoreState, Position, UnitType

from .config import DEFAULT_CONFIG, TacticConfig
from .geometry import DIRECTION_ORDER, add_direction, cardinal_neighbors, diamond, manhattan, vision_is_clear
from .projection import attack_cells
from .rules import CORE_VISION_RADIUS, UNIT_VISION_RADIUS
from .schema import EXPLORATION_MEMORY_SCHEMA_VERSION
from .state import TacticMemory
from .models import (
    CrisisForceBaseline,
    EnemyCoreControlZone,
    EnemyCoreControlLevel,
    EnemyCoreIntel,
    EnemyTrack,
    LongRangeRaidCampaign,
    RaidAttemptMemory,
    RaidConfirmationLease,
    RaidDistanceBand,
    RaidReconMission,
    ResourceWorkOrder,
    ResourceSearchLease,
    SiegeApproachPlan,
    ThreatHeatCell,
    WorkerScoutPhase,
    WorkerScoutState,
)
from .world import _decayed_threat_heat, _update_threat_heat


EXPLORATION_MEMORY_FILENAME = "balanced_tactic_memory.json"


def _position(value: object) -> Position | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value[:2])
    ):
        return value[0], value[1]
    return None


def _uuid(value: object) -> UUID | None:
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _triples(value: object):
    if not isinstance(value, list):
        return
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        position = _position(item)
        amount = item[2]
        if position is not None and isinstance(amount, int) and not isinstance(amount, bool):
            yield position, amount


class ExplorationMemoryStore:
    """Atomic durable-world checkpoint with best-effort legacy migration.

    Service admission and SDK-facing action state are intentionally not
    durable.  Value-only Worker scout leases are durable so a watchdog restart
    does not turn every explorer into an unassigned waiter.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        config: TacticConfig = DEFAULT_CONFIG,
        save_interval_ticks: int = 16,
    ) -> None:
        if save_interval_ticks <= 0:
            raise ValueError("save_interval_ticks must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / EXPLORATION_MEMORY_FILENAME
        self.config = config
        self.save_interval_ticks = save_interval_ticks
        self._last_saved_tick: int | None = None
        self.restored_through_tick: int | None = None
        self.restored_visit_count = 0

    def load(self) -> TacticMemory:
        memory = TacticMemory(home_force_high_water=self.config.home_force_floor)
        saved_tick: int | None = None
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                saved_tick = self._restore(memory, payload)
            except (OSError, ValueError, TypeError, KeyError) as error:
                warnings.warn(
                    f"Ignoring invalid exploration checkpoint: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                memory = TacticMemory(home_force_high_water=self.config.home_force_floor)
                saved_tick = None
        replay_tick = self._catch_up(memory, after_tick=saved_tick)
        latest = replay_tick if replay_tick is not None else saved_tick
        if latest is not None:
            active_resource_targets = {
                order.target for order in memory.resource_work_orders.values()
            }
            for cell, seen_tick in tuple(memory.resource_memory.items()):
                if (
                    latest - seen_tick > self.config.resource_memory_ttl
                    and cell not in active_resource_targets
                ):
                    memory.resource_memory.pop(cell, None)
            for cell, record in tuple(memory.threat_heat.items()):
                if record.score(latest) <= 0:
                    memory.threat_heat.pop(cell, None)
            for enemy_id, track in tuple(memory.enemy_tracks.items()):
                if latest - track.last_seen_tick > self.config.enemy_track_ttl:
                    memory.enemy_tracks.pop(enemy_id, None)
            for core_id, intel in tuple(memory.enemy_core_intel.items()):
                active_long_range = (
                    memory.raid_long_range_campaign is not None
                    and memory.raid_long_range_campaign.target_id == core_id
                    and latest <= memory.raid_long_range_campaign.search_deadline_tick
                )
                if latest - intel.last_seen_tick > self.config.enemy_core_control_ttl and not active_long_range:
                    memory.enemy_core_intel.pop(core_id, None)
                    memory.enemy_core_control_zones.pop(core_id, None)
        self.restored_through_tick = latest
        self.restored_visit_count = sum(memory.visit_counts.values())
        self._last_saved_tick = saved_tick
        return memory

    def save(self, memory: TacticMemory, *, tick: int, force: bool = False) -> bool:
        if (
            not force
            and self._last_saved_tick is not None
            and tick - self._last_saved_tick < self.save_interval_ticks
        ):
            return True
        payload = {
            "schema_version": EXPLORATION_MEMORY_SCHEMA_VERSION,
            "saved_tick": tick,
            "core_id": None if memory.core_id is None else str(memory.core_id),
            "core_position": None if memory.core_position is None else list(memory.core_position),
            "core_position_history": [list(cell) for cell in memory.core_position_history],
            "known_obstacles": [list(cell) for cell in sorted(memory.known_obstacles)],
            "known_passable": [list(cell) for cell in sorted(memory.known_passable)],
            "cell_last_visible": self._counter_rows(memory.cell_last_visible),
            "worker_cell_last_visible": self._counter_rows(memory.worker_cell_last_visible),
            "visit_counts": self._counter_rows(memory.visit_counts),
            "congestion_counts": self._counter_rows(memory.congestion_counts),
            "resource_memory": self._counter_rows(memory.resource_memory),
            "resource_seen_count": self._counter_rows(memory.resource_seen_count),
            "resource_harvest_count": self._counter_rows(memory.resource_harvest_count),
            "resource_work_orders": [
                {
                    "worker_id": str(worker_id),
                    "target": list(order.target),
                    "assigned_tick": order.assigned_tick,
                    "last_confirmed_tick": order.last_confirmed_tick,
                    "last_route_distance": order.last_route_distance,
                    "stalled_ticks": order.stalled_ticks,
                    "failures": order.failures,
                }
                for worker_id, order in sorted(
                    memory.resource_work_orders.items(),
                    key=lambda item: item[0].bytes,
                )
            ],
            "resource_search_leases": [
                {
                    "worker_id": str(worker_id),
                    "direction_slot": lease.direction_slot,
                    "target": None if lease.target is None else list(lease.target),
                    "waypoint": None if lease.waypoint is None else list(lease.waypoint),
                    "assigned_tick": lease.assigned_tick,
                    "last_position": list(lease.last_position),
                    "last_route_distance": lease.last_route_distance,
                    "stalled_ticks": lease.stalled_ticks,
                    "route_version": lease.route_version,
                    "blocked_edge": (
                        None
                        if lease.blocked_edge is None
                        else [list(lease.blocked_edge[0]), list(lease.blocked_edge[1])]
                    ),
                    "backoff_until": lease.backoff_until,
                    "information_gain": lease.information_gain,
                    "visible_gain": lease.visible_gain,
                    "overlap_cells": lease.overlap_cells,
                }
                for worker_id, lease in sorted(
                    memory.resource_search_leases.items(),
                    key=lambda item: item[0].bytes,
                )
            ],
            "threat_heat": [
                [
                    cell[0],
                    cell[1],
                    record.risk,
                    record.updated_tick,
                    record.expires_tick,
                    record.source,
                ]
                for cell, record in sorted(memory.threat_heat.items())
                if record.score(tick) > 0
            ],
            "enemy_tracks": [
                {
                    "enemy_id": str(enemy_id),
                    "unit_type": track.unit_type.value,
                    "samples": [
                        [sample_tick, position[0], position[1]]
                        for sample_tick, position in track.samples
                    ],
                    "last_seen_tick": track.last_seen_tick,
                }
                for enemy_id, track in sorted(
                    memory.enemy_tracks.items(),
                    key=lambda item: item[0].bytes,
                )
                if tick - track.last_seen_tick <= self.config.enemy_track_ttl
            ],
            "enemy_core_intel": [
                {
                    "enemy_id": str(core_id),
                    "position": list(intel.position),
                    "hp": intel.hp,
                    "shield": intel.shield,
                    "state": intel.state.value,
                    "destination": (
                        None if intel.destination is None else list(intel.destination)
                    ),
                    "last_seen_tick": intel.last_seen_tick,
                    "sighting_count": intel.sighting_count,
                    "lifetime_sightings": intel.lifetime_sightings,
                    "confirmation_sightings": intel.confirmation_sightings,
                    "confirmation_window_start_tick": (
                        intel.confirmation_window_start_tick
                    ),
                }
                for core_id, intel in sorted(
                    memory.enemy_core_intel.items(),
                    key=lambda item: item[0].bytes,
                )
                if (
                    tick - intel.last_seen_tick <= self.config.enemy_core_control_ttl
                    or memory.raid_long_range_campaign is not None
                    and memory.raid_long_range_campaign.target_id == core_id
                    and tick <= memory.raid_long_range_campaign.search_deadline_tick
                )
            ],
            "enemy_core_control_zones": [
                {
                    "core_id": str(core_id),
                    "center": list(zone.center),
                    "exclusion_radius": zone.exclusion_radius,
                    "clear_radius": zone.clear_radius,
                    "last_seen_tick": zone.last_seen_tick,
                    "expires_tick": zone.expires_tick,
                    "control_level": zone.control_level.value,
                }
                for core_id, zone in sorted(
                    memory.enemy_core_control_zones.items(),
                    key=lambda item: item[0].bytes,
                )
                if zone.expires_tick is None or tick <= zone.expires_tick
            ],
            "worker_scout_states": [
                {
                    "worker_id": str(worker_id),
                    "slot": state.slot,
                    "sector_index": state.sector_index,
                    "stage": state.stage,
                    "phase": state.phase.value,
                    "target": None if state.target is None else list(state.target),
                    "assigned_tick": state.assigned_tick,
                    "best_route_cost": state.best_route_cost,
                    "stalled_ticks": state.stalled_ticks,
                    "backoff_until": state.backoff_until,
                    "last_scan_tick": state.last_scan_tick,
                    "reachable_candidates": state.reachable_candidates,
                    "scout_eligible": state.scout_eligible,
                    "coverage_version": state.coverage_version,
                    "lease_until": state.lease_until,
                    "visible_gain": state.visible_gain,
                    "overlap_cells": state.overlap_cells,
                }
                for worker_id, state in sorted(
                    memory.worker_scout_states.items(),
                    key=lambda item: item[0].bytes,
                )
            ],
            "last_congestion_decay_tick": memory.last_congestion_decay_tick,
            "opening_complete": memory.opening_complete,
            "home_force_high_water": memory.home_force_high_water,
            "crisis_force_baseline": (
                None
                if memory.crisis_force_baseline is None
                else {
                    "vanguards": memory.crisis_force_baseline.vanguards,
                    "rangers": memory.crisis_force_baseline.rangers,
                    "started_tick": memory.crisis_force_baseline.started_tick,
                    "phase": memory.crisis_force_baseline.phase,
                    "safe_ticks": memory.crisis_force_baseline.safe_ticks,
                }
            ),
            "long_range_raid_campaign": (
                None
                if memory.raid_long_range_campaign is None
                else {
                    "target_id": str(memory.raid_long_range_campaign.target_id),
                    "member_ids": [
                        str(member_id)
                        for member_id in memory.raid_long_range_campaign.member_ids
                    ],
                    "phase": memory.raid_long_range_campaign.phase,
                    "started_tick": memory.raid_long_range_campaign.started_tick,
                    "route_eta": memory.raid_long_range_campaign.route_eta,
                    "search_deadline_tick": memory.raid_long_range_campaign.search_deadline_tick,
                    "last_position": list(memory.raid_long_range_campaign.last_position),
                    "last_group_distance": memory.raid_long_range_campaign.last_group_distance,
                    "no_progress_ticks": memory.raid_long_range_campaign.no_progress_ticks,
                }
            ),
            "raid_attempts": [
                {
                    "core_id": str(core_id),
                    "failed_attempts": attempt.failed_attempts,
                    "last_failure_tick": attempt.last_failure_tick,
                    "last_failure_reason": attempt.last_failure_reason,
                    "last_failure_sighting_tick": attempt.last_failure_sighting_tick,
                }
                for core_id, attempt in sorted(
                    memory.raid_attempts.items(),
                    key=lambda item: item[0].bytes,
                )
            ],
            "raid_state": (
                None
                if memory.raid_phase == "IDLE"
                and not memory.raid_member_ids
                and memory.raid_confirmation_lease is None
                and memory.raid_recon_mission is None
                else {
                    "target_id": (
                        None
                        if memory.raid_target_id is None
                        else str(memory.raid_target_id)
                    ),
                    "last_seen_tick": memory.raid_last_seen_tick,
                    "last_position": (
                        None
                        if memory.raid_last_position is None
                        else list(memory.raid_last_position)
                    ),
                    "member_ids": [str(item) for item in memory.raid_member_ids],
                    "phase": memory.raid_phase,
                    "interrupted_tick": memory.raid_interrupted_tick,
                    "containment_mode": memory.raid_containment_mode,
                    "distance_band": (
                        None
                        if memory.raid_distance_band is None
                        else memory.raid_distance_band.value
                    ),
                    "siege_approach": (
                        None
                        if memory.raid_siege_approach is None
                        else {
                            "target_id": str(memory.raid_siege_approach.target_id),
                            "target_position": list(
                                memory.raid_siege_approach.target_position
                            ),
                            "distance_band": (
                                memory.raid_siege_approach.distance_band.value
                            ),
                            "vanguard_positions": [
                                list(cell)
                                for cell in memory.raid_siege_approach.vanguard_positions
                            ],
                            "ranger_positions": [
                                list(cell)
                                for cell in memory.raid_siege_approach.ranger_positions
                            ],
                            "route_eta": memory.raid_siege_approach.route_eta,
                        }
                    ),
                    "confirmation_lease": (
                        None
                        if memory.raid_confirmation_lease is None
                        else {
                            "target_id": str(memory.raid_confirmation_lease.target_id),
                            "observer_id": str(memory.raid_confirmation_lease.observer_id),
                            "first_seen_tick": memory.raid_confirmation_lease.first_seen_tick,
                            "expires_tick": memory.raid_confirmation_lease.expires_tick,
                        }
                    ),
                    "recon_mission": (
                        None
                        if memory.raid_recon_mission is None
                        else {
                            "target_id": str(memory.raid_recon_mission.target_id),
                            "member_ids": [
                                str(item)
                                for item in memory.raid_recon_mission.member_ids
                            ],
                            "last_position": list(memory.raid_recon_mission.last_position),
                            "started_tick": memory.raid_recon_mission.started_tick,
                            "last_seen_tick": memory.raid_recon_mission.last_seen_tick,
                            "no_progress_ticks": memory.raid_recon_mission.no_progress_ticks,
                            "last_group_distance": (
                                memory.raid_recon_mission.last_group_distance
                            ),
                        }
                    ),
                    "return_reason": memory.raid_return_reason,
                    "handoff_targets": [
                        [str(actor_id), list(position)]
                        for actor_id, position in sorted(
                            memory.raid_handoff_targets.items(),
                            key=lambda item: item[0].bytes,
                        )
                    ],
                }
            ),
            "strategic_relocation_pending": memory.strategic_relocation_pending,
            "strategic_relocation_safe_ticks": memory.strategic_relocation_safe_ticks,
            "strategic_relocation_goal": (
                None
                if memory.strategic_relocation_goal is None
                else list(memory.strategic_relocation_goal)
            ),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError as error:
            warnings.warn(
                f"Could not save exploration checkpoint: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        self._last_saved_tick = tick
        return True

    @staticmethod
    def _counter_rows(values) -> list[list[int]]:
        return [[cell[0], cell[1], int(value)] for cell, value in sorted(values.items())]

    def _restore(self, memory: TacticMemory, payload: object) -> int | None:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint root must be an object")
        schema = payload.get("schema_version")
        if not isinstance(schema, int) or not 1 <= schema <= EXPLORATION_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported exploration checkpoint schema")
        saved_tick = payload.get("saved_tick")
        if not isinstance(saved_tick, int) or isinstance(saved_tick, bool):
            saved_tick = None

        memory.core_id = _uuid(payload.get("core_id"))
        memory.core_position = _position(payload.get("core_position"))
        if schema >= 8:
            memory.core_position_history = tuple(
                cell
                for raw in payload.get("core_position_history", [])
                if (cell := _position(raw)) is not None
            )[-4:]
            if not memory.core_position_history and memory.core_position is not None:
                memory.core_position_history = (memory.core_position,)
        obstacle_key = "known_obstacles" if schema >= 6 else "obstacles"
        for value in payload.get(obstacle_key, []):
            if (cell := _position(value)) is not None:
                memory.known_obstacles.add(cell)
        for value in payload.get("known_passable", []):
            if (cell := _position(value)) is not None:
                memory.known_passable.add(cell)

        fields = (
            ("cell_last_visible", memory.cell_last_visible),
            ("worker_cell_last_visible", memory.worker_cell_last_visible),
            ("visit_counts", memory.visit_counts),
            ("congestion_counts", memory.congestion_counts),
            ("resource_memory", memory.resource_memory),
            ("resource_seen_count", memory.resource_seen_count),
            ("resource_harvest_count", memory.resource_harvest_count),
        )
        for key, target in fields:
            old_key = key
            if schema < 6 and key == "visit_counts":
                old_key = "patrol_visits"
            for cell, value in _triples(payload.get(old_key, [])):
                target[cell] = value

        if schema >= 18:
            for raw in payload.get("resource_work_orders", []):
                if not isinstance(raw, dict):
                    continue
                worker_id = _uuid(raw.get("worker_id"))
                target = _position(raw.get("target"))
                assigned_tick = raw.get("assigned_tick")
                last_confirmed_tick = raw.get("last_confirmed_tick")
                last_route_distance = raw.get("last_route_distance")
                stalled_ticks = raw.get("stalled_ticks", 0)
                failures = raw.get("failures", 0)
                if (
                    worker_id is None
                    or target is None
                    or not isinstance(assigned_tick, int)
                    or isinstance(assigned_tick, bool)
                    or not isinstance(last_confirmed_tick, int)
                    or isinstance(last_confirmed_tick, bool)
                    or not isinstance(stalled_ticks, int)
                    or isinstance(stalled_ticks, bool)
                    or stalled_ticks < 0
                    or not isinstance(failures, int)
                    or isinstance(failures, bool)
                    or failures < 0
                ):
                    continue
                if not isinstance(last_route_distance, int) or isinstance(
                    last_route_distance, bool
                ):
                    last_route_distance = None
                memory.resource_work_orders[worker_id] = ResourceWorkOrder(
                    worker_id=worker_id,
                    target=target,
                    assigned_tick=assigned_tick,
                    last_confirmed_tick=last_confirmed_tick,
                    last_route_distance=last_route_distance,
                    stalled_ticks=stalled_ticks,
                    failures=failures,
                )
                memory.resource_memory.setdefault(target, last_confirmed_tick)

        if schema >= 19:
            for raw in payload.get("resource_search_leases", []):
                if not isinstance(raw, dict):
                    continue
                worker_id = _uuid(raw.get("worker_id"))
                target = _position(raw.get("target"))
                waypoint = _position(raw.get("waypoint"))
                last_position = _position(raw.get("last_position"))
                integer_fields = {
                    key: raw.get(key, 0)
                    for key in (
                        "direction_slot",
                        "assigned_tick",
                        "stalled_ticks",
                        "route_version",
                        "backoff_until",
                        "information_gain",
                        "visible_gain",
                        "overlap_cells",
                    )
                }
                if (
                    worker_id is None
                    or last_position is None
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in integer_fields.values()
                    )
                    or integer_fields["direction_slot"] < 0
                    or integer_fields["stalled_ticks"] < 0
                    or integer_fields["route_version"] < 0
                ):
                    continue
                last_route_distance = raw.get("last_route_distance")
                if not isinstance(last_route_distance, int) or isinstance(
                    last_route_distance, bool
                ):
                    last_route_distance = None
                blocked_edge = None
                raw_edge = raw.get("blocked_edge")
                if isinstance(raw_edge, (list, tuple)) and len(raw_edge) == 2:
                    edge_start = _position(raw_edge[0])
                    edge_end = _position(raw_edge[1])
                    if edge_start is not None and edge_end is not None:
                        blocked_edge = edge_start, edge_end
                memory.resource_search_leases[worker_id] = ResourceSearchLease(
                    worker_id=worker_id,
                    direction_slot=(
                        integer_fields["direction_slot"]
                        % self.config.resource_search_direction_slots
                    ),
                    target=target,
                    waypoint=waypoint,
                    assigned_tick=integer_fields["assigned_tick"],
                    last_position=last_position,
                    last_route_distance=last_route_distance,
                    stalled_ticks=integer_fields["stalled_ticks"],
                    route_version=integer_fields["route_version"],
                    blocked_edge=blocked_edge,
                    backoff_until=integer_fields["backoff_until"],
                    information_gain=integer_fields["information_gain"],
                    visible_gain=integer_fields["visible_gain"],
                    overlap_cells=integer_fields["overlap_cells"],
                )

        if schema >= 9:
            for raw in payload.get("threat_heat", []):
                if not isinstance(raw, (list, tuple)) or len(raw) < 6:
                    continue
                cell = _position(raw)
                risk, updated_tick, expires_tick, source = raw[2:6]
                if (
                    cell is None
                    or not isinstance(risk, int)
                    or isinstance(risk, bool)
                    or risk <= 0
                    or not isinstance(updated_tick, int)
                    or isinstance(updated_tick, bool)
                    or not isinstance(expires_tick, int)
                    or isinstance(expires_tick, bool)
                    or expires_tick <= updated_tick
                    or not isinstance(source, str)
                ):
                    continue
                record = ThreatHeatCell(
                    position=cell,
                    risk=risk,
                    updated_tick=updated_tick,
                    expires_tick=expires_tick,
                    source=source,
                )
                if saved_tick is None or record.score(saved_tick) > 0:
                    memory.threat_heat[cell] = record

        if schema >= 11:
            for raw in payload.get("enemy_tracks", []):
                if not isinstance(raw, dict):
                    continue
                enemy_id = _uuid(raw.get("enemy_id"))
                try:
                    unit_type = UnitType(raw.get("unit_type"))
                except (TypeError, ValueError):
                    continue
                samples: list[tuple[int, Position]] = []
                for sample in raw.get("samples", []):
                    if (
                        not isinstance(sample, (list, tuple))
                        or len(sample) < 3
                        or not isinstance(sample[0], int)
                        or isinstance(sample[0], bool)
                    ):
                        continue
                    position = _position(sample[1:3])
                    if position is not None:
                        samples.append((sample[0], position))
                last_seen_tick = raw.get("last_seen_tick")
                if (
                    enemy_id is None
                    or not samples
                    or not isinstance(last_seen_tick, int)
                    or isinstance(last_seen_tick, bool)
                    or samples[-1][0] != last_seen_tick
                ):
                    continue
                memory.enemy_tracks[enemy_id] = EnemyTrack(
                    id=enemy_id,
                    unit_type=unit_type,
                    samples=tuple(samples[-4:]),
                    last_seen_tick=last_seen_tick,
                )
            for raw in payload.get("enemy_core_intel", []):
                if not isinstance(raw, dict):
                    continue
                core_id = _uuid(raw.get("enemy_id"))
                position = _position(raw.get("position"))
                destination = _position(raw.get("destination"))
                try:
                    state = CoreState(raw.get("state"))
                except (TypeError, ValueError):
                    continue
                hp = raw.get("hp")
                shield = raw.get("shield")
                last_seen_tick = raw.get("last_seen_tick")
                sighting_count = raw.get("sighting_count")
                values = (hp, shield, last_seen_tick, sighting_count)
                if (
                    core_id is None
                    or position is None
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in values
                    )
                    or hp < 0
                    or shield < 0
                    or sighting_count <= 0
                ):
                    continue
                memory.enemy_core_intel[core_id] = EnemyCoreIntel(
                    id=core_id,
                    position=position,
                    hp=hp,
                    shield=shield,
                    state=state,
                    destination=destination,
                    last_seen_tick=last_seen_tick,
                    sighting_count=sighting_count,
                    lifetime_sightings=(
                        raw.get("lifetime_sightings")
                        if isinstance(raw.get("lifetime_sightings"), int)
                        else sighting_count
                    ),
                    confirmation_sightings=(
                        raw.get("confirmation_sightings")
                        if isinstance(raw.get("confirmation_sightings"), int)
                        else sighting_count
                    ),
                    confirmation_window_start_tick=(
                        raw.get("confirmation_window_start_tick")
                        if isinstance(raw.get("confirmation_window_start_tick"), int)
                        else last_seen_tick
                    ),
                )

        if schema >= 10:
            for raw in payload.get("worker_scout_states", []):
                if not isinstance(raw, dict):
                    continue
                worker_id = _uuid(raw.get("worker_id"))
                target = _position(raw.get("target"))
                try:
                    phase = WorkerScoutPhase(raw.get("phase"))
                except (TypeError, ValueError):
                    continue
                integer_fields = {
                    key: raw.get(key)
                    for key in (
                        "slot",
                        "sector_index",
                        "stage",
                        "assigned_tick",
                        "stalled_ticks",
                        "backoff_until",
                        "reachable_candidates",
                    )
                }
                if (
                    worker_id is None
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in integer_fields.values()
                    )
                    or integer_fields["slot"] < 0
                    or not 0 <= integer_fields["sector_index"] < 8
                    or integer_fields["stage"] < 0
                    or integer_fields["stalled_ticks"] < 0
                    or integer_fields["reachable_candidates"] < 0
                ):
                    continue
                best_route_cost = raw.get("best_route_cost")
                if not isinstance(best_route_cost, int) or isinstance(best_route_cost, bool):
                    best_route_cost = None
                last_scan_tick = raw.get("last_scan_tick")
                if not isinstance(last_scan_tick, int) or isinstance(last_scan_tick, bool):
                    last_scan_tick = None
                slot = integer_fields["slot"]
                band_count = len(self.config.exploration_sector_radii)
                if schema >= 17:
                    stage = integer_fields["stage"] % band_count
                    sector_index = (
                        integer_fields["sector_index"]
                        % self.config.exploration_sector_count
                    )
                else:
                    stage = slot % band_count
                    sector_index = (slot // band_count) % 8
                if (
                    target is not None
                    and memory.core_position is not None
                    and manhattan(target, memory.core_position)
                    > self.config.exploration_sector_radii[-1]
                ):
                    target = None
                memory.worker_scout_states[worker_id] = WorkerScoutState(
                    worker_id=worker_id,
                    slot=slot,
                    sector_index=sector_index,
                    stage=stage,
                    phase=(
                        WorkerScoutPhase.SECTOR_SCOUT
                        if schema < 15
                        else phase
                    ),
                    target=target,
                    assigned_tick=integer_fields["assigned_tick"],
                    best_route_cost=best_route_cost,
                    stalled_ticks=integer_fields["stalled_ticks"],
                    backoff_until=integer_fields["backoff_until"],
                    last_scan_tick=last_scan_tick,
                    reachable_candidates=integer_fields["reachable_candidates"],
                    scout_eligible=(
                        bool(raw.get("scout_eligible")) if schema >= 17 else True
                    ),
                    coverage_version=(
                        raw.get("coverage_version")
                        if schema >= 17
                        and isinstance(raw.get("coverage_version"), int)
                        and not isinstance(raw.get("coverage_version"), bool)
                        else 0
                    ),
                    lease_until=(
                        raw.get("lease_until")
                        if schema >= 17
                        and isinstance(raw.get("lease_until"), int)
                        and not isinstance(raw.get("lease_until"), bool)
                        else 0
                    ),
                    visible_gain=(
                        raw.get("visible_gain")
                        if schema >= 17
                        and isinstance(raw.get("visible_gain"), int)
                        and not isinstance(raw.get("visible_gain"), bool)
                        else 0
                    ),
                    overlap_cells=(
                        raw.get("overlap_cells")
                        if schema >= 17
                        and isinstance(raw.get("overlap_cells"), int)
                        and not isinstance(raw.get("overlap_cells"), bool)
                        else 0
                    ),
                )

        decay_tick = payload.get("last_congestion_decay_tick")
        if schema >= 8 and isinstance(decay_tick, int) and not isinstance(decay_tick, bool):
            memory.last_congestion_decay_tick = decay_tick

        high_water = payload.get("home_force_high_water")
        if not isinstance(high_water, int):
            # Accept the short-lived nested form emitted by one schema-5 build.
            defense = payload.get("defense_strength")
            if isinstance(defense, dict):
                high_water = defense.get(
                    "home_force_high_water",
                    defense.get("hostile_force_high_water", defense.get("high_water")),
                )
        if isinstance(high_water, int) and high_water > 0:
            memory.home_force_high_water = max(self.config.home_force_floor, high_water)
        if schema >= 12:
            raw_baseline = payload.get("crisis_force_baseline")
            if isinstance(raw_baseline, dict):
                vanguards = raw_baseline.get("vanguards")
                rangers = raw_baseline.get("rangers")
                started_tick = raw_baseline.get("started_tick")
                phase = raw_baseline.get("phase")
                safe_ticks = raw_baseline.get("safe_ticks", 0)
                if (
                    isinstance(vanguards, int)
                    and not isinstance(vanguards, bool)
                    and vanguards >= 0
                    and isinstance(rangers, int)
                    and not isinstance(rangers, bool)
                    and rangers >= 0
                    and (started_tick is None or isinstance(started_tick, int))
                    and phase in {"SAFE", "ACTIVE", "REBUILD"}
                    and isinstance(safe_ticks, int)
                    and not isinstance(safe_ticks, bool)
                    and safe_ticks >= 0
                ):
                    memory.crisis_force_baseline = CrisisForceBaseline(
                        vanguards=vanguards,
                        rangers=rangers,
                        started_tick=started_tick,
                        phase=phase,
                        safe_ticks=safe_ticks,
                    )
        if schema >= 13:
            raw_campaign = payload.get("long_range_raid_campaign")
            if isinstance(raw_campaign, dict):
                target_id = _uuid(raw_campaign.get("target_id"))
                member_ids = tuple(
                    member_id
                    for raw in raw_campaign.get("member_ids", [])
                    if (member_id := _uuid(raw)) is not None
                )
                last_position = _position(raw_campaign.get("last_position"))
                phase = raw_campaign.get("phase")
                integer_values = {
                    key: raw_campaign.get(key)
                    for key in (
                        "started_tick",
                        "route_eta",
                        "search_deadline_tick",
                        "no_progress_ticks",
                    )
                }
                last_group_distance = raw_campaign.get("last_group_distance")
                if (
                    target_id is not None
                    and member_ids
                    and last_position is not None
                    and isinstance(phase, str)
                    and all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in integer_values.values()
                    )
                    and integer_values["route_eta"] >= 0
                    and integer_values["no_progress_ticks"] >= 0
                    and (
                        last_group_distance is None
                        or isinstance(last_group_distance, int)
                        and not isinstance(last_group_distance, bool)
                        and last_group_distance >= 0
                    )
                ):
                    memory.raid_long_range_campaign = LongRangeRaidCampaign(
                        target_id=target_id,
                        member_ids=member_ids,
                        phase=phase,
                        started_tick=integer_values["started_tick"],
                        route_eta=integer_values["route_eta"],
                        search_deadline_tick=integer_values["search_deadline_tick"],
                        last_position=last_position,
                        last_group_distance=last_group_distance,
                        no_progress_ticks=integer_values["no_progress_ticks"],
                    )
                    memory.raid_target_id = target_id
                    memory.raid_member_ids = member_ids
                    memory.raid_phase = phase
                    memory.raid_last_position = last_position
        if schema >= 14:
            for raw in payload.get("enemy_core_control_zones", []):
                if not isinstance(raw, dict):
                    continue
                core_id = _uuid(raw.get("core_id"))
                center = _position(raw.get("center"))
                exclusion = raw.get("exclusion_radius")
                clear = raw.get("clear_radius")
                last_seen = raw.get("last_seen_tick")
                expires = raw.get("expires_tick")
                if (
                    core_id is None
                    or center is None
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in (exclusion, clear, last_seen)
                    )
                    or exclusion < 0
                    or clear <= exclusion
                    or (
                        expires is not None
                        and (
                            not isinstance(expires, int)
                            or isinstance(expires, bool)
                            or expires < last_seen
                        )
                    )
                ):
                    continue
                memory.enemy_core_control_zones[core_id] = EnemyCoreControlZone(
                    core_id=core_id,
                    center=center,
                    exclusion_radius=exclusion,
                    clear_radius=clear,
                    last_seen_tick=last_seen,
                    visible_now=False,
                    expires_tick=expires,
                    control_level=(
                        EnemyCoreControlLevel(raw.get("control_level"))
                        if schema >= 16
                        and raw.get("control_level")
                        in {item.value for item in EnemyCoreControlLevel}
                        else EnemyCoreControlLevel.HARD
                    ),
                )
            for raw in payload.get("raid_attempts", []):
                if not isinstance(raw, dict):
                    continue
                core_id = _uuid(raw.get("core_id"))
                failed = raw.get("failed_attempts")
                failure_tick = raw.get("last_failure_tick")
                failure_reason = raw.get("last_failure_reason")
                sighting_tick = raw.get("last_failure_sighting_tick")
                if (
                    core_id is None
                    or not isinstance(failed, int)
                    or isinstance(failed, bool)
                    or failed < 0
                    or any(
                        value is not None
                        and (not isinstance(value, int) or isinstance(value, bool))
                        for value in (failure_tick, sighting_tick)
                    )
                    or failure_reason is not None
                    and not isinstance(failure_reason, str)
                ):
                    continue
                memory.raid_attempts[core_id] = RaidAttemptMemory(
                    core_id=core_id,
                    failed_attempts=failed,
                    last_failure_tick=failure_tick,
                    last_failure_reason=failure_reason,
                    last_failure_sighting_tick=sighting_tick,
                )
            raw_raid = payload.get("raid_state")
            if isinstance(raw_raid, dict):
                target_id = _uuid(raw_raid.get("target_id"))
                member_ids = tuple(
                    actor_id
                    for value in raw_raid.get("member_ids", [])
                    if (actor_id := _uuid(value)) is not None
                )
                phase = raw_raid.get("phase")
                last_seen_tick = raw_raid.get("last_seen_tick")
                interrupted_tick = raw_raid.get("interrupted_tick")
                if (
                    isinstance(phase, str)
                    and all(
                        value is None
                        or isinstance(value, int) and not isinstance(value, bool)
                        for value in (last_seen_tick, interrupted_tick)
                    )
                ):
                    memory.raid_target_id = target_id
                    memory.raid_last_seen_tick = last_seen_tick
                    memory.raid_last_position = _position(raw_raid.get("last_position"))
                    memory.raid_member_ids = member_ids
                    memory.raid_phase = phase
                    memory.raid_interrupted_tick = interrupted_tick
                    memory.raid_containment_mode = bool(
                        raw_raid.get("containment_mode", False)
                    )
                    if schema >= 16:
                        try:
                            memory.raid_distance_band = RaidDistanceBand(
                                raw_raid.get("distance_band")
                            )
                        except (TypeError, ValueError):
                            memory.raid_distance_band = None
                        raw_approach = raw_raid.get("siege_approach")
                        if isinstance(raw_approach, dict):
                            approach_target_id = _uuid(raw_approach.get("target_id"))
                            approach_position = _position(
                                raw_approach.get("target_position")
                            )
                            try:
                                approach_band = RaidDistanceBand(
                                    raw_approach.get("distance_band")
                                )
                            except (TypeError, ValueError):
                                approach_band = None
                            vanguard_positions = tuple(
                                cell
                                for value in raw_approach.get("vanguard_positions", [])
                                if (cell := _position(value)) is not None
                            )
                            ranger_positions = tuple(
                                cell
                                for value in raw_approach.get("ranger_positions", [])
                                if (cell := _position(value)) is not None
                            )
                            route_eta = raw_approach.get("route_eta")
                            if (
                                approach_target_id is not None
                                and approach_position is not None
                                and approach_band is not None
                                and vanguard_positions
                                and ranger_positions
                                and isinstance(route_eta, int)
                                and route_eta >= 0
                            ):
                                memory.raid_siege_approach = SiegeApproachPlan(
                                    target_id=approach_target_id,
                                    target_position=approach_position,
                                    distance_band=approach_band,
                                    vanguard_positions=vanguard_positions,
                                    ranger_positions=ranger_positions,
                                    route_eta=route_eta,
                                )
                        raw_confirmation = raw_raid.get("confirmation_lease")
                        if isinstance(raw_confirmation, dict):
                            confirmation_target = _uuid(raw_confirmation.get("target_id"))
                            observer_id = _uuid(raw_confirmation.get("observer_id"))
                            first_seen = raw_confirmation.get("first_seen_tick")
                            expires_tick = raw_confirmation.get("expires_tick")
                            if (
                                confirmation_target is not None
                                and observer_id is not None
                                and isinstance(first_seen, int)
                                and isinstance(expires_tick, int)
                            ):
                                memory.raid_confirmation_lease = RaidConfirmationLease(
                                    target_id=confirmation_target,
                                    observer_id=observer_id,
                                    first_seen_tick=first_seen,
                                    expires_tick=expires_tick,
                                )
                        raw_recon = raw_raid.get("recon_mission")
                        if isinstance(raw_recon, dict):
                            recon_target = _uuid(raw_recon.get("target_id"))
                            recon_members = tuple(
                                actor_id
                                for value in raw_recon.get("member_ids", [])
                                if (actor_id := _uuid(value)) is not None
                            )
                            recon_position = _position(raw_recon.get("last_position"))
                            started = raw_recon.get("started_tick")
                            seen = raw_recon.get("last_seen_tick")
                            no_progress = raw_recon.get("no_progress_ticks", 0)
                            last_group_distance = raw_recon.get("last_group_distance")
                            if (
                                recon_target is not None
                                and recon_members
                                and recon_position is not None
                                and all(
                                    isinstance(value, int)
                                    for value in (started, seen, no_progress)
                                )
                                and (
                                    last_group_distance is None
                                    or isinstance(last_group_distance, int)
                                    and last_group_distance >= 0
                                )
                            ):
                                memory.raid_recon_mission = RaidReconMission(
                                    target_id=recon_target,
                                    member_ids=recon_members,
                                    last_position=recon_position,
                                    started_tick=started,
                                    last_seen_tick=seen,
                                    no_progress_ticks=no_progress,
                                    last_group_distance=last_group_distance,
                                )
                    reason = raw_raid.get("return_reason")
                    memory.raid_return_reason = reason if isinstance(reason, str) else None
                    for row in raw_raid.get("handoff_targets", []):
                        if not isinstance(row, (list, tuple)) or len(row) < 2:
                            continue
                        actor_id = _uuid(row[0])
                        position = _position(row[1])
                        if actor_id is not None and position is not None:
                            memory.raid_handoff_targets[actor_id] = position
        memory.opening_complete = bool(payload.get("opening_complete", False))
        memory.strategic_relocation_pending = bool(
            payload.get("strategic_relocation_pending", False)
        )
        relocation_safe_ticks = payload.get("strategic_relocation_safe_ticks", 0)
        if isinstance(relocation_safe_ticks, int) and relocation_safe_ticks >= 0:
            memory.strategic_relocation_safe_ticks = relocation_safe_ticks
        memory.strategic_relocation_goal = _position(
            payload.get("strategic_relocation_goal")
        )
        # Legacy schemas persisted controller-relative missions and Core
        # service choreography.  They cannot be trusted after reconnecting to
        # a newer complete Turn, so current schemas deliberately migrate only
        # the durable map/economy fields above.
        return saved_tick

    def _catch_up(self, memory: TacticMemory, *, after_tick: int | None) -> int | None:
        # Logs can overlap after a reconnect or a watchdog restart.  Replaying
        # them in filename order used to count a duplicated Turn twice and,
        # worse, could leave ``memory.last_tick`` pointing at an older Turn
        # when files were copied or renamed.  Keep the last complete state for
        # each Tick and apply the authoritative states in game order.
        states_by_tick: dict[int, dict] = {}
        for path in sorted(self.directory.glob("arena_hero_*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        if record.get("record_type") != "turn":
                            continue
                        tick = record.get("tick")
                        if not isinstance(tick, int) or (after_tick is not None and tick <= after_tick):
                            continue
                        state = record.get("state")
                        if not isinstance(state, dict):
                            continue
                        states_by_tick[tick] = state
            except OSError:
                continue
        for tick in sorted(states_by_tick):
            self._learn_logged_state(memory, tick, states_by_tick[tick])
        return max(states_by_tick, default=None)

    def _learn_logged_state(self, memory: TacticMemory, tick: int, state: dict) -> None:
        visible_resources: set[Position] = set()
        sources: list[tuple[Position, int, bool]] = []
        visible_hostiles: list[tuple[UUID, Position, UnitType]] = []
        for obj in state.get("objects", []):
            if not isinstance(obj, dict):
                continue
            kind = obj.get("kind")
            if kind == "OBSTACLE":
                for raw in obj.get("positions", []):
                    if (cell := _position(raw)) is not None:
                        memory.known_obstacles.add(cell)
            elif kind == "RESOURCE":
                for raw in obj.get("positions", []):
                    if (cell := _position(raw)) is not None:
                        visible_resources.add(cell)
                        memory.resource_memory[cell] = tick
            elif kind == "CORE" and obj.get("controlled") is True:
                core_id = _uuid(obj.get("id"))
                core_position = _position(obj.get("position"))
                if core_id is not None and core_position is not None:
                    if memory.core_id != core_id:
                        memory.reset_for_core(core_id, core_position)
                        memory.home_force_high_water = self.config.home_force_floor
                    else:
                        memory.core_id = core_id
                elif core_id is not None:
                    memory.core_id = core_id
                if core_position is not None:
                    if memory.core_position != core_position:
                        memory.core_position_history = (
                            *memory.core_position_history,
                            core_position,
                        )[-4:]
                    memory.core_position = core_position
                    memory.visit_counts[core_position] += 1
                    sources.append((core_position, CORE_VISION_RADIUS, False))
            elif kind == "CORE" and obj.get("controlled") is False:
                core_id = _uuid(obj.get("id"))
                position = _position(obj.get("position"))
                destination = _position(obj.get("destination"))
                try:
                    core_state = CoreState(obj.get("state"))
                except (TypeError, ValueError):
                    continue
                hp = obj.get("hp")
                shield = obj.get("shield")
                if (
                    core_id is None
                    or position is None
                    or not isinstance(hp, int)
                    or isinstance(hp, bool)
                    or not isinstance(shield, int)
                    or isinstance(shield, bool)
                ):
                    continue
                previous = memory.enemy_core_intel.get(core_id)
                unique_sighting = previous is None or tick > previous.last_seen_tick
                within_window = bool(
                    previous is not None
                    and tick - previous.last_seen_tick
                    <= self.config.raid_confirmation_window_ticks
                )
                lifetime_sightings = (
                    1
                    if previous is None
                    else previous.lifetime_sightings + int(unique_sighting)
                )
                confirmation_sightings = (
                    1
                    if not within_window
                    else previous.confirmation_sightings + int(unique_sighting)
                )
                memory.enemy_core_intel[core_id] = EnemyCoreIntel(
                    id=core_id,
                    position=position,
                    hp=hp,
                    shield=shield,
                    state=core_state,
                    destination=destination,
                    last_seen_tick=tick,
                    sighting_count=confirmation_sightings,
                    lifetime_sightings=lifetime_sightings,
                    confirmation_sightings=confirmation_sightings,
                    confirmation_window_start_tick=(
                        tick
                        if not within_window
                        else previous.confirmation_window_start_tick
                        or previous.last_seen_tick
                    ),
                )
            elif kind == "UNIT" and obj.get("controlled") is True:
                if (cell := _position(obj.get("position"))) is not None:
                    memory.visit_counts[cell] += 1
                    memory.known_passable.add(cell)
                    try:
                        unit_type = UnitType(obj.get("unit_type"))
                    except (TypeError, ValueError):
                        continue
                    sources.append(
                        (
                            cell,
                            UNIT_VISION_RADIUS[unit_type],
                            unit_type is UnitType.WORKER,
                        )
                    )
            elif kind == "UNIT" and obj.get("controlled") is False:
                enemy_id = _uuid(obj.get("id"))
                position = _position(obj.get("position"))
                try:
                    unit_type = UnitType(obj.get("unit_type"))
                except (TypeError, ValueError):
                    continue
                if enemy_id is not None and position is not None:
                    previous = memory.enemy_tracks.get(enemy_id)
                    sample = tick, position
                    if previous is None or tick > previous.last_seen_tick + 1:
                        samples = (sample,)
                    elif tick == previous.last_seen_tick:
                        samples = (*previous.samples[:-1], sample)
                    else:
                        samples = (*previous.samples, sample)[-4:]
                    memory.enemy_tracks[enemy_id] = EnemyTrack(
                        id=enemy_id,
                        unit_type=unit_type,
                        samples=samples,
                        last_seen_tick=tick,
                    )
                    if unit_type in {UnitType.VANGUARD, UnitType.RANGER}:
                        visible_hostiles.append((enemy_id, position, unit_type))
        obstacles = frozenset(memory.known_obstacles)
        visible_cells: set[Position] = set()
        for origin, radius, from_worker in sources:
            for cell in diamond(origin, radius):
                if not vision_is_clear(origin, cell, obstacles):
                    continue
                visible_cells.add(cell)
                memory.cell_last_visible[cell] = tick
                if from_worker:
                    memory.worker_cell_last_visible[cell] = tick
                if cell not in obstacles:
                    memory.known_passable.add(cell)
        for cell in tuple(memory.resource_memory):
            if cell in visible_cells and cell not in visible_resources:
                memory.resource_memory.pop(cell, None)
        self._learn_logged_threat_heat(
            memory,
            tick,
            visible_hostiles,
            state.get("events", []),
        )
        for enemy_id, track in tuple(memory.enemy_tracks.items()):
            if tick - track.last_seen_tick > self.config.enemy_track_ttl:
                memory.enemy_tracks.pop(enemy_id, None)
        for core_id, intel in tuple(memory.enemy_core_intel.items()):
            active_long_range = (
                memory.raid_long_range_campaign is not None
                and memory.raid_long_range_campaign.target_id == core_id
                and tick <= memory.raid_long_range_campaign.search_deadline_tick
            )
            if (
                tick - intel.last_seen_tick > self.config.enemy_core_control_ttl
                and not active_long_range
            ):
                memory.enemy_core_intel.pop(core_id, None)
                memory.enemy_core_control_zones.pop(core_id, None)
        memory.last_tick = tick

    def _learn_logged_threat_heat(
        self,
        memory: TacticMemory,
        tick: int,
        visible_hostiles: list[tuple[UUID, Position, UnitType]],
        events: object,
    ) -> None:
        obstacles = frozenset(memory.known_obstacles)
        for _, position, unit_type in visible_hostiles:
            current_zone = {position, *attack_cells(position, unit_type, obstacles)}
            for cell in current_zone:
                _update_threat_heat(
                    memory,
                    position=cell,
                    risk=self.config.threat_heat_visible_risk,
                    tick=tick,
                    ttl=self.config.threat_heat_visible_ttl,
                    source="VISIBLE_ATTACK_ZONE",
                )
            possible = {position}
            possible.update(
                add_direction(position, direction)
                for direction in DIRECTION_ORDER
                if add_direction(position, direction) not in obstacles
            )
            projected: set[Position] = set()
            for candidate in possible:
                projected.add(candidate)
                projected.update(attack_cells(candidate, unit_type, obstacles))
            for cell in projected - current_zone:
                _update_threat_heat(
                    memory,
                    position=cell,
                    risk=self.config.threat_heat_projected_risk,
                    tick=tick,
                    ttl=self.config.threat_heat_projected_ttl,
                    source="PROJECTED_ATTACK_ZONE",
                )
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                position = _position(event.get("position"))
                event_type = event.get("event_type")
                if position is None or event_type not in {
                    "UNIT_DAMAGED",
                    "UNIT_DESTROYED",
                }:
                    continue
                destroyed = event_type == "UNIT_DESTROYED"
                center_risk = (
                    self.config.threat_heat_destroyed_risk
                    if destroyed
                    else self.config.threat_heat_damage_risk
                )
                neighbor_risk = (
                    self.config.threat_heat_destroyed_neighbor_risk
                    if destroyed
                    else self.config.threat_heat_damage_neighbor_risk
                )
                ttl = (
                    self.config.threat_heat_destroyed_ttl
                    if destroyed
                    else self.config.threat_heat_damage_ttl
                )
                _update_threat_heat(
                    memory,
                    position=position,
                    risk=center_risk,
                    tick=tick,
                    ttl=ttl,
                    source=str(event_type),
                )
                for _, neighbor in cardinal_neighbors(position):
                    _update_threat_heat(
                        memory,
                        position=neighbor,
                        risk=neighbor_risk,
                        tick=tick,
                        ttl=ttl,
                        source=f"{event_type}_NEARBY",
                    )
        _decayed_threat_heat(
            memory,
            tick=tick,
            cell_limit=self.config.threat_heat_cell_limit,
        )
