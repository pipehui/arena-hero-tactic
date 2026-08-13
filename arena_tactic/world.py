from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from uuid import UUID

from arena_hero import CoreView, Position, Turn, UnitType, UnitView

from .config import DEFAULT_CONFIG, TacticConfig
from .geometry import (
    DIRECTION_ORDER,
    add_direction,
    cardinal_neighbors,
    diamond,
    manhattan,
    unit_attack_cells,
    vision_is_clear,
)
from .models import (
    BeaconSnapshot,
    CoreSnapshot,
    EnemyCoreSnapshot,
    EnemyTrack,
    EnemyCoreIntel,
    EntitySnapshot,
    WorldModel,
    ShotFeedback,
    MoveFailure,
    MissionState,
    ResourceIntel,
    ThreatHeatCell,
    UnitMission,
    VisionSource,
)
from .state import TacticMemory
from .rules import CORE_VISION_RADIUS, UNIT_MAX_HP, UNIT_VISION_RADIUS


def _unit_snapshot(view: UnitView, *, controlled: bool) -> EntitySnapshot:
    return EntitySnapshot(
        id=view.id,
        position=view.position,
        hp=view.hp,
        unit_type=view.unit_type,
        cargo=int(view.cargo or 0) if controlled else 0,
        controlled=controlled,
    )


@dataclass(frozen=True, slots=True)
class _ObservationFrame:
    visible_cells: frozenset[Position]
    worker_visible_cells: frozenset[Position]
    sources: tuple[VisionSource, ...]
    coverage: tuple[tuple[Position, tuple[UUID, ...]], ...]


def _visible_cells(
    turn: Turn,
    obstacles: frozenset[Position],
) -> _ObservationFrame:
    raw_sources: list[tuple[UUID, str, UnitType | None, Position, int]] = []
    if turn.core is not None:
        raw_sources.append(
            (
                turn.core.id,
                "CORE",
                None,
                turn.core.position,
                CORE_VISION_RADIUS,
            )
        )
    raw_sources.extend(
        (
            unit.id,
            unit.unit_type.value,
            unit.unit_type,
            unit.position,
            UNIT_VISION_RADIUS[unit.unit_type],
        )
        for unit in turn.units
    )
    visible: set[Position] = set()
    worker_visible: set[Position] = set()
    sources: list[VisionSource] = []
    coverage: defaultdict[Position, set[UUID]] = defaultdict(set)
    for actor_id, actor_kind, unit_type, origin, radius in raw_sources:
        cells = tuple(
            sorted(
                {
            cell
            for cell in diamond(origin, radius)
            if vision_is_clear(origin, cell, obstacles)
                }
            )
        )
        visible.update(cells)
        if unit_type is UnitType.WORKER:
            worker_visible.update(cells)
        for cell in cells:
            coverage[cell].add(actor_id)
        sources.append(
            VisionSource(
                actor_id=actor_id,
                actor_kind=actor_kind,
                unit_type=unit_type,
                position=origin,
                radius=radius,
                visible_cells=cells,
            )
        )
    return _ObservationFrame(
        visible_cells=frozenset(visible),
        worker_visible_cells=frozenset(worker_visible),
        sources=tuple(
            sorted(sources, key=lambda source: source.actor_id.bytes)
        ),
        coverage=tuple(
            (
                cell,
                tuple(sorted(observer_ids, key=lambda observer_id: observer_id.bytes)),
            )
            for cell, observer_ids in sorted(coverage.items())
        ),
    )


def _attack_cells(
    enemy: EntitySnapshot,
    obstacles: frozenset[Position],
) -> set[Position]:
    return set(unit_attack_cells(enemy.position, enemy.unit_type, obstacles))


def _sync_events(
    turn: Turn,
    memory: TacticMemory,
    config: TacticConfig,
) -> None:
    depleted: set = set()
    for event in turn.events:
        memory.event_counts[event.event_type] += 1
        if event.event_type == "SWEEP_RESOLVED":
            hits = (event.values or {}).get("targets_hit", 0)
            if isinstance(hits, int) and hits > 0:
                memory.event_counts["SWEEP_TARGET_HITS"] += hits
        if event.event_type in {"SHOT_HIT", "SHOT_MISSED"} and event.actor_id is not None:
            plan = memory.last_ranger_shots.get(event.actor_id)
            if plan is not None:
                key = plan.target_id, plan.expected_cell
                if event.event_type == "SHOT_HIT":
                    memory.ranger_shot_feedback[key] = ShotFeedback(
                        target_id=plan.target_id,
                        expected_cell=plan.expected_cell,
                        misses=0,
                        suppressed_until=turn.tick,
                        last_evidence_tick=turn.tick,
                        release_reason="SHOT_HIT",
                    )
                else:
                    previous = memory.ranger_shot_feedback.get(key)
                    misses = 1 if previous is None else previous.misses + 1
                    memory.ranger_shot_feedback[key] = ShotFeedback(
                        target_id=plan.target_id,
                        expected_cell=plan.expected_cell,
                        misses=misses,
                        suppressed_until=turn.tick,
                        last_evidence_tick=turn.tick,
                        release_reason=None,
                    )
        if event.event_type == "SWEEP_RESOLVED" and event.actor_id is not None:
            plan = memory.last_vanguard_sweeps.get(event.actor_id)
            if plan is not None:
                key = plan.shooter_id, plan.expected_cell
                values = event.values or {}
                hits = values.get("targets_hit", 0)
                if isinstance(hits, int) and hits > 0:
                    memory.vanguard_sweep_feedback.pop(key, None)
                else:
                    previous = memory.vanguard_sweep_feedback.get(key)
                    misses = 1 if previous is None else previous.misses + 1
                    memory.vanguard_sweep_feedback[key] = ShotFeedback(
                        target_id=plan.target_id,
                        expected_cell=plan.expected_cell,
                        misses=misses,
                        suppressed_until=turn.tick,
                        last_evidence_tick=turn.tick,
                        release_reason=None,
                    )
        if event.event_type == "DEPOSIT_SUCCEEDED" and event.actor_id is not None:
            # The delivery completes the old resource work order.  Keep the
            # Worker's stable scout slot, but require a fresh target while it
            # is still on the Core so CLEAR_CORE becomes the first scout step.
            memory.unit_missions.pop(event.actor_id, None)
            memory.worker_task_progress.pop(event.actor_id, None)
            memory.service_egress_worker_ids.add(event.actor_id)
            scout = memory.worker_scout_states.get(event.actor_id)
            if scout is not None:
                memory.worker_scout_states[event.actor_id] = replace(
                    scout,
                    target=None,
                    best_route_cost=None,
                    stalled_ticks=0,
                    assigned_tick=turn.tick,
                )
        elif (
            event.event_type == "DESTRUCTION_PARTICIPATION"
            and event.reason_code == "CORE"
            and event.target_id is not None
        ):
            memory.enemy_core_intel.pop(event.target_id, None)
            destroyed_zone = memory.enemy_core_control_zones.pop(
                event.target_id,
                None,
            )
            for worker_id, lease in tuple(memory.worker_disengage_leases.items()):
                if lease.core_id == event.target_id:
                    memory.worker_disengage_leases.pop(worker_id, None)
            if destroyed_zone is not None:
                still_controlled: set[Position] = set()
                for zone in memory.enemy_core_control_zones.values():
                    still_controlled.update(
                        diamond(zone.center, zone.exclusion_radius)
                    )
                released = set(
                    diamond(
                        destroyed_zone.center,
                        destroyed_zone.exclusion_radius,
                    )
                ) - still_controlled
                for target in released:
                    memory.target_backoff_until.pop(target, None)
                for key in tuple(memory.worker_resource_backoff):
                    if key[1] in released:
                        memory.worker_resource_backoff.pop(key, None)
            if memory.counter_siege_target_id == event.target_id:
                memory.counter_siege_target_id = None
                memory.counter_siege_last_seen_tick = None
                memory.counter_siege_last_position = None
                memory.counter_siege_member_ids = ()
                memory.counter_siege_reserve_ids = ()
                memory.counter_siege_phase = "IDLE"
            if memory.raid_target_id == event.target_id:
                memory.raid_target_id = None
                memory.raid_last_seen_tick = None
                memory.raid_last_position = None
                memory.raid_member_ids = ()
                memory.raid_phase = "IDLE"
                memory.raid_long_range_campaign = None
        elif event.event_type == "HARVEST_SUCCEEDED":
            if event.position is not None:
                memory.resource_harvest_count[event.position] += 1
                memory.resource_memory.pop(event.position, None)
        elif event.event_type == "HARVEST_FAILED" and event.reason_code in {
            "NOT_RESOURCE_CELL",
            "RESOURCE_DEPLETED",
        }:
            if event.position is not None:
                memory.resource_memory.pop(event.position, None)
            if event.actor_id is not None:
                depleted.add(event.actor_id)
                mission = memory.unit_missions.get(event.actor_id)
                if mission is not None and mission.target is not None:
                    memory.resource_memory.pop(mission.target, None)
                    memory.unit_missions.pop(event.actor_id, None)
                    memory.worker_task_progress.pop(event.actor_id, None)
        elif event.event_type in {
            "UNIT_MOVE_FAILED",
            "CORE_MOVE_FAILED",
            "CORE_MOVE_START_FAILED",
        }:
            # Resolution events report the unchanged *origin* for a failed
            # move.  The attempted Core destination is the one selected on
            # the previous Tick, not ``event.position``.
            destination = memory.last_core_move_destination
            if destination is not None and event.event_type.startswith("CORE"):
                memory.failed_core_destinations[destination] = (
                    turn.tick + config.core_move_failure_ttl
                )
            elif event.event_type == "UNIT_MOVE_FAILED" and event.actor_id is not None:
                attempt = memory.last_move_attempts.get(event.actor_id)
                if attempt is not None and attempt.tick == event.tick:
                    memory.failed_unit_moves[event.actor_id] = MoveFailure(
                        destination=attempt.destination,
                        expires_tick=turn.tick + config.unit_move_failure_ttl,
                        reason=event.reason_code or "UNKNOWN",
                    )
                    memory.congestion_counts[attempt.destination] += 2
        elif event.event_type == "UNIT_MOVE_SUCCEEDED" and event.actor_id is not None:
            memory.failed_unit_moves.pop(event.actor_id, None)
        elif event.event_type == "CORE_MOVE_SUCCEEDED":
            memory.last_core_move_destination = event.position
        elif event.event_type == "CORE_DAMAGED":
            # A reconnect/respawn Turn can still carry terminal events for the
            # previous Core.  Those events must not start a relocation campaign
            # for the newly spawned Core.
            if (
                turn.core is not None
                and event.target_id is not None
                and event.target_id != turn.core.id
            ):
                continue
            values = event.values or {}
            hp_damage = values.get("hp_damage", 0)
            if isinstance(hp_damage, int) and hp_damage > 0:
                memory.strategic_relocation_pending = True
                memory.strategic_relocation_safe_ticks = 0
                memory.strategic_relocation_goal = None
    memory.last_depleted_workers = frozenset(depleted)


def _sync_memory(
    turn: Turn,
    memory: TacticMemory,
    config: TacticConfig,
) -> tuple[_ObservationFrame, dict[Position, int], dict[Position, int]]:
    if turn.core is None:
        memory.last_tick = turn.tick
        frame = _visible_cells(turn, frozenset(turn.obstacle_cells))
        return frame, {}, {
            cell: record.score(turn.tick)
            for cell, record in memory.threat_heat.items()
            if record.score(turn.tick) > 0
        }

    _sync_core_identity(turn, memory, config)
    frame = _sync_map_memory(turn, memory, config)
    _sync_friendly_memory(turn, memory, config)
    visible_enemy_units = _sync_enemy_memory(turn, memory, config)
    danger = _sync_danger_memory(
        turn,
        memory,
        config,
        visible_enemy_units,
    )
    threat_heat = _sync_threat_heat(
        turn,
        memory,
        config,
        visible_enemy_units,
    )
    _prune_tactical_timers(turn.tick, memory)
    memory.last_tick = turn.tick
    return frame, danger, threat_heat


def _sync_core_identity(
    turn: Turn,
    memory: TacticMemory,
    config: TacticConfig,
) -> None:
    assert turn.core is not None

    if memory.core_id != turn.core.id:
        memory.reset_for_core(turn.core.id, turn.core.position)
        memory.home_force_high_water = config.home_force_floor
    else:
        if memory.core_position != turn.core.position:
            memory.core_position_history = (
                *memory.core_position_history,
                turn.core.position,
            )[-4:]
        memory.core_position = turn.core.position

    _sync_events(turn, memory, config)


def _sync_map_memory(
    turn: Turn,
    memory: TacticMemory,
    config: TacticConfig,
) -> _ObservationFrame:
    visible_obstacles = frozenset(turn.obstacle_cells)
    # Obstacles are permanent.  A wall learned by any earlier friendly still
    # occludes every current contributor even when that wall itself is outside
    # the server's union view this Tick.
    memory.known_obstacles.update(visible_obstacles)
    frame = _visible_cells(turn, frozenset(memory.known_obstacles))
    visible = frame.visible_cells
    worker_visible = frame.worker_visible_cells
    for cell in visible:
        memory.cell_last_visible[cell] = turn.tick
        if cell not in visible_obstacles:
            memory.known_passable.add(cell)
    for cell in worker_visible:
        memory.worker_cell_last_visible[cell] = turn.tick
    memory.known_passable.difference_update(memory.known_obstacles)

    visible_resources = set(turn.resource_cells)
    for cell in visible_resources:
        if cell not in memory.resource_memory:
            memory.resource_seen_count[cell] += 1
        memory.resource_memory[cell] = turn.tick
    for cell in tuple(memory.resource_memory):
        if cell in visible and cell not in visible_resources:
            memory.resource_memory.pop(cell, None)
        elif turn.tick - memory.resource_memory[cell] > config.resource_memory_ttl:
            memory.resource_memory.pop(cell, None)
    return frame


def _sync_friendly_memory(
    turn: Turn,
    memory: TacticMemory,
    config: TacticConfig,
) -> None:

    living_ids = {unit.id for unit in turn.units}
    current_combat_ids = {
        unit.id
        for unit in turn.units
        if unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
    }
    if memory.last_tick is not None:
        losses = memory.last_combat_unit_ids - current_combat_ids
        if losses:
            memory.recent_combat_loss_ticks = (
                *memory.recent_combat_loss_ticks,
                *(turn.tick for _ in losses),
            )
    memory.recent_combat_loss_ticks = tuple(
        tick
        for tick in memory.recent_combat_loss_ticks
        if turn.tick - tick < 8
    )
    memory.last_combat_unit_ids = current_combat_ids
    for unit_id in tuple(memory.unit_missions):
        if unit_id not in living_ids:
            memory.unit_missions.pop(unit_id, None)
    for unit_id in tuple(memory.position_history):
        if unit_id not in living_ids:
            memory.position_history.pop(unit_id, None)
            memory.last_positions.pop(unit_id, None)
    for unit_id in tuple(memory.worker_escape_states):
        if unit_id not in living_ids:
            memory.worker_escape_states.pop(unit_id, None)
    for unit_id in tuple(memory.worker_disengage_leases):
        if unit_id not in living_ids:
            memory.worker_disengage_leases.pop(unit_id, None)
    for unit_id in tuple(memory.worker_scout_states):
        if unit_id not in living_ids:
            memory.worker_scout_states.pop(unit_id, None)
    for unit_id in tuple(memory.worker_task_progress):
        if unit_id not in living_ids:
            memory.worker_task_progress.pop(unit_id, None)
    for key in tuple(memory.worker_resource_backoff):
        if key[0] not in living_ids:
            memory.worker_resource_backoff.pop(key, None)
    service_cells = {
        *(
            cell
            for cell in (
                memory.core_position,
                memory.service_entrance,
                memory.service_exit_cell,
            )
            if cell is not None
        ),
        *memory.service_queue_cells,
    }
    units_by_id = {unit.id: unit for unit in turn.units}
    for unit_id in tuple(memory.service_egress_worker_ids):
        unit = units_by_id.get(unit_id)
        if unit is None or unit.position not in service_cells:
            memory.service_egress_worker_ids.discard(unit_id)
    for unit_id in tuple(memory.manual_move_leases):
        if unit_id not in living_ids or turn.tick > memory.manual_move_leases[unit_id].expires_tick:
            memory.manual_move_leases.pop(unit_id, None)
    for unit_id, failure in tuple(memory.failed_unit_moves.items()):
        if unit_id not in living_ids or turn.tick > failure.expires_tick:
            memory.failed_unit_moves.pop(unit_id, None)
    for unit_id in tuple(memory.partner_dependency_feedback):
        if unit_id not in living_ids:
            memory.partner_dependency_feedback.pop(unit_id, None)
    for unit in turn.units:
        existing_mission = memory.unit_missions.get(unit.id)
        if (
            unit.unit_type is UnitType.WORKER
            and unit.hp < UNIT_MAX_HP[UnitType.WORKER]
        ):
            memory.unit_missions[unit.id] = MissionState(
                UnitMission.RECOVER,
                turn.core.position if turn.core is not None else None,
                (
                    existing_mission.assigned_tick
                    if existing_mission is not None
                    and existing_mission.mission is UnitMission.RECOVER
                    else turn.tick
                ),
            )
        elif (
            existing_mission is not None
            and existing_mission.mission is UnitMission.RECOVER
        ):
            memory.unit_missions.pop(unit.id, None)
        history = (*memory.position_history.get(unit.id, ()), unit.position)
        memory.position_history[unit.id] = history[-config.loop_history_length :]
        memory.last_positions[unit.id] = unit.position
        memory.visit_counts[unit.position] += 1
        memory.congestion_counts[unit.position] += 1
    if (
        memory.last_congestion_decay_tick is None
        or turn.tick - memory.last_congestion_decay_tick >= config.congestion_decay_ticks
    ):
        for cell, count in tuple(memory.congestion_counts.items()):
            reduced = count // 2
            if reduced:
                memory.congestion_counts[cell] = reduced
            else:
                memory.congestion_counts.pop(cell, None)
        memory.last_congestion_decay_tick = turn.tick


def _sync_enemy_memory(
    turn: Turn,
    memory: TacticMemory,
    config: TacticConfig,
) -> tuple[UnitView, ...]:

    visible_enemy_units = tuple(
        enemy for enemy in turn.visible_enemies if isinstance(enemy, UnitView)
    )
    visible_enemy_cores = tuple(
        enemy for enemy in turn.visible_enemies if isinstance(enemy, CoreView)
    )
    for enemy_core in visible_enemy_cores:
        previous = memory.enemy_core_intel.get(enemy_core.id)
        sightings = 1
        if previous is not None and turn.tick - previous.last_seen_tick <= config.raid_intel_ttl:
            sightings = previous.sighting_count + int(turn.tick > previous.last_seen_tick)
        memory.enemy_core_intel[enemy_core.id] = EnemyCoreIntel(
            id=enemy_core.id,
            position=enemy_core.position,
            hp=enemy_core.hp,
            shield=enemy_core.shield,
            state=enemy_core.state,
            destination=enemy_core.destination,
            last_seen_tick=turn.tick,
            sighting_count=sightings,
        )
    for core_id, intel in tuple(memory.enemy_core_intel.items()):
        active_long_range = (
            memory.raid_long_range_campaign is not None
            and memory.raid_long_range_campaign.target_id == core_id
            and turn.tick <= memory.raid_long_range_campaign.search_deadline_tick
        )
        if turn.tick - intel.last_seen_tick > config.raid_intel_ttl and not active_long_range:
            memory.enemy_core_intel.pop(core_id, None)
    if turn.core is not None:
        nearby_combat = [
            enemy
            for enemy in visible_enemy_units
            if enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.position, turn.core.position)
            <= config.home_warning_radius
        ]
        if nearby_combat:
            memory.recent_home_threat_position = min(
                nearby_combat,
                key=lambda enemy: (
                    manhattan(enemy.position, turn.core.position),
                    enemy.id.bytes,
                ),
            ).position
    for enemy in visible_enemy_units:
        previous = memory.enemy_tracks.get(enemy.id)
        sample = turn.tick, enemy.position
        if previous is None or turn.tick > previous.last_seen_tick + 1:
            samples = (sample,)
        elif turn.tick == previous.last_seen_tick:
            samples = (*previous.samples[:-1], sample)
        else:
            samples = (*previous.samples, sample)[-4:]
        memory.enemy_tracks[enemy.id] = EnemyTrack(
            id=enemy.id,
            unit_type=enemy.unit_type,
            samples=samples,
            last_seen_tick=turn.tick,
        )
        if (
            enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and turn.core is not None
            and manhattan(enemy.position, turn.core.position)
            <= config.home_warning_radius
        ):
            memory.hostile_force_ids.add(enemy.id)
    for enemy_id, track in tuple(memory.enemy_tracks.items()):
        if turn.tick - track.last_seen_tick > config.enemy_track_ttl:
            memory.enemy_tracks.pop(enemy_id, None)

    home_enemies_visible = any(
        enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
        and turn.core is not None
        and manhattan(enemy.position, turn.core.position) <= config.home_warning_radius
        for enemy in visible_enemy_units
    )
    if home_enemies_visible:
        memory.home_safe_ticks = 0
    else:
        memory.home_safe_ticks += 1
        if memory.home_safe_ticks >= 8:
            memory.hostile_force_ids.clear()
    memory.home_force_high_water = max(
        config.home_force_floor,
        memory.home_force_high_water,
        len(memory.hostile_force_ids),
    )
    workers = sum(unit.unit_type is UnitType.WORKER for unit in turn.units)
    vanguards = sum(unit.unit_type is UnitType.VANGUARD for unit in turn.units)
    rangers = sum(unit.unit_type is UnitType.RANGER for unit in turn.units)
    if workers >= config.opening_worker_target and vanguards and rangers:
        memory.opening_complete = True
    return visible_enemy_units


def _sync_danger_memory(
    turn: Turn,
    memory: TacticMemory,
    config: TacticConfig,
    visible_enemy_units: tuple[UnitView, ...],
) -> dict[Position, int]:

    danger: dict[Position, int] = {}
    current_snapshots = tuple(
        _unit_snapshot(enemy, controlled=False) for enemy in visible_enemy_units
    )
    for enemy in current_snapshots:
        for cell in _attack_cells(enemy, frozenset(memory.known_obstacles)):
            danger[cell] = max(danger.get(cell, 0), config.danger_envelope_ttl)
            memory.danger_until[cell] = turn.tick + config.danger_envelope_ttl
    # Recently fogged enemies contribute a conservative expanding envelope.
    for track in memory.enemy_tracks.values():
        age = turn.tick - track.last_seen_tick
        if age <= 0:
            continue
        phantom = EntitySnapshot(
            id=track.id,
            position=track.position,
            hp=1,
            unit_type=track.unit_type,
            controlled=False,
        )
        base_attack_cells = _attack_cells(
            phantom,
            frozenset(memory.known_obstacles),
        )
        for base_cell in base_attack_cells:
            for cell in diamond(base_cell, min(age, config.enemy_track_ttl)):
                danger[cell] = max(danger.get(cell, 0), config.enemy_track_ttl - age + 1)
                memory.danger_until[cell] = max(
                    memory.danger_until.get(cell, 0),
                    turn.tick + config.enemy_track_ttl - age,
                )
    for cell, expires in tuple(memory.danger_until.items()):
        if expires < turn.tick:
            memory.danger_until.pop(cell, None)
        else:
            danger[cell] = max(danger.get(cell, 0), expires - turn.tick + 1)

    return danger


def _update_threat_heat(
    memory: TacticMemory,
    *,
    position: Position,
    risk: int,
    tick: int,
    ttl: int,
    source: str,
) -> None:
    """Refresh one uncertain danger cell without treating it as occupancy."""

    previous = memory.threat_heat.get(position)
    previous_score = 0 if previous is None else previous.score(tick)
    if risk >= previous_score:
        memory.threat_heat[position] = ThreatHeatCell(
            position=position,
            risk=risk,
            updated_tick=tick,
            expires_tick=tick + ttl,
            source=source,
        )
        return
    # A lower-ranked observation must not erase stronger evidence.  It may,
    # however, keep that evidence alive while the area remains actively hot.
    memory.threat_heat[position] = ThreatHeatCell(
        position=position,
        risk=previous_score,
        updated_tick=tick,
        expires_tick=max(previous.expires_tick, tick + ttl),
        source=previous.source,
    )


def _sync_threat_heat(
    turn: Turn,
    memory: TacticMemory,
    config: TacticConfig,
    visible_enemy_units: tuple[UnitView, ...],
) -> dict[Position, int]:
    """Maintain durable, decaying route risk separately from exact tracks."""

    obstacles = frozenset(memory.known_obstacles)
    for enemy_view in visible_enemy_units:
        enemy = _unit_snapshot(enemy_view, controlled=False)
        if enemy.unit_type not in {UnitType.VANGUARD, UnitType.RANGER}:
            continue
        current_zone = {enemy.position, *_attack_cells(enemy, obstacles)}
        for cell in current_zone:
            _update_threat_heat(
                memory,
                position=cell,
                risk=config.threat_heat_visible_risk,
                tick=turn.tick,
                ttl=config.threat_heat_visible_ttl,
                source="VISIBLE_ATTACK_ZONE",
            )
        possible_positions = {enemy.position}
        possible_positions.update(
            add_direction(enemy.position, direction)
            for direction in DIRECTION_ORDER
            if add_direction(enemy.position, direction) not in obstacles
        )
        projected_zone: set[Position] = set()
        for position in possible_positions:
            projected = EntitySnapshot(
                id=enemy.id,
                position=position,
                hp=enemy.hp,
                unit_type=enemy.unit_type,
                controlled=False,
            )
            projected_zone.add(position)
            projected_zone.update(_attack_cells(projected, obstacles))
        for cell in projected_zone - current_zone:
            _update_threat_heat(
                memory,
                position=cell,
                risk=config.threat_heat_projected_risk,
                tick=turn.tick,
                ttl=config.threat_heat_projected_ttl,
                source="PROJECTED_ATTACK_ZONE",
            )

    for event in turn.events:
        if event.position is None:
            continue
        if event.event_type == "UNIT_DAMAGED":
            center_risk = config.threat_heat_damage_risk
            neighbor_risk = config.threat_heat_damage_neighbor_risk
            ttl = config.threat_heat_damage_ttl
            source = "UNIT_DAMAGED"
        elif event.event_type == "UNIT_DESTROYED":
            center_risk = config.threat_heat_destroyed_risk
            neighbor_risk = config.threat_heat_destroyed_neighbor_risk
            ttl = config.threat_heat_destroyed_ttl
            source = "UNIT_DESTROYED"
        else:
            continue
        _update_threat_heat(
            memory,
            position=event.position,
            risk=center_risk,
            tick=turn.tick,
            ttl=ttl,
            source=source,
        )
        for _, neighbor in cardinal_neighbors(event.position):
            _update_threat_heat(
                memory,
                position=neighbor,
                risk=neighbor_risk,
                tick=turn.tick,
                ttl=ttl,
                source=f"{source}_NEARBY",
            )

    return _decayed_threat_heat(
        memory,
        tick=turn.tick,
        cell_limit=config.threat_heat_cell_limit,
    )


def _decayed_threat_heat(
    memory: TacticMemory,
    *,
    tick: int,
    cell_limit: int,
) -> dict[Position, int]:
    scored = {
        cell: record.score(tick)
        for cell, record in memory.threat_heat.items()
        if record.score(tick) > 0
    }
    for cell in tuple(memory.threat_heat):
        if cell not in scored:
            memory.threat_heat.pop(cell, None)
    if len(scored) > cell_limit:
        retained = {
            cell
            for cell, _ in sorted(
                scored.items(),
                key=lambda item: (
                    -item[1],
                    -memory.threat_heat[item[0]].updated_tick,
                    item[0],
                ),
            )[:cell_limit]
        }
        for cell in tuple(memory.threat_heat):
            if cell not in retained:
                memory.threat_heat.pop(cell, None)
        scored = {cell: scored[cell] for cell in retained}
    return scored


def _prune_tactical_timers(tick: int, memory: TacticMemory) -> None:
    for cell, until in tuple(memory.target_backoff_until.items()):
        if tick >= until:
            memory.target_backoff_until.pop(cell, None)
    for cell, until in tuple(memory.failed_core_destinations.items()):
        if tick >= until:
            memory.failed_core_destinations.pop(cell, None)


def build_world_model(
    turn: Turn,
    memory: TacticMemory | None = None,
    config: TacticConfig = DEFAULT_CONFIG,
) -> WorldModel:
    """Build a controller-free immutable snapshot and optionally update memory."""

    if memory is None:
        memory = TacticMemory()
    observation, danger, threat_heat = _sync_memory(turn, memory, config)

    core = None
    if turn.core is not None:
        view = turn.core.view
        core = CoreSnapshot(
            id=turn.core.id,
            position=turn.core.position,
            hp=turn.core.hp,
            shield=turn.core.shield,
            state=view.state,
            destination=view.destination,
            move_progress=view.move_progress,
            move_required_ticks=view.move_required_ticks,
        )
    friendlies = tuple(
        sorted(
            (_unit_snapshot(unit.view, controlled=True) for unit in turn.units),
            key=lambda unit: unit.id.bytes,
        )
    )
    enemies: list[EntitySnapshot] = []
    enemy_cores: list[EnemyCoreSnapshot] = []
    for enemy in turn.visible_enemies:
        if isinstance(enemy, UnitView):
            enemies.append(_unit_snapshot(enemy, controlled=False))
        elif isinstance(enemy, CoreView):
            enemy_cores.append(
                EnemyCoreSnapshot(
                    id=enemy.id,
                    position=enemy.position,
                    hp=enemy.hp,
                    shield=enemy.shield,
                    state=enemy.state,
                    destination=enemy.destination,
                    move_progress=enemy.move_progress,
                    move_required_ticks=enemy.move_required_ticks,
                )
            )

    occupied = Counter(unit.position for unit in friendlies)
    occupied.update(enemy.position for enemy in enemies)
    occupied.update(enemy.position for enemy in enemy_cores)
    if core is not None:
        occupied[core.position] += 1

    resource_claims: defaultdict[Position, list[UUID]] = defaultdict(list)
    for worker_id, mission in memory.unit_missions.items():
        if (
            mission.mission is UnitMission.HARVEST
            and mission.target is not None
            and mission.target in memory.resource_memory
        ):
            resource_claims[mission.target].append(worker_id)
    resources = tuple(
        ResourceIntel(
            position=position,
            last_seen_tick=seen_tick,
            visible_now=position in turn.resource_cells,
            assigned_worker_ids=tuple(
                sorted(resource_claims.get(position, ()), key=lambda item: item.bytes)
            ),
        )
        for position, seen_tick in sorted(memory.resource_memory.items())
    )

    return WorldModel(
        tick=turn.tick,
        resources=turn.resources,
        population=turn.state.population,
        resource_capacity=turn.resource_capacity,
        core=core,
        friendlies=friendlies,
        enemies=tuple(sorted(enemies, key=lambda enemy: enemy.id.bytes)),
        enemy_cores=tuple(sorted(enemy_cores, key=lambda enemy: enemy.id.bytes)),
        beacon=BeaconSnapshot(
            position=turn.beacon.position,
            status=turn.beacon.status,
            carrier_id=turn.beacon.carrier_id,
        ),
        visible_resources=frozenset(turn.resource_cells),
        remembered_resources=tuple(sorted(memory.resource_memory.items())),
        visible_obstacles=frozenset(turn.obstacle_cells),
        known_obstacles=frozenset(memory.known_obstacles),
        known_passable=frozenset(memory.known_passable),
        visible_cells=observation.visible_cells,
        vision_sources=observation.sources,
        visibility_coverage=observation.coverage,
        cell_last_visible=tuple(sorted(memory.cell_last_visible.items())),
        resource_intel=resources,
        enemy_tracks=tuple(sorted(memory.enemy_tracks.values(), key=lambda track: track.id.bytes)),
        remembered_enemy_cores=tuple(
            sorted(memory.enemy_core_intel.values(), key=lambda intel: intel.id.bytes)
        ),
        danger_cells=tuple(sorted(danger.items())),
        threat_heat=tuple(sorted(threat_heat.items())),
        occupied_cells=tuple(sorted(occupied.items())),
        congestion_cells=tuple(sorted(memory.congestion_counts.items())),
    )
