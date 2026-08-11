from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping
from uuid import UUID

from arena_hero import CoreState, Position, UnitType

from .config import DEFAULT_CONFIG, TacticConfig
from .geometry import (
    DIRECTION_ORDER,
    add_direction,
    diamond,
    manhattan,
    unit_attack_cells,
)
from .models import EnemyTrack, ResourceIntel, VisionSource, WorldModel


@dataclass(frozen=True, slots=True)
class EnemyProjection:
    """Shared team intelligence for one enemy Unit.

    A visible enemy is authoritative for this Tick.  A fogged projection is
    explicitly uncertain and contributes route risk, never hard occupancy.
    """

    enemy_id: UUID
    unit_type: UnitType
    observed_position: Position
    visible_now: bool
    last_seen_tick: int
    age: int
    confidence: str
    observer_ids: tuple[UUID, ...]
    samples: tuple[tuple[int, Position], ...]
    possible_positions: tuple[Position, ...]
    movement_corridor: tuple[Position, ...]
    immediate_attack_cells: frozenset[Position]
    future_attack_cells: frozenset[Position]

    @property
    def position(self) -> Position:
        """Compatibility alias for planners that only need the shared fix."""

        return self.observed_position


@dataclass(frozen=True, slots=True)
class EnemyCoreProjection:
    enemy_id: UUID
    position: Position
    visible_now: bool
    last_seen_tick: int
    age: int
    confidence: str
    possible_positions: tuple[Position, ...]


@dataclass(frozen=True, slots=True)
class ThreatCell:
    """All tactical evidence about one cell, produced once for the team."""

    position: Position
    immediate_attackers: int
    future_attackers: int
    remembered_risk: int
    heat: int
    corridor_risk: int
    proximity_risk: int
    source_enemy_ids: tuple[UUID, ...]

    @property
    def worker_risk(self) -> int:
        return (
            max(self.remembered_risk, self.heat)
            + self.corridor_risk
            + self.proximity_risk
        )


@dataclass(frozen=True, slots=True)
class TacticalMap:
    """Immutable team-level tactical interpretation for one Turn.

    Every role consumes this same value.  Role planners may choose different
    risk tolerances, but they must not rebuild private enemy truth.
    """

    tick: int
    known_obstacles: frozenset[Position]
    known_passable: frozenset[Position]
    vision_sources: tuple[VisionSource, ...]
    visible_cells: frozenset[Position]
    visibility_coverage: Mapping[Position, tuple[UUID, ...]]
    last_visible_ticks: Mapping[Position, int]
    resources: tuple[ResourceIntel, ...]
    enemies: tuple[EnemyProjection, ...]
    enemy_cores: tuple[EnemyCoreProjection, ...]
    threat_cells: Mapping[Position, ThreatCell]
    immediate_damage: Mapping[Position, int]
    future_damage: Mapping[Position, int]
    remembered_danger: Mapping[Position, int]
    threat_heat: Mapping[Position, int]
    worker_route_costs: Mapping[Position, int]
    hostile_occupied: frozenset[Position]
    friendly_positions: Mapping[UUID, Position]
    friendly_types: Mapping[UUID, UnitType | None]
    occupied_cells: Mapping[Position, int]
    congestion: Mapping[Position, int]
    projected_core_position: Position | None
    core_completes_move: bool
    service_positions: frozenset[Position] = frozenset()
    planned_positions: Mapping[UUID, Position] = field(
        default_factory=lambda: MappingProxyType({})
    )
    reserved_positions: frozenset[Position] = frozenset()

    def immediate_attackers(self, cell: Position) -> int:
        return self.immediate_damage.get(cell, 0)

    def future_attackers(self, cell: Position) -> int:
        return self.future_damage.get(cell, 0)

    def exposure(self, cell: Position) -> tuple[int, int, int]:
        """Compatibility exposure for combat, Core and formation planners."""

        return (
            self.immediate_damage.get(cell, 0),
            self.future_damage.get(cell, 0),
            self.remembered_danger.get(cell, 0),
        )

    def worker_exposure(self, cell: Position) -> tuple[int, int, int]:
        threat = self.threat_cells.get(cell)
        return (
            self.immediate_damage.get(cell, 0),
            self.future_damage.get(cell, 0),
            0 if threat is None else threat.worker_risk,
        )

    def role_risk(self, unit_type: UnitType | None, cell: Position) -> int:
        if unit_type is UnitType.WORKER:
            immediate, future, uncertain = self.worker_exposure(cell)
        else:
            immediate, future, uncertain = self.exposure(cell)
        return immediate * 100 + future * 10 + uncertain

    def route_costs_for(self, unit_type: UnitType | None) -> Mapping[Position, int]:
        if unit_type is UnitType.WORKER:
            return self.worker_route_costs
        return MappingProxyType(
            {
                cell: self.future_damage.get(cell, 0) * 8 + risk
                for cell, risk in self.remembered_danger.items()
            }
        )

    def enemy(self, enemy_id: UUID) -> EnemyProjection | None:
        return next((item for item in self.enemies if item.enemy_id == enemy_id), None)

    def enemy_core(self, enemy_id: UUID) -> EnemyCoreProjection | None:
        return next(
            (item for item in self.enemy_cores if item.enemy_id == enemy_id),
            None,
        )

    def observers(self, cell: Position) -> tuple[UUID, ...]:
        return self.visibility_coverage.get(cell, ())

    def resource(self, cell: Position) -> ResourceIntel | None:
        return next((item for item in self.resources if item.position == cell), None)

    def with_operations(
        self,
        *,
        service_positions: frozenset[Position] | None = None,
        planned_positions: Mapping[UUID, Position] | None = None,
        reserved_positions: frozenset[Position] | None = None,
        projected_core_position: Position | None = None,
    ) -> TacticalMap:
        return replace(
            self,
            service_positions=(
                self.service_positions
                if service_positions is None
                else service_positions
            ),
            planned_positions=(
                self.planned_positions
                if planned_positions is None
                else MappingProxyType(dict(planned_positions))
            ),
            reserved_positions=(
                self.reserved_positions
                if reserved_positions is None
                else reserved_positions
            ),
            projected_core_position=(
                self.projected_core_position
                if projected_core_position is None
                else projected_core_position
            ),
        )

    def with_resource_assignments(
        self,
        assignments: Mapping[Position, tuple[UUID, ...]],
    ) -> TacticalMap:
        return replace(
            self,
            resources=tuple(
                replace(
                    resource,
                    assigned_worker_ids=tuple(
                        sorted(
                            assignments.get(resource.position, ()),
                            key=lambda worker_id: worker_id.bytes,
                        )
                    ),
                )
                for resource in self.resources
            ),
        )


def attack_cells(
    position: Position,
    unit_type: UnitType,
    obstacles: frozenset[Position],
) -> frozenset[Position]:
    return unit_attack_cells(position, unit_type, obstacles)


def _possible_enemy_positions(position: Position, world: WorldModel) -> tuple[Position, ...]:
    obstacles = world.known_obstacles
    occupied = dict(world.occupied_cells)
    positions = {position}
    positions.update(
        add_direction(position, direction)
        for direction in DIRECTION_ORDER
        if add_direction(position, direction) not in obstacles
        and occupied.get(add_direction(position, direction), 0) < 2
    )
    return tuple(sorted(positions))


def _motion_delta(track: EnemyTrack | None) -> tuple[int, int] | None:
    if track is None or len(track.samples) < 2:
        return None
    (previous_tick, previous), (current_tick, current) = track.samples[-2:]
    delta = current[0] - previous[0], current[1] - previous[1]
    if current_tick != previous_tick + 1 or abs(delta[0]) + abs(delta[1]) != 1:
        return None
    return delta


def _confidence(track: EnemyTrack | None, *, visible: bool, age: int) -> str:
    if not visible:
        return "MEDIUM" if age == 1 else "LOW"
    if track is None or len(track.samples) < 2:
        return "LOW"
    if len(track.samples) >= 3:
        positions = tuple(position for _, position in track.samples[-3:])
        if len(set(positions)) == 1:
            return "HIGH"
        first = (
            positions[1][0] - positions[0][0],
            positions[1][1] - positions[0][1],
        )
        second = (
            positions[2][0] - positions[1][0],
            positions[2][1] - positions[1][1],
        )
        if first == second and abs(second[0]) + abs(second[1]) == 1:
            return "HIGH"
    return "MEDIUM"


def _fog_possible_positions(
    track: EnemyTrack,
    world: WorldModel,
    age: int,
) -> tuple[Position, ...]:
    radius = max(1, age)
    candidates = {
        cell
        for cell in diamond(track.position, radius)
        if cell not in world.known_obstacles
        and (cell in world.known_passable or cell == track.position)
    }
    candidates.add(track.position)
    return tuple(sorted(candidates))


def _movement_corridor(
    position: Position,
    track: EnemyTrack | None,
    possible_positions: tuple[Position, ...],
    world: WorldModel,
    config: TacticConfig,
) -> tuple[Position, ...]:
    corridor = set(possible_positions)
    delta = _motion_delta(track)
    if delta is None:
        corridor.update(
            cell
            for cell in diamond(position, config.global_worker_corridor_width)
            if cell not in world.known_obstacles
        )
        return tuple(sorted(corridor))
    for step in range(1, config.global_worker_corridor_projection_ticks + 1):
        center = position[0] + delta[0] * step, position[1] + delta[1] * step
        if center in world.known_obstacles:
            break
        corridor.update(
            cell
            for cell in diamond(center, config.global_worker_corridor_width)
            if cell not in world.known_obstacles
        )
    return tuple(sorted(corridor))


def _build_enemy_projections(
    world: WorldModel,
    config: TacticConfig,
) -> tuple[EnemyProjection, ...]:
    visible = {enemy.id: enemy for enemy in world.enemies}
    tracks = {track.id: track for track in world.enemy_tracks}
    ids = set(visible)
    ids.update(
        track_id
        for track_id, track in tracks.items()
        if world.tick - track.last_seen_tick <= config.enemy_track_ttl
    )
    projections: list[EnemyProjection] = []
    for enemy_id in sorted(ids, key=lambda item: item.bytes):
        enemy = visible.get(enemy_id)
        track = tracks.get(enemy_id)
        visible_now = enemy is not None
        if enemy is not None:
            position = enemy.position
            unit_type = enemy.unit_type
            last_seen_tick = world.tick
            possible = _possible_enemy_positions(position, world)
        else:
            assert track is not None
            position = track.position
            unit_type = track.unit_type
            last_seen_tick = track.last_seen_tick
            possible = _fog_possible_positions(
                track,
                world,
                world.tick - track.last_seen_tick,
            )
        age = world.tick - last_seen_tick
        immediate = (
            attack_cells(position, unit_type, world.known_obstacles)
            if visible_now
            else frozenset()
        )
        future: set[Position] = set()
        if visible_now:
            for candidate in possible:
                future.update(attack_cells(candidate, unit_type, world.known_obstacles))
        projections.append(
            EnemyProjection(
                enemy_id=enemy_id,
                unit_type=unit_type,
                observed_position=position,
                visible_now=visible_now,
                last_seen_tick=last_seen_tick,
                age=age,
                confidence=_confidence(track, visible=visible_now, age=age),
                observer_ids=(world.observers(position) if visible_now else ()),
                samples=() if track is None else track.samples,
                possible_positions=possible,
                movement_corridor=_movement_corridor(
                    position,
                    track,
                    possible,
                    world,
                    config,
                ),
                immediate_attack_cells=immediate,
                future_attack_cells=frozenset(future),
            )
        )
    return tuple(projections)


def _build_enemy_core_projections(world: WorldModel) -> tuple[EnemyCoreProjection, ...]:
    visible = {core.id: core for core in world.enemy_cores}
    remembered = {core.id: core for core in world.remembered_enemy_cores}
    result: list[EnemyCoreProjection] = []
    for core_id in sorted(set(visible) | set(remembered), key=lambda item: item.bytes):
        current = visible.get(core_id)
        old = remembered.get(core_id)
        if current is not None:
            possible = {current.position}
            if current.destination is not None:
                possible.add(current.destination)
            result.append(
                EnemyCoreProjection(
                    enemy_id=core_id,
                    position=current.position,
                    visible_now=True,
                    last_seen_tick=world.tick,
                    age=0,
                    confidence="EXACT",
                    possible_positions=tuple(sorted(possible)),
                )
            )
        elif old is not None:
            possible = {old.position}
            if old.destination is not None:
                possible.add(old.destination)
            result.append(
                EnemyCoreProjection(
                    enemy_id=core_id,
                    position=old.position,
                    visible_now=False,
                    last_seen_tick=old.last_seen_tick,
                    age=world.tick - old.last_seen_tick,
                    confidence="LOW",
                    possible_positions=tuple(sorted(possible)),
                )
            )
    return tuple(result)


def build_tactical_map(
    world: WorldModel,
    config: TacticConfig = DEFAULT_CONFIG,
) -> TacticalMap:
    projections = _build_enemy_projections(world, config)
    immediate = Counter[Position]()
    future = Counter[Position]()
    source_ids: defaultdict[Position, set[UUID]] = defaultdict(set)
    corridor_risk: dict[Position, int] = {}
    proximity_risk: dict[Position, int] = {}

    for enemy in projections:
        if enemy.visible_now:
            immediate.update(enemy.immediate_attack_cells)
            future.update(enemy.future_attack_cells)
            for cell in enemy.immediate_attack_cells | enemy.future_attack_cells:
                source_ids[cell].add(enemy.enemy_id)
        if enemy.unit_type not in {UnitType.VANGUARD, UnitType.RANGER}:
            continue
        age_penalty = enemy.age * 2
        corridor_value = max(1, config.global_worker_corridor_risk - age_penalty)
        for cell in enemy.movement_corridor:
            corridor_risk[cell] = max(corridor_risk.get(cell, 0), corridor_value)
            source_ids[cell].add(enemy.enemy_id)
        for cell in diamond(
            enemy.observed_position,
            config.global_worker_threat_awareness_radius,
        ):
            if cell in world.known_obstacles:
                continue
            distance = manhattan(cell, enemy.observed_position)
            value = max(
                0,
                config.global_worker_threat_awareness_radius - distance + 1 - enemy.age,
            )
            if value:
                proximity_risk[cell] = max(proximity_risk.get(cell, 0), value)
                source_ids[cell].add(enemy.enemy_id)

    remembered = dict(world.danger_cells)
    heat = dict(world.threat_heat)
    for cell, risk in heat.items():
        remembered[cell] = max(remembered.get(cell, 0), risk)
    all_cells = (
        set(immediate)
        | set(future)
        | set(remembered)
        | set(heat)
        | set(corridor_risk)
        | set(proximity_risk)
    )
    threat_cells = {
        cell: ThreatCell(
            position=cell,
            immediate_attackers=immediate.get(cell, 0),
            future_attackers=future.get(cell, 0),
            remembered_risk=remembered.get(cell, 0),
            heat=heat.get(cell, 0),
            corridor_risk=corridor_risk.get(cell, 0),
            proximity_risk=proximity_risk.get(cell, 0),
            source_enemy_ids=tuple(
                sorted(source_ids.get(cell, ()), key=lambda item: item.bytes)
            ),
        )
        for cell in all_cells
    }
    worker_costs = {
        cell: threat.future_attackers * 8 + threat.worker_risk
        for cell, threat in threat_cells.items()
        if threat.future_attackers or threat.worker_risk
    }

    projected_core_position = None
    core_completes_move = False
    if world.core is not None:
        projected_core_position = world.core.position
        if (
            world.core.state is CoreState.MOVING
            and world.core.destination is not None
            and world.core.move_progress is not None
            and world.core.move_required_ticks is not None
            and world.core.move_progress >= world.core.move_required_ticks - 1
        ):
            projected_core_position = world.core.destination
            core_completes_move = True

    occupied = {enemy.position for enemy in world.enemies}
    for enemy_core in world.enemy_cores:
        occupied.add(enemy_core.position)
        if (
            enemy_core.state is CoreState.MOVING
            and enemy_core.destination is not None
            and enemy_core.move_progress is not None
            and enemy_core.move_required_ticks is not None
            and enemy_core.move_progress >= enemy_core.move_required_ticks - 1
        ):
            occupied.add(enemy_core.destination)

    return TacticalMap(
        tick=world.tick,
        known_obstacles=world.known_obstacles,
        known_passable=world.known_passable,
        vision_sources=world.vision_sources,
        visible_cells=world.visible_cells,
        visibility_coverage=MappingProxyType(dict(world.visibility_coverage)),
        last_visible_ticks=MappingProxyType(dict(world.cell_last_visible)),
        resources=world.resource_intel,
        enemies=projections,
        enemy_cores=_build_enemy_core_projections(world),
        threat_cells=MappingProxyType(threat_cells),
        immediate_damage=MappingProxyType(dict(immediate)),
        future_damage=MappingProxyType(dict(future)),
        remembered_danger=MappingProxyType(remembered),
        threat_heat=MappingProxyType(heat),
        worker_route_costs=MappingProxyType(worker_costs),
        hostile_occupied=frozenset(occupied),
        friendly_positions=MappingProxyType(
            {
                **(
                    {}
                    if world.core is None
                    else {world.core.id: world.core.position}
                ),
                **{unit.id: unit.position for unit in world.friendlies},
            }
        ),
        friendly_types=MappingProxyType(
            {
                **({} if world.core is None else {world.core.id: None}),
                **{unit.id: unit.unit_type for unit in world.friendlies},
            }
        ),
        occupied_cells=MappingProxyType(dict(world.occupied_cells)),
        congestion=MappingProxyType(dict(world.congestion_cells)),
        projected_core_position=projected_core_position,
        core_completes_move=core_completes_move,
    )


# Public compatibility names.  Existing plugins can keep importing the old
# projection terminology while every planner now receives one TacticalMap.
ProjectedTurn = TacticalMap
build_projected_turn = build_tactical_map


__all__ = (
    "EnemyCoreProjection",
    "EnemyProjection",
    "ProjectedTurn",
    "TacticalMap",
    "ThreatCell",
    "attack_cells",
    "build_projected_turn",
    "build_tactical_map",
)
