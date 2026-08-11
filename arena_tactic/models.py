from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Any
from uuid import UUID

from arena_hero import BeaconStatus, CoreState, Direction, Position, UnitType


class WorkerPatrolMode(str, Enum):
    INFORMATION_GAIN = "INFORMATION_GAIN"
    FRONTIER_ONLY = "FRONTIER_ONLY"
    LOCAL_FALLBACK = "LOCAL_FALLBACK"


class WorkerScoutPhase(str, Enum):
    FRONTIER = "FRONTIER"
    STALE_REVISIT = "STALE_REVISIT"
    SECTOR_SCOUT = "SECTOR_SCOUT"
    LOCAL_DISPERSAL = "LOCAL_DISPERSAL"


class UnitMission(str, Enum):
    CORE_SURVIVAL = "CORE_SURVIVAL"
    DEPOSIT = "DEPOSIT"
    ESCAPE = "ESCAPE"
    ATTACK = "ATTACK"
    RECOVER = "RECOVER"
    RETURN_CARGO = "RETURN_CARGO"
    CLEAR_CORE = "CLEAR_CORE"
    HARVEST = "HARVEST"
    HOME_GUARD = "HOME_GUARD"
    HOME_DEFENSE = "HOME_DEFENSE"
    RAID = "RAID"
    EXPLORE = "EXPLORE"
    PATROL = "PATROL"
    PRODUCTION = "PRODUCTION"
    BEACON = "BEACON"
    WAIT = "WAIT"


class IntentAction(str, Enum):
    WAIT = "WAIT"
    MOVE = "MOVE"
    HARVEST = "HARVEST"
    DEPOSIT = "DEPOSIT"
    SWEEP = "SWEEP"
    SHOOT_CELL = "SHOOT_CELL"
    SHOOT = "SHOOT"
    HEAL = "HEAL"
    SPAWN = "SPAWN"
    START_MOVE = "START_MOVE"
    CANCEL_MOVE = "CANCEL_MOVE"
    REPAIR_SHIELD = "REPAIR_SHIELD"
    PICKUP_BEACON = "PICKUP_BEACON"
    DROP_BEACON = "DROP_BEACON"
    SELF_DESTRUCT = "SELF_DESTRUCT"


class VanguardIntent(str, Enum):
    ATTACKING = "ATTACKING"
    RETREATING = "RETREATING"
    BLIND_SPOT_APPROACH = "BLIND_SPOT_APPROACH"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class EntitySnapshot:
    id: UUID
    position: Position
    hp: int
    unit_type: UnitType
    cargo: int = 0
    controlled: bool = True


@dataclass(frozen=True, slots=True)
class CoreSnapshot:
    id: UUID
    position: Position
    hp: int
    shield: int
    state: CoreState
    destination: Position | None = None
    move_progress: int | None = None
    move_required_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class EnemyCoreSnapshot:
    id: UUID
    position: Position
    hp: int
    shield: int
    state: CoreState
    destination: Position | None = None
    move_progress: int | None = None
    move_required_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class BeaconSnapshot:
    position: Position
    status: BeaconStatus | None
    carrier_id: UUID | None


@dataclass(frozen=True, slots=True)
class VisionSource:
    """One friendly object's contribution to the current team view."""

    actor_id: UUID
    actor_kind: str
    unit_type: UnitType | None
    position: Position
    radius: int
    visible_cells: tuple[Position, ...]


@dataclass(frozen=True, slots=True)
class ResourceIntel:
    """Shared, controller-free knowledge about one resource coordinate."""

    position: Position
    last_seen_tick: int
    visible_now: bool
    assigned_worker_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True, slots=True)
class EnemyTrack:
    id: UUID
    unit_type: UnitType
    samples: tuple[tuple[int, Position], ...]
    last_seen_tick: int

    @property
    def position(self) -> Position:
        return self.samples[-1][1]


@dataclass(frozen=True, slots=True)
class ThreatHeatCell:
    """Durable, linearly decaying evidence that a cell is dangerous.

    Unlike :class:`EnemyTrack`, this record is deliberately spatial and
    uncertain: it raises route cost but never turns a fogged cell into a hard
    obstacle.
    """

    position: Position
    risk: int
    updated_tick: int
    expires_tick: int
    source: str

    def score(self, tick: int) -> int:
        if self.risk <= 0 or tick >= self.expires_tick:
            return 0
        duration = max(1, self.expires_tick - self.updated_tick)
        remaining = self.expires_tick - tick
        return max(1, ceil(self.risk * remaining / duration))


@dataclass(frozen=True, slots=True)
class EnemyCoreIntel:
    id: UUID
    position: Position
    hp: int
    shield: int
    state: CoreState
    destination: Position | None
    last_seen_tick: int
    sighting_count: int


@dataclass(frozen=True, slots=True)
class MoveAttempt:
    actor_id: UUID
    tick: int
    origin: Position
    destination: Position
    direction: Direction


@dataclass(frozen=True, slots=True)
class MoveFailure:
    destination: Position
    expires_tick: int
    reason: str


@dataclass(frozen=True, slots=True)
class WorldModel:
    tick: int
    resources: int
    population: int
    resource_capacity: int
    core: CoreSnapshot | None
    friendlies: tuple[EntitySnapshot, ...]
    enemies: tuple[EntitySnapshot, ...]
    enemy_cores: tuple[EnemyCoreSnapshot, ...]
    beacon: BeaconSnapshot
    visible_resources: frozenset[Position]
    remembered_resources: tuple[tuple[Position, int], ...]
    visible_obstacles: frozenset[Position]
    known_obstacles: frozenset[Position]
    known_passable: frozenset[Position]
    visible_cells: frozenset[Position]
    vision_sources: tuple[VisionSource, ...]
    visibility_coverage: tuple[tuple[Position, tuple[UUID, ...]], ...]
    cell_last_visible: tuple[tuple[Position, int], ...]
    resource_intel: tuple[ResourceIntel, ...]
    enemy_tracks: tuple[EnemyTrack, ...]
    remembered_enemy_cores: tuple[EnemyCoreIntel, ...]
    danger_cells: tuple[tuple[Position, int], ...]
    threat_heat: tuple[tuple[Position, int], ...]
    occupied_cells: tuple[tuple[Position, int], ...]
    congestion_cells: tuple[tuple[Position, int], ...]

    def friendly(self, actor_id: UUID) -> EntitySnapshot | None:
        return next((unit for unit in self.friendlies if unit.id == actor_id), None)

    def enemy(self, enemy_id: UUID) -> EntitySnapshot | None:
        return next((unit for unit in self.enemies if unit.id == enemy_id), None)

    def actor_position(self, actor_id: UUID | None) -> Position | None:
        if actor_id is None:
            return None if self.core is None else self.core.position
        actor = self.friendly(actor_id)
        return None if actor is None else actor.position

    def track(self, enemy_id: UUID) -> EnemyTrack | None:
        return next((track for track in self.enemy_tracks if track.id == enemy_id), None)

    def observers(self, position: Position) -> tuple[UUID, ...]:
        return next(
            (
                observer_ids
                for cell, observer_ids in self.visibility_coverage
                if cell == position
            ),
            (),
        )

    def resource(self, position: Position) -> ResourceIntel | None:
        return next(
            (intel for intel in self.resource_intel if intel.position == position),
            None,
        )


Metadata = tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ActionIntent:
    actor_id: UUID | None
    action: IntentAction
    mission: UnitMission
    priority: int
    direction: Direction | None = None
    target_position: Position | None = None
    target_id: UUID | None = None
    expected_cell: Position | None = None
    unit_type: UnitType | None = None
    risk: int = 0
    resource_cost: int = 0
    resource_gain: int = 0
    reserve_positions: tuple[Position, ...] = ()
    exclusive_destination: bool = False
    tie_break: tuple[int, ...] = ()
    reason: str = ""
    metadata: Metadata = ()

    @classmethod
    def simple(
        cls,
        actor_id: UUID | None,
        action: IntentAction,
        mission: UnitMission,
        priority: int,
        *,
        resource_cost: int = 0,
        resource_gain: int = 0,
        reason: str = "",
        target_id: UUID | None = None,
        target_position: Position | None = None,
        expected_cell: Position | None = None,
        unit_type: UnitType | None = None,
        tie_break: tuple[int, ...] = (),
        metadata: Metadata = (),
    ) -> ActionIntent:
        return cls(
            actor_id=actor_id,
            action=action,
            mission=mission,
            priority=priority,
            resource_cost=resource_cost,
            resource_gain=resource_gain,
            reason=reason,
            target_id=target_id,
            target_position=target_position,
            expected_cell=expected_cell,
            unit_type=unit_type,
            tie_break=tie_break,
            metadata=metadata,
        )

    @classmethod
    def move(
        cls,
        actor_id: UUID,
        mission: UnitMission,
        priority: int,
        direction: Direction,
        destination: Position,
        *,
        risk: int = 0,
        exclusive_destination: bool = False,
        tie_break: tuple[int, ...] = (),
        reason: str = "",
        metadata: Metadata = (),
    ) -> ActionIntent:
        return cls(
            actor_id=actor_id,
            action=IntentAction.MOVE,
            mission=mission,
            priority=priority,
            direction=direction,
            target_position=destination,
            risk=risk,
            reserve_positions=(destination,),
            exclusive_destination=exclusive_destination,
            tie_break=tie_break,
            reason=reason,
            metadata=metadata,
        )

    def sort_key(self) -> tuple[Any, ...]:
        actor_key = b"" if self.actor_id is None else self.actor_id.bytes
        return (
            self.priority,
            self.risk,
            self.tie_break,
            actor_key,
            self.action.value,
            self.reason,
        )


@dataclass(frozen=True, slots=True)
class RejectedIntent:
    intent: ActionIntent
    reason: str


@dataclass(frozen=True, slots=True)
class IntentResolution:
    selected: tuple[ActionIntent, ...]
    rejected: tuple[RejectedIntent, ...]
    reserved_positions: tuple[Position, ...] = ()
    resource_spent: int = 0
    resource_gained: int = 0

    def for_actor(self, actor_id: UUID | None) -> ActionIntent:
        for intent in self.selected:
            if intent.actor_id == actor_id:
                return intent
        raise KeyError(actor_id)


@dataclass(frozen=True, slots=True)
class CoreServiceQueue:
    service: str
    admission_id: UUID | None
    service_core_position: Position | None = None
    depositors: tuple[UUID, ...] = ()
    ready_depositors: tuple[UUID, ...] = ()
    approaching_depositors: tuple[UUID, ...] = ()
    holding_depositors: tuple[UUID, ...] = ()
    ready_ticks: tuple[tuple[UUID, int], ...] = ()
    queue_slots: tuple[tuple[UUID, Position], ...] = ()
    wounded: tuple[UUID, ...] = ()
    entrance: Position | None = None
    queue_cells: tuple[Position, ...] = ()
    exit_cell: Position | None = None
    reserved_resources: int = 0
    paused_reason: str | None = None
    previous_admission_id: UUID | None = None
    admission_reason: str | None = None
    release_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FireMission:
    target_id: UUID
    target_type: UnitType | None
    target_kind: str
    urgent: bool
    confidence: str
    candidate_cells: tuple[Position, ...]
    required_hits: int
    prediction_mode: str = "GENERIC"
    candidate_roles: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    split_fire: bool = False
    assigned_shooters: tuple[UUID, ...] = ()
    assignments: tuple[tuple[UUID, Position], ...] = ()


@dataclass(frozen=True, slots=True)
class VanguardIntentEstimate:
    target_id: UUID
    intent: VanguardIntent
    confidence: str
    candidate_cells: tuple[Position, ...]
    candidate_roles: tuple[str, ...]
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnemyRangerFireEstimate:
    target_id: UUID
    confidence: str
    current_cell: Position
    firing_position: Position | None
    candidate_cells: tuple[Position, ...]
    candidate_roles: tuple[str, ...]
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnemyActionEstimate:
    target_id: UUID
    confidence: str
    candidate_cells: tuple[Position, ...]
    candidate_roles: tuple[str, ...]
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScreeningGroupState:
    target_id: UUID
    vanguard_ids: tuple[UUID, UUID]
    ranger_ids: tuple[UUID, UUID]
    started_tick: int
    last_seen_tick: int
    last_distance: int
    outward_ticks: int = 0
    phase: str = "INTERCEPTING"


@dataclass(frozen=True, slots=True)
class CrisisForceBaseline:
    vanguards: int
    rangers: int
    started_tick: int | None
    phase: str = "SAFE"
    safe_ticks: int = 0


@dataclass(frozen=True, slots=True)
class ShotPlan:
    shooter_id: UUID
    target_id: UUID
    expected_cell: Position


@dataclass(frozen=True, slots=True)
class ShotFeedback:
    target_id: UUID
    expected_cell: Position
    misses: int
    suppressed_until: int


@dataclass(frozen=True, slots=True)
class MissionState:
    mission: UnitMission
    target: Position | None
    assigned_tick: int
    failures: int = 0


@dataclass(frozen=True, slots=True)
class WorkerEscapeState:
    phase: str
    threat_ids: tuple[UUID, ...]
    last_threat_tick: int
    safe_ticks: int = 0


@dataclass(frozen=True, slots=True)
class WorkerScoutState:
    """Durable, controller-free exploration lease for one Worker."""

    worker_id: UUID
    slot: int
    sector_index: int
    stage: int
    phase: WorkerScoutPhase
    target: Position | None
    assigned_tick: int
    best_route_cost: int | None = None
    stalled_ticks: int = 0
    backoff_until: int = 0
    last_scan_tick: int | None = None
    reachable_candidates: int = 0


@dataclass(frozen=True, slots=True)
class ManualMoveLease:
    direction: Direction
    expires_tick: int


@dataclass(frozen=True, slots=True)
class SquadState:
    vanguard_id: UUID
    ranger_id: UUID
    radius: int
    sector_index: int


@dataclass(frozen=True, slots=True)
class CoreEvacuationCampaign:
    active: bool
    started_tick: int | None
    safe_ticks: int
    last_destination: Position | None
    reason: str | None
