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
    RETURN_TO_BAND = "RETURN_TO_BAND"


class EnemyCoreControlLevel(str, Enum):
    """How remembered hostile-Core geometry may influence Worker movement."""

    HARD = "HARD"
    SOFT = "SOFT"
    STRATEGIC = "STRATEGIC"


class RaidDistanceBand(str, Enum):
    NEAR = "NEAR"
    EXTENDED = "EXTENDED"
    LONG_RANGE = "LONG_RANGE"


class UnitMission(str, Enum):
    CORE_SURVIVAL = "CORE_SURVIVAL"
    DEPOSIT = "DEPOSIT"
    ESCAPE = "ESCAPE"
    CORE_DISENGAGE = "CORE_DISENGAGE"
    ATTACK = "ATTACK"
    RECOVER = "RECOVER"
    RETURN_CARGO = "RETURN_CARGO"
    CLEAR_CORE = "CLEAR_CORE"
    CLEAR_SERVICE_CELL = "CLEAR_SERVICE_CELL"
    DECONFLICT_CELL = "DECONFLICT_CELL"
    HARVEST = "HARVEST"
    HOME_GUARD = "HOME_GUARD"
    FULL_STORAGE_STAGING = "FULL_STORAGE_STAGING"
    RETURN_TO_SCOUT_BAND = "RETURN_TO_SCOUT_BAND"
    HOME_DEFENSE = "HOME_DEFENSE"
    COUNTER_SIEGE = "COUNTER_SIEGE"
    RAID = "RAID"
    RETURN_HOME = "RETURN_HOME"
    EXPLORE = "EXPLORE"
    PATROL = "PATROL"
    PRODUCTION = "PRODUCTION"
    BEACON = "BEACON"
    WAIT = "WAIT"


class CoreServicePhase(str, Enum):
    """Physical lifecycle shared by deposits, healing and slot-using spawn."""

    APPROACHING = "APPROACHING"
    ENTRY = "ENTRY"
    SERVICE = "SERVICE"
    EGRESS = "EGRESS"


class ServiceTransitKind(str, Enum):
    """Why an actor is travelling through the shared Core service network."""

    DEPOSIT = "DEPOSIT"
    HEAL = "HEAL"
    DEPOSIT_THEN_HEAL = "DEPOSIT_THEN_HEAL"


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


class DestinationExclusivity(str, Enum):
    """Which occupants a movement reservation excludes at its destination."""

    NONE = "NONE"
    COMBAT = "COMBAT_EXCLUSIVE"
    SERVICE_TRANSIT = "SERVICE_TRANSIT"
    PHYSICAL = "PHYSICAL_EXCLUSIVE"


class VanguardIntent(str, Enum):
    ATTACKING = "ATTACKING"
    STATIONARY = "STATIONARY"
    RETREATING = "RETREATING"
    BLIND_SPOT_APPROACH = "BLIND_SPOT_APPROACH"
    UNCERTAIN = "UNCERTAIN"


class UnitLossProvenance(str, Enum):
    HOME_DEFENSE = "HOME_DEFENSE"
    RAID = "RAID"
    REMOTE = "REMOTE"
    UNKNOWN = "UNKNOWN"


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
    lifetime_sightings: int = 1
    confirmation_sightings: int = 1
    confirmation_window_start_tick: int | None = None


# Public name for the split strategic/current intelligence record introduced
# by checkpoint schema 16.  Keep EnemyCoreIntel as the compatibility spelling.
EnemyCoreIntelState = EnemyCoreIntel


@dataclass(frozen=True, slots=True)
class EnemyCoreControlZone:
    core_id: UUID
    center: Position
    exclusion_radius: int
    clear_radius: int
    last_seen_tick: int
    visible_now: bool
    expires_tick: int | None = None
    control_level: EnemyCoreControlLevel = EnemyCoreControlLevel.HARD


@dataclass(frozen=True, slots=True)
class BoundedScoutAssignment:
    worker_id: UUID
    radius: int
    sector_index: int
    target: Position | None


@dataclass(frozen=True, slots=True)
class RaidConfirmationLease:
    target_id: UUID
    observer_id: UUID
    first_seen_tick: int
    expires_tick: int


@dataclass(frozen=True, slots=True)
class RaidReconMission:
    target_id: UUID
    member_ids: tuple[UUID, ...]
    last_position: Position
    started_tick: int
    last_seen_tick: int
    no_progress_ticks: int = 0
    last_group_distance: int | None = None


@dataclass(frozen=True, slots=True)
class SiegeApproachPlan:
    target_id: UUID
    target_position: Position
    distance_band: RaidDistanceBand
    vanguard_positions: tuple[Position, ...]
    ranger_positions: tuple[Position, ...]
    route_eta: int


@dataclass(frozen=True, slots=True)
class WorkerDisengageLease:
    worker_id: UUID
    core_id: UUID
    center: Position
    waypoint: Position | None
    assigned_tick: int
    safe_ticks: int = 0
    last_distance: int = 0
    last_position: Position | None = None
    stalled_ticks: int = 0
    abandoned_target: Position | None = None


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
class ServiceMoveFeedback:
    worker_id: UUID
    tick: int
    destination: Position | None
    selected: bool
    rejection_reason: str | None
    stalled_ticks: int = 0


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
    destination_exclusivity: DestinationExclusivity = DestinationExclusivity.NONE
    # Compatibility input retained for older planners and embedders.  A plain
    # exclusive destination historically meant a combat/formation lease; it
    # must never shrink the official physical capacity for Workers.
    exclusive_destination: bool = False
    tie_break: tuple[int, ...] = ()
    reason: str = ""
    metadata: Metadata = ()

    def __post_init__(self) -> None:
        if (
            self.exclusive_destination
            and self.destination_exclusivity is DestinationExclusivity.NONE
        ):
            object.__setattr__(
                self,
                "destination_exclusivity",
                DestinationExclusivity.COMBAT,
            )

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
        destination_exclusivity: DestinationExclusivity = DestinationExclusivity.NONE,
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
            destination_exclusivity=destination_exclusivity,
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
class CargoReturnReservation:
    """A cargo Worker's executable route and future Core appointment.

    ``route_distance`` counts movement Ticks until the Worker is on the Core;
    the following Tick is the scheduled DEPOSIT action.  Keeping the first
    route step beside the appointment prevents the service calendar and the
    Worker executor from planning against different threat maps.
    """

    worker_id: UUID
    route_target: Position | None
    route_distance: int | None
    first_direction: Direction | None
    first_position: Position | None
    earliest_deposit_tick: int | None
    scheduled_deposit_tick: int | None
    departure_tick: int | None
    slack_ticks: int | None
    status: str
    delay_reason: str | None = None
    route_mode: str = "FULL"
    waypoint: Position | None = None
    lane_version: int = 0
    previous_scheduled_tick: int | None = None
    schedule_change_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ServiceLaneLease:
    """Stable Core logistics geometry for one Core lifecycle."""

    core_id: UUID
    core_position: Position
    entrance: Position | None
    queue_cells: tuple[Position, ...]
    exit_cell: Position | None
    established_tick: int
    version: int = 1
    invalidation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CargoRouteProgress:
    """Per-Worker return progress; it may never invalidate a global lane."""

    worker_id: UUID
    lane_version: int
    last_position: Position
    remaining_distance: int | None
    previous_position: Position | None = None
    stalled_ticks: int = 0
    ping_pong_ticks: int = 0
    last_rejection_reason: str | None = None
    scheduled_deposit_tick: int | None = None


@dataclass(frozen=True, slots=True)
class SegmentedReturnLease:
    """Sticky bounded-search waypoint used when a full route exceeds budget."""

    worker_id: UUID
    lane_version: int
    waypoint: Position
    established_tick: int
    last_position: Position
    remaining_distance: int
    stalled_ticks: int = 0


@dataclass(frozen=True, slots=True)
class CoreServiceJob:
    """One actor's indivisible visit to the single Unit slot on the Core.

    A wounded loaded Worker owns one job with two operations rather than a
    cargo appointment competing with an unrelated treatment appointment.
    ``service_tick`` is the first operation Tick; subsequent operations occupy
    consecutive Ticks and ``exit_tick`` is the following egress Tick.
    """

    actor_id: UUID | None
    operations: tuple[str, ...]
    phase: CoreServicePhase
    route_distance: int | None
    first_direction: Direction | None
    first_position: Position | None
    gateway: Position | None
    earliest_service_tick: int | None
    service_tick: int | None
    exit_tick: int | None
    priority: int
    ready_since_tick: int | None
    resource_cost: int = 0
    resource_gain: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ServiceTransitRoute:
    """Executable, shared route projection for a Core service actor."""

    actor_id: UUID
    kind: ServiceTransitKind
    target: Position
    route_distance: int | None
    options: tuple[tuple[Direction, Position, int], ...] = ()
    service_tick: int | None = None
    exit_tick: int | None = None


@dataclass(frozen=True, slots=True)
class ServiceTransitProgress:
    """Resolver feedback for either cargo return or patient recovery transit."""

    actor_id: UUID
    kind: ServiceTransitKind
    tick: int
    destination: Position | None
    remaining_distance: int | None
    selected: bool
    stalled_ticks: int = 0
    rejection_reason: str | None = None
    shared_with_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class CoreSlotSchedule:
    """Work-conserving calendar for the Core's single Unit service slot."""

    tick: int
    jobs: tuple[CoreServiceJob, ...] = ()
    current_job_id: UUID | None = None
    next_job_id: UUID | None = None
    slot_owner_id: UUID | None = None
    slot_reserved: bool = False
    production_allowed: bool = True
    spawn_egress_cell: Position | None = None
    reason: str = "NO_SERVICE_JOB"


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
    overflow_slots: tuple[tuple[UUID, Position], ...] = ()
    scheduled_deposits: tuple[tuple[UUID, int], ...] = ()
    return_reservations: tuple[CargoReturnReservation, ...] = ()
    worker_progress: tuple[tuple[UUID, Position, int], ...] = ()
    wounded: tuple[UUID, ...] = ()
    entrance: Position | None = None
    queue_cells: tuple[Position, ...] = ()
    exit_cell: Position | None = None
    patient_gateway: Position | None = None
    core_slot_reserved: bool = False
    timeline: "CoreOperationTimeline | None" = None
    patient_progress: "PatientAdmissionProgress | None" = None
    service_windows: tuple["CoreServiceWindow", ...] = ()
    patient_queue: tuple["PatientQueueEntry", ...] = ()
    service_cell_leases: tuple["ServiceCellLease", ...] = ()
    jobs: tuple[CoreServiceJob, ...] = ()
    slot_schedule: CoreSlotSchedule | None = None
    blocking_units: tuple[tuple[UUID, Position, str], ...] = ()
    reschedule_reasons: tuple[str, ...] = ()
    reserved_resources: int = 0
    paused_reason: str | None = None
    previous_admission_id: UUID | None = None
    admission_reason: str | None = None
    release_reason: str | None = None
    lane_lease: ServiceLaneLease | None = None
    lane_replan_reason: str | None = None
    liveness_indicators: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PatientAdmissionProgress:
    patient_id: UUID
    gateway: Position | None
    started_tick: int
    last_position: Position
    stalled_ticks: int = 0
    entry_distance: int | None = None


@dataclass(frozen=True, slots=True)
class CoreOperationRequest:
    actor_id: UUID | None
    operation: str
    eta: int
    occupy_tick: int
    release_tick: int
    priority: int
    resource_cost: int = 0
    resource_gain: int = 0
    gateway: Position | None = None


@dataclass(frozen=True, slots=True)
class CoreOperationTimeline:
    tick: int
    requests: tuple[CoreOperationRequest, ...] = ()
    current_slot_owner: UUID | None = None
    current_slot_reserved: bool = False
    next_service_eta: int | None = None
    next_service_tick: int | None = None
    next_release_tick: int | None = None
    production_allowed: bool = True
    spawn_egress_cell: Position | None = None
    reason: str = "NO_CURRENT_SERVICE"


@dataclass(frozen=True, slots=True)
class CoreServiceWindow:
    actor_id: UUID
    operation: str
    enter_tick: int
    service_tick: int
    exit_tick: int
    gateway: Position | None = None
    status: str = "FUTURE"


@dataclass(frozen=True, slots=True)
class PatientQueueEntry:
    patient_id: UUID
    urgent: bool
    hp_percent: int
    eta: int | None
    gateway: Position | None
    stalled_ticks: int
    resource_cost: int
    status: str


@dataclass(frozen=True, slots=True)
class ServiceCellLease:
    cell: Position
    purpose: str
    owner_id: UUID | None
    start_tick: int
    end_tick: int
    active: bool


@dataclass(frozen=True, slots=True)
class WorkerTaskProgress:
    worker_id: UUID
    target: Position
    route_distance: int | None
    last_progress_tick: int
    stalled_ticks: int = 0
    rejection_reason: str | None = None
    backoff_until: int | None = None


@dataclass(frozen=True, slots=True)
class HomeCounterSiegeDecision:
    phase: str = "IDLE"
    target_id: UUID | None = None
    target_position: Position | None = None
    member_ids: tuple[UUID, ...] = ()
    reserve_ids: tuple[UUID, ...] = ()
    last_seen_tick: int | None = None
    reason: str = "NO_LOCAL_ENEMY_CORE"


@dataclass(frozen=True, slots=True)
class VanguardInterceptTask:
    vanguard_id: UUID
    target_id: UUID
    sector: Direction
    phase: str
    intercept_cell: Position
    candidate_cells: tuple[Position, ...]
    cost: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VanguardInterceptLease:
    vanguard_id: UUID
    target_id: UUID
    intercept_cell: Position
    assigned_tick: int
    last_route_distance: int
    no_progress_ticks: int = 0
    invalidation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class VanguardAssignmentCandidate:
    vanguard_id: UUID
    target_id: UUID
    cost: tuple[int, ...] | None
    selected: bool
    reason: str


@dataclass(frozen=True, slots=True)
class HomeCombatAssignment:
    tasks: tuple[VanguardInterceptTask, ...] = ()
    candidates: tuple[VanguardAssignmentCandidate, ...] = ()
    unassigned_vanguards: tuple[UUID, ...] = ()
    uncovered_targets: tuple[UUID, ...] = ()

    @property
    def assigned_vanguard_ids(self) -> frozenset[UUID]:
        return frozenset(task.vanguard_id for task in self.tasks)


@dataclass(frozen=True, slots=True)
class HostileApproachEstimate:
    target_id: UUID
    candidate_cells: tuple[Position, ...]
    path_next_cells: tuple[Position, ...]
    path_costs: tuple[tuple[Position, int], ...]
    protected_targets: tuple[Position, ...]
    evidence: tuple[str, ...] = ()


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
    stationary_ticks: int = 0


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
class RangerStanceOption:
    ranger_id: UUID
    target_id: UUID
    role: str
    stance: Position
    first_direction: Direction | None
    first_position: Position | None
    route_distance: int
    visible_candidates: tuple[Position, ...]
    firing_candidates: tuple[Position, ...]
    risk: int
    viable: bool = True
    rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ScreeningContactDecision:
    target_id: UUID
    target_visible: bool
    candidate_cells: tuple[Position, ...]
    contact_ranger_id: UUID | None
    fire_support_ranger_id: UUID | None
    options: tuple[RangerStanceOption, ...] = ()
    visible_before: int = 0
    visible_after: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RangerStanceLease:
    target_id: UUID
    ranger_id: UUID
    role: str
    stance: Position
    assigned_tick: int
    expires_tick: int
    last_position: Position
    last_route_distance: int
    no_progress_ticks: int = 0
    last_direction: Direction | None = None


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
    last_evidence_tick: int | None = None
    release_reason: str | None = None
    last_attempt_tick: int | None = None


@dataclass(frozen=True, slots=True)
class MissionState:
    mission: UnitMission
    target: Position | None
    assigned_tick: int
    failures: int = 0


@dataclass(frozen=True, slots=True)
class WorkerSurvivalLease:
    phase: str
    threat_ids: tuple[UUID, ...]
    last_threat_tick: int
    safe_ticks: int = 0
    waypoint: Position | None = None
    last_min_enemy_distance: int | None = None
    stalled_ticks: int = 0
    loop_period: int | None = None
    route_version: int = 0
    waypoint_assigned_tick: int | None = None
    waypoint_expires_tick: int | None = None
    waypoint_invalid_reason: str | None = None
    last_waypoint_distance: int | None = None
    control_core_ids: tuple[UUID, ...] = ()
    control_centers: tuple[Position, ...] = ()


# Compatibility alias for callers and old tests that imported the split-state
# name.  WorkerSurvivalLease is the single authoritative survival state.
WorkerEscapeState = WorkerSurvivalLease


@dataclass(frozen=True, slots=True)
class RaidAttemptMemory:
    core_id: UUID
    failed_attempts: int = 0
    last_failure_tick: int | None = None
    last_failure_reason: str | None = None
    last_failure_sighting_tick: int | None = None


@dataclass(frozen=True, slots=True)
class HomeReturnMission:
    actor_id: UUID
    target: Position
    assigned_tick: int
    last_distance: int | None = None
    stalled_ticks: int = 0


@dataclass(frozen=True, slots=True)
class CombatLossRecord:
    actor_id: UUID
    unit_type: UnitType
    tick: int
    provenance: UnitLossProvenance


@dataclass(frozen=True, slots=True)
class FullStorageParkingAssignment:
    worker_id: UUID
    position: Position
    zone: str
    assigned_tick: int


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
    patrol_anchor: Position | None = None
    support_target: Position | None = None
    target_assigned_tick: int | None = None


@dataclass(frozen=True, slots=True)
class SquadRendezvousLease:
    vanguard_id: UUID
    ranger_id: UUID
    rendezvous: Position
    assigned_tick: int
    best_separation: int
    best_route_distance: int | None = None
    stalled_ticks: int = 0
    last_vanguard_position: Position | None = None
    last_ranger_position: Position | None = None


@dataclass(frozen=True, slots=True)
class SquadFormationBundle:
    """One globally reservable peaceful formation for a mixed squad."""

    vanguard_id: UUID
    ranger_id: UUID
    vanguard_origin: Position
    ranger_origin: Position
    anchor: Position
    support: Position
    vanguard_route_distance: int
    ranger_route_distance: int
    vanguard_first_direction: Direction | None = None
    vanguard_first_position: Position | None = None
    ranger_first_direction: Direction | None = None
    ranger_first_position: Position | None = None
    score: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PeacefulFormationAssignment:
    """Per-Tick global formation result, detached from SDK controllers."""

    tick: int
    bundles: tuple[SquadFormationBundle, ...] = ()
    reserved_positions: tuple[Position, ...] = ()
    unassigned_squads: tuple[tuple[UUID, UUID], ...] = ()
    rejected: tuple[tuple[UUID, UUID, Position, Position, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SquadFormationLease:
    """Authoritative progress for one patrol bundle across Turns."""

    vanguard_id: UUID
    ranger_id: UUID
    anchor: Position
    support: Position
    assigned_tick: int
    last_evaluated_tick: int
    vanguard_best_distance: int | None
    ranger_best_distance: int | None
    vanguard_arrived: bool = False
    ranger_arrived: bool = False
    stalled_ticks: int = 0
    blocked_ticks: int = 0
    partner_hold_ticks: int = 0
    partner_progressing: bool = False
    last_vanguard_position: Position | None = None
    last_ranger_position: Position | None = None
    last_rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FormationMoveFeedback:
    """Resolver outcome consumed by the next Turn's formation planner."""

    actor_id: UUID
    tick: int
    action: str
    reason: str
    target_position: Position | None
    rejection_reason: str | None = None
    consecutive_blocked_ticks: int = 0
    consecutive_partner_wait_ticks: int = 0


@dataclass(frozen=True, slots=True)
class PartnerDependencyFeedback:
    actor_id: UUID
    partner_id: UUID
    tick: int
    reason: str
    remaining_route_distance: int | None
    resolver_accepted: bool
    wait_ticks: int = 0


@dataclass(frozen=True, slots=True)
class LongRangeRaidCampaign:
    target_id: UUID
    member_ids: tuple[UUID, ...]
    phase: str
    started_tick: int
    route_eta: int
    search_deadline_tick: int
    last_position: Position
    last_group_distance: int | None = None
    no_progress_ticks: int = 0


@dataclass(frozen=True, slots=True)
class PairingCooldown:
    vanguard_id: UUID
    ranger_id: UUID
    expires_tick: int
    reason: str = "STALLED_REASSEMBLY"


@dataclass(frozen=True, slots=True)
class CoreMoveCandidateEvaluation:
    direction: Direction
    destination: Position
    forward_exits: int
    local_open: bool
    unknown_frontier: bool
    service_exits: int
    viable: bool
    rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CoreEvacuationCampaign:
    active: bool
    started_tick: int | None
    safe_ticks: int
    last_destination: Position | None
    reason: str | None
    candidate_evaluations: tuple[CoreMoveCandidateEvaluation, ...] = ()
    no_escape_route: bool = False
