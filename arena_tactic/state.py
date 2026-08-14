from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID

from arena_hero import Position, UnitType

from .models import (
    CombatLossRecord,
    EnemyTrack,
    EnemyCoreIntel,
    EnemyCoreControlZone,
    CrisisForceBaseline,
    ManualMoveLease,
    MissionState,
    MoveAttempt,
    MoveFailure,
    PatientAdmissionProgress,
    PairingCooldown,
    PartnerDependencyFeedback,
    PeacefulFormationAssignment,
    RangerStanceLease,
    ScreeningContactDecision,
    WorkerTaskProgress,
    SquadRendezvousLease,
    SquadFormationLease,
    FormationMoveFeedback,
    FullStorageParkingAssignment,
    HomeReturnMission,
    SquadState,
    WorkerEscapeState,
    WorkerDisengageLease,
    WorkerPatrolMode,
    WorkerScoutState,
    ShotFeedback,
    ShotPlan,
    ScreeningGroupState,
    ServiceMoveFeedback,
    ServiceTransitProgress,
    ServiceTransitRoute,
    ServiceLaneLease,
    CargoRouteProgress,
    SegmentedReturnLease,
    ThreatHeatCell,
    LongRangeRaidCampaign,
    RaidAttemptMemory,
    RaidConfirmationLease,
    RaidDistanceBand,
    RaidReconMission,
    SiegeApproachPlan,
    VanguardInterceptLease,
)


@dataclass(slots=True)
class TacticMemory:
    """Persistent value-only memory.

    The class intentionally stores no Turn, Core, Unit or SDK controller.
    Everything here is serialisable geometry, identifiers, counters and small
    immutable tactical records.
    """

    core_id: UUID | None = None
    core_position: Position | None = None
    core_position_history: tuple[Position, ...] = ()
    last_tick: int | None = None

    known_obstacles: set[Position] = field(default_factory=set)
    known_passable: set[Position] = field(default_factory=set)
    cell_last_visible: dict[Position, int] = field(default_factory=dict)
    worker_cell_last_visible: dict[Position, int] = field(default_factory=dict)
    visit_counts: Counter[Position] = field(default_factory=Counter)
    congestion_counts: Counter[Position] = field(default_factory=Counter)
    last_congestion_decay_tick: int | None = None
    resource_memory: dict[Position, int] = field(default_factory=dict)
    resource_seen_count: Counter[Position] = field(default_factory=Counter)
    resource_harvest_count: Counter[Position] = field(default_factory=Counter)
    # Session-level outcome counters are intentionally transient.  They make
    # the live trace useful for tuning without polluting durable world memory.
    event_counts: Counter[str] = field(default_factory=Counter)

    enemy_tracks: dict[UUID, EnemyTrack] = field(default_factory=dict)
    enemy_core_intel: dict[UUID, EnemyCoreIntel] = field(default_factory=dict)
    enemy_core_control_zones: dict[UUID, EnemyCoreControlZone] = field(default_factory=dict)
    engaged_enemy_until: dict[UUID, int] = field(default_factory=dict)
    danger_until: dict[Position, int] = field(default_factory=dict)
    threat_heat: dict[Position, ThreatHeatCell] = field(default_factory=dict)
    unit_missions: dict[UUID, MissionState] = field(default_factory=dict)
    target_backoff_until: dict[Position, int] = field(default_factory=dict)
    position_history: dict[UUID, tuple[Position, ...]] = field(default_factory=dict)
    last_positions: dict[UUID, Position] = field(default_factory=dict)
    worker_escape_states: dict[UUID, WorkerEscapeState] = field(default_factory=dict)
    worker_disengage_leases: dict[UUID, WorkerDisengageLease] = field(default_factory=dict)
    worker_scout_states: dict[UUID, WorkerScoutState] = field(default_factory=dict)
    worker_task_progress: dict[UUID, WorkerTaskProgress] = field(default_factory=dict)
    worker_resource_backoff: dict[tuple[UUID, Position], int] = field(default_factory=dict)
    manual_move_leases: dict[UUID, ManualMoveLease] = field(default_factory=dict)
    last_move_attempts: dict[UUID, MoveAttempt] = field(default_factory=dict)
    failed_unit_moves: dict[UUID, MoveFailure] = field(default_factory=dict)
    last_ranger_shots: dict[UUID, ShotPlan] = field(default_factory=dict)
    last_vanguard_sweeps: dict[UUID, ShotPlan] = field(default_factory=dict)
    ranger_shot_feedback: dict[tuple[UUID, Position], ShotFeedback] = field(
        default_factory=dict
    )
    vanguard_sweep_feedback: dict[tuple[UUID, Position], ShotFeedback] = field(
        default_factory=dict
    )

    # Compatibility name retained for operators/checkpoints.  Since schema 7
    # this is the first Tick a loaded Worker entered the *current* near-Core
    # service zone (Core, queue line, or a cardinally adjacent cell), not the
    # Tick when it harvested cargo in the field.
    cargo_arrival_ticks: dict[UUID, int] = field(default_factory=dict)
    service_admission_id: UUID | None = None
    service_kind: str | None = None
    service_started_tick: int | None = None
    service_entrance: Position | None = None
    service_queue_cells: tuple[Position, ...] = ()
    service_exit_cell: Position | None = None
    service_egress_worker_ids: set[UUID] = field(default_factory=set)
    service_worker_progress: dict[UUID, tuple[Position, int]] = field(default_factory=dict)
    # Remaining executable cargo-route distance and consecutive non-progress
    # Ticks.  This is session-only and rebuilt after process restart.
    service_return_progress: dict[UUID, tuple[int | None, int]] = field(default_factory=dict)
    service_move_feedback: dict[UUID, ServiceMoveFeedback] = field(default_factory=dict)
    service_cargo_first_seen_ticks: dict[UUID, int] = field(default_factory=dict)
    service_deposit_ticks: dict[UUID, int] = field(default_factory=dict)
    service_lane_lease: ServiceLaneLease | None = None
    service_lane_change_ticks: tuple[int, ...] = ()
    service_all_unreachable_ticks: int = 0
    service_cargo_route_progress: dict[UUID, CargoRouteProgress] = field(
        default_factory=dict
    )
    service_segmented_returns: dict[UUID, SegmentedReturnLease] = field(
        default_factory=dict
    )
    # Session-only fairness age for actors that can physically enter the Core.
    # The unified service calendar is rebuilt from the authoritative Turn, but
    # this small value prevents maintenance patients from starving forever.
    service_ready_since_ticks: dict[UUID, int] = field(default_factory=dict)
    service_patient_gateways: dict[UUID, Position] = field(default_factory=dict)
    patient_admission_progress: PatientAdmissionProgress | None = None
    service_transit_progress: dict[UUID, ServiceTransitProgress] = field(
        default_factory=dict
    )
    service_transit_routes: dict[UUID, ServiceTransitRoute] = field(
        default_factory=dict
    )
    storage_saturated: bool = False
    worker_home_guard_targets: dict[UUID, Position] = field(default_factory=dict)
    worker_parking_assignments: dict[UUID, FullStorageParkingAssignment] = field(
        default_factory=dict
    )

    opening_complete: bool = False
    home_force_high_water: int = 12
    hostile_force_ids: set[UUID] = field(default_factory=set)
    home_safe_ticks: int = 0
    home_defense_alert_until: int = 0
    last_combat_unit_ids: set[UUID] = field(default_factory=set)
    recent_combat_loss_ticks: tuple[int, ...] = ()
    recent_combat_losses: tuple[CombatLossRecord, ...] = ()
    last_combat_unit_types: dict[UUID, UnitType] = field(default_factory=dict)
    crisis_force_baseline: CrisisForceBaseline | None = None

    evacuation_active: bool = False
    evacuation_started_tick: int | None = None
    evacuation_safe_ticks: int = 0
    evacuation_reason: str | None = None
    last_core_move_destination: Position | None = None
    failed_core_destinations: dict[Position, int] = field(default_factory=dict)
    strategic_relocation_pending: bool = False
    strategic_relocation_safe_ticks: int = 0
    strategic_relocation_goal: Position | None = None
    recent_home_threat_position: Position | None = None

    squad_states: dict[tuple[UUID, UUID], SquadState] = field(default_factory=dict)
    squad_rendezvous_leases: dict[tuple[UUID, UUID], SquadRendezvousLease] = field(
        default_factory=dict
    )
    squad_formation_leases: dict[tuple[UUID, UUID], SquadFormationLease] = field(
        default_factory=dict
    )
    formation_move_feedback: dict[UUID, FormationMoveFeedback] = field(
        default_factory=dict
    )
    partner_dependency_feedback: dict[UUID, PartnerDependencyFeedback] = field(
        default_factory=dict
    )
    squad_pairing_cooldowns: dict[tuple[UUID, UUID], PairingCooldown] = field(
        default_factory=dict
    )
    squad_target_backoffs: dict[
        tuple[UUID, UUID, Position, Position], int
    ] = field(default_factory=dict)
    peaceful_formation_assignment: PeacefulFormationAssignment | None = None
    screening_groups: dict[UUID, ScreeningGroupState] = field(default_factory=dict)
    screening_contact_decisions: dict[UUID, ScreeningContactDecision] = field(
        default_factory=dict
    )
    ranger_stance_leases: dict[tuple[UUID, UUID], RangerStanceLease] = field(
        default_factory=dict
    )
    vanguard_intercept_leases: dict[UUID, VanguardInterceptLease] = field(
        default_factory=dict
    )
    defense_sector_anchors: dict[str, tuple[Position, int]] = field(
        default_factory=dict
    )
    defense_reserve_leases: dict[UUID, tuple[Position, int, str]] = field(
        default_factory=dict
    )
    raid_target_id: UUID | None = None
    raid_last_seen_tick: int | None = None
    raid_last_position: Position | None = None
    raid_member_ids: tuple[UUID, ...] = ()
    raid_phase: str = "IDLE"
    raid_interrupted_tick: int | None = None
    raid_containment_mode: bool = False
    raid_long_range_campaign: LongRangeRaidCampaign | None = None
    raid_attempts: dict[UUID, RaidAttemptMemory] = field(default_factory=dict)
    raid_confirmation_lease: RaidConfirmationLease | None = None
    raid_recon_mission: RaidReconMission | None = None
    raid_distance_band: RaidDistanceBand | None = None
    raid_siege_approach: SiegeApproachPlan | None = None
    raid_return_reason: str | None = None
    raid_handoff_targets: dict[UUID, Position] = field(default_factory=dict)
    home_return_missions: dict[UUID, HomeReturnMission] = field(default_factory=dict)
    counter_siege_target_id: UUID | None = None
    counter_siege_last_seen_tick: int | None = None
    counter_siege_last_position: Position | None = None
    counter_siege_member_ids: tuple[UUID, ...] = ()
    counter_siege_reserve_ids: tuple[UUID, ...] = ()
    counter_siege_phase: str = "IDLE"
    beacon_mission_actor_id: UUID | None = None
    beacon_mission_target: Position | None = None

    worker_patrol_mode: WorkerPatrolMode = WorkerPatrolMode.INFORMATION_GAIN
    last_depleted_workers: frozenset[UUID] = frozenset()

    def reset_for_core(self, core_id: UUID, position: Position) -> None:
        """Reset Core-relative tactical state while retaining world knowledge."""

        self.core_id = core_id
        self.core_position = position
        self.core_position_history = (position,)
        self.unit_missions.clear()
        self.position_history.clear()
        self.last_positions.clear()
        self.worker_escape_states.clear()
        self.worker_disengage_leases.clear()
        self.worker_scout_states.clear()
        self.worker_task_progress.clear()
        self.worker_resource_backoff.clear()
        self.engaged_enemy_until.clear()
        self.manual_move_leases.clear()
        self.last_move_attempts.clear()
        self.failed_unit_moves.clear()
        self.last_ranger_shots.clear()
        self.last_vanguard_sweeps.clear()
        self.ranger_shot_feedback.clear()
        self.vanguard_sweep_feedback.clear()
        self.vanguard_intercept_leases.clear()
        self.cargo_arrival_ticks.clear()
        self.service_admission_id = None
        self.service_kind = None
        self.service_started_tick = None
        self.service_entrance = None
        self.service_queue_cells = ()
        self.service_exit_cell = None
        self.service_egress_worker_ids.clear()
        self.service_worker_progress.clear()
        self.service_return_progress.clear()
        self.service_move_feedback.clear()
        self.service_cargo_first_seen_ticks.clear()
        self.service_deposit_ticks.clear()
        self.service_lane_lease = None
        self.service_lane_change_ticks = ()
        self.service_all_unreachable_ticks = 0
        self.service_cargo_route_progress.clear()
        self.service_segmented_returns.clear()
        self.service_ready_since_ticks.clear()
        self.service_patient_gateways.clear()
        self.patient_admission_progress = None
        self.service_transit_progress.clear()
        self.service_transit_routes.clear()
        self.storage_saturated = False
        self.worker_home_guard_targets.clear()
        self.worker_parking_assignments.clear()
        self.opening_complete = False
        self.home_force_high_water = 12
        self.hostile_force_ids.clear()
        self.home_safe_ticks = 0
        self.home_defense_alert_until = 0
        self.last_combat_unit_ids.clear()
        self.recent_combat_loss_ticks = ()
        self.recent_combat_losses = ()
        self.last_combat_unit_types.clear()
        self.crisis_force_baseline = None
        self.evacuation_active = False
        self.evacuation_started_tick = None
        self.evacuation_safe_ticks = 0
        self.evacuation_reason = None
        self.last_core_move_destination = None
        self.failed_core_destinations.clear()
        self.strategic_relocation_pending = False
        self.strategic_relocation_safe_ticks = 0
        self.strategic_relocation_goal = None
        self.recent_home_threat_position = None
        self.squad_states.clear()
        self.squad_rendezvous_leases.clear()
        self.squad_formation_leases.clear()
        self.formation_move_feedback.clear()
        self.partner_dependency_feedback.clear()
        self.squad_pairing_cooldowns.clear()
        self.squad_target_backoffs.clear()
        self.peaceful_formation_assignment = None
        self.screening_groups.clear()
        self.screening_contact_decisions.clear()
        self.ranger_stance_leases.clear()
        self.defense_sector_anchors.clear()
        self.defense_reserve_leases.clear()
        self.raid_target_id = None
        self.raid_last_seen_tick = None
        self.raid_last_position = None
        self.raid_member_ids = ()
        self.raid_phase = "IDLE"
        self.raid_interrupted_tick = None
        self.raid_containment_mode = False
        self.raid_long_range_campaign = None
        self.raid_confirmation_lease = None
        self.raid_recon_mission = None
        self.raid_distance_band = None
        self.raid_siege_approach = None
        self.raid_attempts.clear()
        self.raid_return_reason = None
        self.raid_handoff_targets.clear()
        self.home_return_missions.clear()
        self.counter_siege_target_id = None
        self.counter_siege_last_seen_tick = None
        self.counter_siege_last_position = None
        self.counter_siege_member_ids = ()
        self.counter_siege_reserve_ids = ()
        self.counter_siege_phase = "IDLE"
        self.beacon_mission_actor_id = None
        self.beacon_mission_target = None

    @property
    def previous_positions(self) -> dict[UUID, Position]:
        """Compatibility view used by older operators and diagnostics."""

        return self.last_positions

    @property
    def current_resource_cells(self) -> frozenset[Position]:
        return frozenset(self.resource_memory)
