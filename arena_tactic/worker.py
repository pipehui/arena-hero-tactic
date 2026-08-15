from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, pi
from uuid import UUID

from arena_hero import CoreState, Direction, Position, UnitType

from .config import TacticConfig
from .geometry import cardinal_neighbors, diamond, manhattan, manhattan_ring
from .models import (
    ActionIntent,
    CoreServiceQueue,
    EconomyPolicyDecision,
    EntitySnapshot,
    EnemyCoreControlZone,
    EnemyCoreControlLevel,
    FullStorageParkingAssignment,
    IntentAction,
    MissionState,
    ResourceWorkOrder,
    ResourceSearchLease,
    UnitMission,
    WorkerEconomyMode,
    WorkerSurvivalLease,
    WorkerDisengageLease,
    WorkerScoutPhase,
    WorkerScoutState,
    WorkerTaskProgress,
    WorldModel,
    RaidDistanceBand,
    ScoutReturnRouteLease,
)
from .planning import (
    MoveViability,
    exploration_candidates,
    information_gain,
    move_viability,
    Route,
    route_from_field,
    sector_scout_candidates,
    weighted_distance_field,
    weighted_progress_route,
    weighted_route_to,
    route_to,
    scout_visible_cells,
    scout_sector_index,
    siege_approach_plan,
)
from .projection import TacticalMap
from .resource_allocator import ResourceAllocator
from .rules import UNIT_MAX_HP
from .state import TacticMemory
from .service_transit import CoreServiceTransitPlanner
from .worker_safety import WorkerSafetyEvaluator


@dataclass(frozen=True, slots=True)
class _EscapeCandidate:
    direction: Direction
    destination: Position
    viability: MoveViability
    immediate_attackers: int
    future_attackers: int
    recent_threat: int
    heat: int
    minimum_enemy_distance: int
    total_enemy_distance: int
    visible_minimum_distance: int
    visible_total_distance: int
    visible_vanguard_minimum_distance: int
    survival_terminals: int
    forward_exits: int
    waypoint: Position
    waypoint_minimum_distance: int
    waypoint_total_distance: int
    waypoint_heat: int
    direction_index: int
    recently_visited: bool
    backtracking: bool
    lease_distance: int

    @property
    def score(self) -> tuple[int, ...]:
        return (
            self.immediate_attackers,
            self.future_attackers,
            -self.visible_minimum_distance,
            -self.visible_total_distance,
            -self.minimum_enemy_distance,
            -self.total_enemy_distance,
            -self.waypoint_minimum_distance,
            -self.waypoint_total_distance,
            -self.survival_terminals,
            0 if self.forward_exits >= 2 else 1,
            self.heat,
            self.waypoint_heat,
            self.recent_threat,
            self.lease_distance,
            int(self.recently_visited),
            int(self.backtracking),
            self.direction_index,
        )


class WorkerPlanner:
    """Worker survival, logistics, harvesting and information-gain exploration."""

    def __init__(
        self,
        config: TacticConfig,
        memory: TacticMemory,
        resources: ResourceAllocator,
    ) -> None:
        self.config = config
        self.memory = memory
        self.resources = resources
        self.safety = WorkerSafetyEvaluator()
        self.transit = CoreServiceTransitPlanner(config)

    def intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        economy_policy: EconomyPolicyDecision,
    ) -> list[ActionIntent]:
        if world.core is None:
            return []
        workers = tuple(
            unit for unit in world.friendlies if unit.unit_type is UnitType.WORKER
        )
        worker_ids = {worker.id for worker in workers}
        self.memory.worker_economy_modes.clear()
        for worker_id in tuple(self.memory.resource_work_orders):
            worker = next((item for item in workers if item.id == worker_id), None)
            if (
                worker is None
                or worker.cargo > 0
                or worker.hp < UNIT_MAX_HP[UnitType.WORKER]
            ):
                self.memory.resource_work_orders.pop(worker_id, None)
        for worker_id in tuple(self.memory.resource_work_orders):
            if worker_id not in worker_ids:
                self.memory.resource_work_orders.pop(worker_id, None)
        self._refresh_enemy_core_control_zones(world)
        intents, escaping = self._survival_intents(world, projection, service, workers)
        deconflicting: set[UUID] = set()
        home_defense_active = (
            self.memory.home_defense_alert_until >= world.tick
            or any(
                enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                and manhattan(enemy.position, world.core.position)
                <= self.config.home_engage_radius + 4
                for enemy in world.enemies
            )
        )
        if home_defense_active:
            by_position: dict[Position, list[EntitySnapshot]] = {}
            for worker in workers:
                by_position.setdefault(worker.position, []).append(worker)
            service_ids = set(service.depositors) | set(service.wounded)
            for group in by_position.values():
                if len(group) < 2:
                    continue
                stayer = min(
                    group,
                    key=lambda worker: (
                        worker.id != service.admission_id,
                        worker.id not in service_ids,
                        worker.id.bytes,
                    ),
                )
                for worker in sorted(group, key=lambda item: item.id.bytes):
                    if (
                        worker.id == stayer.id
                        or worker.id in escaping
                        or worker.id in service_ids
                        or worker.cargo > 0
                        or worker.hp < UNIT_MAX_HP[UnitType.WORKER]
                    ):
                        continue
                    intents.extend(
                        self._wartime_deconflict_intents(
                            world,
                            projection,
                            service,
                            worker,
                        )
                    )
                    deconflicting.add(worker.id)
        guarding: dict[UUID, Position] = {}
        service_ids = set(service.depositors)
        guard_eligible = tuple(
            worker
            for worker in workers
            if worker.id not in escaping and worker.id not in deconflicting
            and worker.hp >= UNIT_MAX_HP[UnitType.WORKER]
            and worker.id != world.beacon.carrier_id
            and worker.id not in service_ids
        )
        carriers = tuple(worker for worker in guard_eligible if worker.cargo > 0)
        if self.memory.storage_saturated:
            # The storage latch applies to cargo because the Core cannot accept
            # it yet.  Empty Workers only inherit this latch after the mature
            # economy is complete; temporary fullness must not interrupt work.
            guarding = self._home_guard_assignments(
                world,
                projection,
                service,
                carriers,
            )
            self.memory.worker_economy_modes.update(
                {
                    worker.id: WorkerEconomyMode.FULL_STORAGE_STAGING
                    for worker in carriers
                }
            )
        else:
            self.memory.worker_home_guard_targets.clear()
            self.memory.worker_parking_assignments.clear()

        if economy_policy.saturated_patrol_active:
            self._suspend_resource_search_leases(workers)
            self.memory.resource_work_orders.clear()
            self.memory.resource_candidate_counts.clear()
            self.memory.resource_rejection_counts.clear()
            for worker_id, mission in tuple(self.memory.unit_missions.items()):
                if mission.mission is UnitMission.HARVEST:
                    self.memory.unit_missions.pop(worker_id, None)
                    self.memory.worker_task_progress.pop(worker_id, None)
            resource_assignments: dict[UUID, Position] = {}
            exploring = tuple(
                worker
                for worker in guard_eligible
                if worker.cargo == 0 and worker.id not in guarding
            )
            for worker in (*carriers, *exploring):
                self.memory.worker_task_progress.pop(worker.id, None)
                mission = self.memory.unit_missions.get(worker.id)
                if mission is not None and mission.mission in {
                    UnitMission.HARVEST,
                    UnitMission.HOME_GUARD,
                    UnitMission.FULL_STORAGE_STAGING,
                }:
                    self.memory.unit_missions.pop(worker.id, None)
            exploration_assignments = self._exploration_assignments(
                world,
                projection,
                exploring,
                service,
            )
            self.memory.worker_economy_modes.update(
                {
                    worker.id: WorkerEconomyMode.SATURATED_PATROL
                    for worker in exploring
                }
            )
        else:
            # Legacy ring missions belong exclusively to mature saturated
            # patrol.  They must never pull an active resource searcher back
            # toward the 20/25/30 belt.
            self.memory.scout_return_route_leases.clear()
            for worker_id, mission in tuple(self.memory.unit_missions.items()):
                if mission.mission in {
                    UnitMission.EXPLORE,
                    UnitMission.RETURN_TO_SCOUT_BAND,
                }:
                    self.memory.unit_missions.pop(worker_id, None)
            available = tuple(
                worker
                for worker in workers
                if worker.id not in escaping and worker.id not in deconflicting
                and worker.hp >= UNIT_MAX_HP[UnitType.WORKER]
                and worker.cargo == 0
                and worker.id != world.beacon.carrier_id
            )
            resource_assignments, exploration_assignments = self._assign_work(
                world, projection, service, available
            )
            self.memory.worker_economy_modes.update(
                {
                    worker_id: WorkerEconomyMode.RESOURCE_ACQUISITION
                    for worker_id in resource_assignments
                }
            )
            self.memory.worker_economy_modes.update(
                {
                    worker_id: WorkerEconomyMode.RESOURCE_SEARCH
                    for worker_id in (
                        worker.id
                        for worker in available
                        if worker.id not in resource_assignments
                    )
                }
            )

        for worker in workers:
            if worker.id in escaping or worker.id in deconflicting:
                continue
            if worker.id in guarding:
                intents.extend(
                    self._home_guard_intents(
                        world,
                        projection,
                        service,
                        worker,
                        guarding[worker.id],
                    )
                )
                continue
            if worker.hp < UNIT_MAX_HP[UnitType.WORKER] and worker.cargo == 0:
                # RecoveryPlanner owns the sticky return/heal task.  Excluding
                # the actor here prevents resource or exploration intent
                # generation from silently replacing it when enemies fog.
                continue
            intents.extend(
                self._routine_intents(
                    world,
                    projection,
                    service,
                    worker,
                    resource_assignments.get(worker.id),
                    exploration_assignments.get(worker.id),
                    allow_harvest=not economy_policy.saturated_patrol_active,
                )
            )
        return intents

    def _wartime_deconflict_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        worker: EntitySnapshot,
    ) -> list[ActionIntent]:
        """Separate an existing Worker stack without blocking combat traffic."""

        assert world.core is not None
        occupied = dict(world.occupied_cells)
        service_cells = {
            cell
            for cell in (
                world.core.position,
                service.entrance,
                service.exit_cell,
                *service.queue_cells,
            )
            if cell is not None
        }
        rows = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(worker.position)
        ):
            if (
                destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or destination in service_cells
                or projection.immediate_attackers(destination) >= worker.hp
                or occupied.get(destination, 0) >= 2
            ):
                continue
            viability = move_viability(
                world,
                worker.position,
                destination,
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            )
            if not viability.viable:
                continue
            score = (
                occupied.get(destination, 0),
                projection.future_attackers(destination),
                projection.threat_heat.get(destination, 0),
                -viability.forward_exits,
                index,
            )
            rows.append((score, direction, destination, viability))
        intents = [
            ActionIntent.move(
                worker.id,
                UnitMission.DECONFLICT_CELL,
                47,
                direction,
                destination,
                risk=score[1] * 100 + score[2],
                tie_break=score,
                reason="WARTIME_WORKER_DECONFLICT",
                metadata=viability.metadata,
            )
            for score, direction, destination, viability in sorted(rows)
        ]
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.DECONFLICT_CELL,
                48,
                reason="WARTIME_WORKER_DECONFLICT_BLOCKED",
            )
        )
        return intents

    def _home_guard_assignments(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        workers: tuple[EntitySnapshot, ...],
    ) -> dict[UUID, Position]:
        """Return every saturated carrier to one bounded economic belt."""

        assert world.core is not None
        living = {worker.id for worker in workers}
        for worker_id in tuple(self.memory.worker_home_guard_targets):
            if worker_id not in living:
                self.memory.worker_home_guard_targets.pop(worker_id, None)
                self.memory.worker_parking_assignments.pop(worker_id, None)
        if not workers:
            return {}

        core = world.core.position
        service_cells = {
            cell
            for cell in (
                core,
                service.service_core_position,
                service.entrance,
                service.exit_cell,
                *service.queue_cells,
                *(cell for _, cell in cardinal_neighbors(core)),
            )
            if cell is not None
        }
        combat_positions = {
            unit.position
            for unit in world.friendlies
            if unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
        }
        formation_positions = set()
        if self.memory.peaceful_formation_assignment is not None:
            formation_positions.update(
                self.memory.peaceful_formation_assignment.reserved_positions
            )
        for lease in self.memory.squad_formation_leases.values():
            formation_positions.update((lease.anchor, lease.support))
        for _, (cell, _, _) in self.memory.defense_reserve_leases.items():
            formation_positions.add(cell)

        def legal_cells(radii: tuple[int, ...]) -> tuple[Position, ...]:
            return tuple(
                cell
                for radius in radii
                for cell in manhattan_ring(core, radius)
                if cell in world.known_passable
                and cell not in world.known_obstacles
                and cell not in projection.hostile_occupied
                and cell not in service_cells
                and cell not in combat_positions
                and cell not in formation_positions
                and projection.immediate_attackers(cell) == 0
                and projection.future_attackers(cell) == 0
                and projection.threat_heat.get(cell, 0) == 0
            )

        primary_candidates = legal_cells(
            tuple(
                range(
                    self.config.worker_full_storage_parking_min_radius,
                    self.config.worker_full_storage_parking_max_radius + 1,
                )
            )
        )
        fallback_candidates = legal_cells((13, 14))
        if not primary_candidates and not fallback_candidates:
            self.memory.worker_home_guard_targets.clear()
            self.memory.worker_parking_assignments.clear()
            return {}

        ordered_workers = tuple(sorted(workers, key=lambda unit: unit.id.bytes))
        used: set[Position] = set()
        assignments: dict[UUID, Position] = {}

        primary_set = set(primary_candidates)
        all_candidates = primary_candidates + tuple(
            cell for cell in fallback_candidates if cell not in primary_set
        )
        # A carrier already resting in a legal 8--12 cell is stable by
        # definition.  Preserve it before considering older assignments so a
        # remap never pulls it inward and then pushes it out again.
        for worker in ordered_workers:
            if worker.position in primary_set and worker.position not in used:
                assignments[worker.id] = worker.position
                used.add(worker.position)
        for worker in ordered_workers:
            if worker.id in assignments:
                continue
            previous = self.memory.worker_home_guard_targets.get(worker.id)
            if previous in all_candidates and previous not in used:
                assignments[worker.id] = previous
                used.add(previous)

        for rank, worker in enumerate(ordered_workers):
            if worker.id in assignments:
                continue
            candidates = tuple(cell for cell in primary_candidates if cell not in used)
            if not candidates:
                candidates = tuple(cell for cell in fallback_candidates if cell not in used)
            if not candidates:
                continue
            desired = rank * len(candidates) // max(1, len(ordered_workers))

            def score(row: tuple[int, Position]):
                index, cell = row
                cyclic = min(
                    (index - desired) % len(candidates),
                    (desired - index) % len(candidates),
                )
                immediate, future, remembered = projection.worker_exposure(cell)
                return (
                    immediate,
                    future,
                    remembered,
                    cyclic,
                    self.memory.congestion_counts.get(cell, 0),
                    manhattan(worker.position, cell),
                    cell,
                )

            available = enumerate(candidates)
            selected = min(available, key=score, default=None)
            if selected is None:
                continue
            _, target = selected
            assignments[worker.id] = target
            used.add(target)

        self.memory.worker_home_guard_targets = dict(assignments)
        self.memory.worker_parking_assignments = {
            worker_id: FullStorageParkingAssignment(
                worker_id=worker_id,
                position=target,
                zone=(
                    "CARGO_STAGING"
                    if manhattan(core, target)
                    <= self.config.worker_full_storage_parking_max_radius
                    else "CARGO_STAGING_FALLBACK"
                ),
                assigned_tick=(
                    previous.assigned_tick
                    if (
                        (previous := self.memory.worker_parking_assignments.get(worker_id))
                        is not None
                        and previous.position == target
                        and previous.zone
                        in {"CARGO_STAGING", "CARGO_STAGING_FALLBACK"}
                    )
                    else world.tick
                ),
            )
            for worker_id, target in assignments.items()
        }
        return assignments

    def _home_guard_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        worker: EntitySnapshot,
        target: Position,
    ) -> list[ActionIntent]:
        assert world.core is not None
        self.memory.unit_missions[worker.id] = MissionState(
            UnitMission.FULL_STORAGE_STAGING,
            target,
            world.tick,
        )
        if worker.position == target:
            return [
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.FULL_STORAGE_STAGING,
                    68,
                    target_position=target,
                    reason="FULL_STORAGE_STAGING_HOLD",
                    metadata=(("staging_post", target),),
                )
            ]
        route = self._route(
            world,
            projection,
            worker,
            target,
            service,
            logistics=True,
            extra_blocked=frozenset(
                {world.core.position}
                if worker.position != world.core.position
                else set()
            ),
        )
        blocked, _ = self._exploration_navigation(
            world,
            projection,
            worker,
            service,
        )
        route_viability = None
        if route is not None and route.first_position is not None:
            route_viability = self._worker_move_viability(
                world,
                worker.position,
                route.first_position,
                target=target,
                blocked=blocked,
            )
        if (
            route is not None
            and route.first_direction is not None
            and route_viability is not None
            and route_viability.viable
            and self._manual_allowed(worker.id, route.first_direction, world.tick)
        ):
            intents = [
                ActionIntent.move(
                    worker.id,
                    UnitMission.FULL_STORAGE_STAGING,
                    68,
                    route.first_direction,
                    route.first_position,
                    risk=self._risk(projection, route.first_position),
                    exclusive_destination=True,
                    tie_break=(route.distance,),
                    reason="FULL_STORAGE_RETURN_TO_STAGING",
                    metadata=(
                        ("staging_post", target),
                        ("allow_protected", True),
                    ) + route_viability.metadata,
                )
            ]
        else:
            intents = []

        service_cells = {
            cell
            for cell in (
                world.core.position,
                service.service_core_position,
                service.entrance,
                service.exit_cell,
                *service.queue_cells,
            )
            if cell is not None and cell != worker.position
        }
        options = []
        occupied = dict(world.occupied_cells)
        current_distance = manhattan(worker.position, world.core.position)
        outer = manhattan(world.core.position, target)
        for index, (direction, destination) in enumerate(cardinal_neighbors(worker.position)):
            if (
                destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or destination == world.core.position
                or not self._manual_allowed(worker.id, direction, world.tick)
                or projection.immediate_attackers(destination) >= worker.hp
            ):
                continue
            immediate, future, remembered = projection.worker_exposure(destination)
            viability = self._worker_move_viability(
                world,
                worker.position,
                destination,
                target=target,
                blocked=blocked,
            )
            if not viability.viable:
                continue
            home_distance = manhattan(destination, world.core.position)
            home_progress = (
                home_distance
                if current_distance > outer
                else manhattan(destination, target)
            )
            options.append(
                (
                    (
                        immediate,
                        future,
                        remembered,
                        home_progress,
                        occupied.get(destination, 0),
                        self.memory.congestion_counts.get(destination, 0),
                        index,
                    ),
                    direction,
                    destination,
                    viability,
                )
            )
        preferred_step = (
            route.first_position
            if route is not None
            and route_viability is not None
            and route_viability.viable
            else None
        )
        for score, direction, destination, viability in sorted(options)[:4]:
            if destination == preferred_step:
                continue
            intents.append(
                ActionIntent.move(
                    worker.id,
                    UnitMission.FULL_STORAGE_STAGING,
                    69,
                    direction,
                    destination,
                    risk=score[0] * 100 + score[1] * 10 + score[2],
                    exclusive_destination=True,
                    tie_break=score,
                    reason="FULL_STORAGE_STAGING_FALLBACK",
                    metadata=(
                        ("staging_post", target),
                        ("allow_protected", destination in service_cells),
                    ) + viability.metadata,
                )
            )
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.FULL_STORAGE_STAGING,
                71,
                target_position=target,
                reason=(
                    "FULL_STORAGE_STAGING_CONGESTION_HOLD"
                    if options
                    else "FULL_STORAGE_STAGING_NO_SAFE_STEP"
                ),
                metadata=(("guard_post", target),),
            )
        )
        return intents

    def _survival_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        workers: tuple[EntitySnapshot, ...],
    ) -> tuple[list[ActionIntent], set[UUID]]:
        assert world.core is not None
        intents: list[ActionIntent] = []
        escaping: set[UUID] = set()
        for worker in workers:
            survival = self._update_escape(world, projection, worker)
            if survival is not None:
                escaping.add(worker.id)
                intents.extend(
                    self._escape_intents(world, projection, worker, survival)
                )
                continue
            if (
                worker.cargo > 0
                and worker.position == world.core.position
                and world.core.state is CoreState.NORMAL
                and service.paused_reason != "CORE_STARTING_MOVE"
                and service.admission_id in {None, worker.id}
                and world.resources < world.resource_capacity
            ):
                intents.append(
                    ActionIntent.simple(
                        worker.id,
                        IntentAction.DEPOSIT,
                        UnitMission.DEPOSIT,
                        10,
                        resource_gain=min(
                            worker.cargo,
                            world.resource_capacity - world.resources,
                        ),
                        reason="CARGO_ON_STATIONARY_CORE",
                    )
                )
                continue
            if worker.cargo > 0 and worker.id in service.ready_depositors:
                # A Worker already in the explicit service pipeline follows
                # that one-slot choreography unless its current cell or next
                # slot is actually fatal.  Fog heat and broad home alerts are
                # route costs, not permission to abandon a safe queue slot.
                next_slot = dict(service.queue_slots).get(worker.id)
                current_fatal = (
                    projection.immediate_attackers(worker.position) >= worker.hp
                )
                next_fatal = bool(
                    next_slot is not None
                    and projection.immediate_attackers(next_slot) >= worker.hp
                )
                if not current_fatal and not next_fatal:
                    continue
            escape = self._update_escape(world, projection, worker)
            if escape is not None:
                escaping.add(worker.id)
                intents.extend(self._escape_intents(world, projection, worker, escape))
        return intents, escaping

    def _refresh_enemy_core_control_zones(self, world: WorldModel) -> None:
        visible_ids = {core.id for core in world.enemy_cores}
        self.memory.enemy_core_control_zones = {
            intel.id: EnemyCoreControlZone(
                core_id=intel.id,
                center=intel.position,
                exclusion_radius=self.config.enemy_core_worker_exclusion_radius,
                clear_radius=self.config.enemy_core_worker_clear_radius,
                last_seen_tick=intel.last_seen_tick,
                visible_now=intel.id in visible_ids,
                expires_tick=intel.last_seen_tick + self.config.enemy_core_control_ttl,
                control_level=(
                    EnemyCoreControlLevel.HARD
                    if world.tick - intel.last_seen_tick
                    <= self.config.enemy_core_hard_control_ticks
                    else EnemyCoreControlLevel.SOFT
                    if world.tick - intel.last_seen_tick
                    <= self.config.enemy_core_soft_control_ticks
                    else EnemyCoreControlLevel.STRATEGIC
                ),
            )
            for intel in world.remembered_enemy_cores
            if world.tick - intel.last_seen_tick <= self.config.enemy_core_control_ttl
        }
        valid_workers = {
            unit.id for unit in world.friendlies if unit.unit_type is UnitType.WORKER
        }
        for worker_id, lease in tuple(self.memory.worker_disengage_leases.items()):
            zone = self.memory.enemy_core_control_zones.get(lease.core_id)
            if (
                worker_id not in valid_workers
                or zone is None
                or zone.control_level is not EnemyCoreControlLevel.HARD
            ):
                self.memory.worker_disengage_leases.pop(worker_id, None)

    def _control_zone_cells(self) -> frozenset[Position]:
        cells: set[Position] = set()
        for zone in self.memory.enemy_core_control_zones.values():
            if zone.control_level is EnemyCoreControlLevel.HARD:
                cells.update(diamond(zone.center, zone.exclusion_radius))
        return frozenset(cells)

    def _confirmation_observer_allowed(
        self,
        world: WorldModel,
        projection: TacticalMap,
        worker: EntitySnapshot,
        zone: EnemyCoreControlZone,
    ) -> bool:
        intel = next(
            (item for item in world.remembered_enemy_cores if item.id == zone.core_id),
            None,
        )
        if (
            intel is None
            or not zone.visible_now
            or intel.sighting_count >= self.config.raid_confirmed_sightings
            or world.core is None
            or not (
                self.config.raid_start_radius
                < manhattan(zone.center, world.core.position)
                <= self.config.raid_long_range_start_radius
            )
            or worker.cargo > 0
            or worker.hp < UNIT_MAX_HP[UnitType.WORKER]
            or manhattan(worker.position, zone.center) > 3
        ):
            return False
        guards = sum(
            enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.position, zone.center) <= 8
            for enemy in world.enemies
        )
        home_threat = self.memory.home_defense_alert_until >= world.tick or any(
            enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.position, world.core.position)
            <= self.config.home_warning_radius
            for enemy in world.enemies
        )
        healthy = tuple(
            unit
            for unit in world.friendlies
            if unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and unit.hp * 2 > UNIT_MAX_HP[unit.unit_type]
            and unit.id != world.beacon.carrier_id
            and manhattan(unit.position, world.core.position)
            <= self.config.home_pursuit_radius
        )
        vanguards = sum(unit.unit_type is UnitType.VANGUARD for unit in healthy)
        rangers = sum(unit.unit_type is UnitType.RANGER for unit in healthy)
        home_target = max(
            self.config.home_force_floor,
            self.memory.home_force_high_water,
        )
        distance = manhattan(zone.center, world.core.position)
        remote = distance > self.config.raid_confirmed_start_radius
        attempt = self.memory.raid_attempts.get(zone.core_id)
        required_pairs = (
            self.config.raid_initial_pair_count
            + (0 if attempt is None else attempt.failed_attempts)
            * self.config.raid_escalation_pair_step
        )
        required = required_pairs * 2
        if remote:
            band = (
                RaidDistanceBand.LONG_RANGE
                if distance > self.config.raid_containment_radius
                else RaidDistanceBand.EXTENDED
            )
            approach = siege_approach_plan(
                world,
                zone.core_id,
                zone.center,
                band=band,
                node_limit=self.config.path_node_limit,
                max_route=self.config.raid_long_range_max_route,
            )
            force_ready = (
                len(healthy) - required >= home_target
                and vanguards >= self.config.minimum_vanguards + required_pairs
                and rangers >= self.config.minimum_rangers + required_pairs
            )
            route_ready = approach is not None
        else:
            force_ready = (
                len(healthy) - required
                >= min(home_target, self.config.raid_peace_home_reserve)
                and vanguards >= required_pairs
                and rangers >= required_pairs
            )
            route_ready = True
        if (
            guards
            or home_threat
            or not force_ready
            or not route_ready
            or projection.immediate_attackers(worker.position) > 0
        ):
            return False
        return any(
            destination not in world.known_obstacles
            and destination not in projection.hostile_occupied
            and projection.immediate_attackers(destination) < worker.hp
            and manhattan(destination, zone.center)
            > manhattan(worker.position, zone.center)
            for _, destination in cardinal_neighbors(worker.position)
        )

    def _update_core_disengage(
        self,
        world: WorldModel,
        projection: TacticalMap,
        worker: EntitySnapshot,
    ) -> WorkerDisengageLease | None:
        previous = self.memory.worker_disengage_leases.get(worker.id)
        zones = self.memory.enemy_core_control_zones
        zone = None if previous is None else zones.get(previous.core_id)
        if zone is not None and zone.control_level is not EnemyCoreControlLevel.HARD:
            zone = None
        if zone is None:
            zone = min(
                (
                    candidate
                    for candidate in zones.values()
                    if candidate.control_level is EnemyCoreControlLevel.HARD
                    and manhattan(worker.position, candidate.center)
                    <= candidate.exclusion_radius
                ),
                key=lambda item: (manhattan(worker.position, item.center), item.core_id.bytes),
                default=None,
            )
        if zone is None:
            self.memory.worker_disengage_leases.pop(worker.id, None)
            return None
        if self._confirmation_observer_allowed(
            world, projection, worker, zone
        ):
            self.memory.worker_disengage_leases.pop(worker.id, None)
            return None

        distance = manhattan(worker.position, zone.center)
        safe_now = (
            distance > zone.clear_radius
            and projection.immediate_attackers(worker.position) == 0
            and projection.future_attackers(worker.position) < worker.hp
        )
        safe_ticks = (
            (previous.safe_ticks + 1 if previous is not None else 1)
            if safe_now
            else 0
        )
        if safe_ticks >= self.config.worker_escape_safe_ticks:
            self.memory.worker_disengage_leases.pop(worker.id, None)
            return None

        mission = self.memory.unit_missions.get(worker.id)
        abandoned = None
        if mission is not None and mission.mission in {
            UnitMission.HARVEST,
            UnitMission.EXPLORE,
        }:
            abandoned = mission.target
            if mission.target is not None:
                self.memory.worker_resource_backoff[(worker.id, mission.target)] = max(
                    self.memory.worker_resource_backoff.get((worker.id, mission.target), 0),
                    zone.last_seen_tick + self.config.raid_intel_ttl,
                )
                self.memory.target_backoff_until[mission.target] = max(
                    self.memory.target_backoff_until.get(mission.target, 0),
                    zone.last_seen_tick + self.config.raid_intel_ttl,
                )
            self.memory.unit_missions.pop(worker.id, None)
            self.memory.worker_task_progress.pop(worker.id, None)
            scout = self.memory.worker_scout_states.get(worker.id)
            if scout is not None:
                self.memory.worker_scout_states[worker.id] = replace(
                    scout,
                    target=None,
                    assigned_tick=world.tick,
                    best_route_cost=None,
                    stalled_ticks=0,
                )

        waypoint = None if previous is None else previous.waypoint
        stalled = 0
        if previous is not None:
            improved = distance > previous.last_distance
            waypoint_progressed = bool(
                waypoint is not None
                and previous.last_position is not None
                and manhattan(worker.position, waypoint)
                < manhattan(previous.last_position, waypoint)
            )
            stalled = 0 if improved or waypoint_progressed else previous.stalled_ticks + 1
        if (
            waypoint is None
            or waypoint == worker.position
            or waypoint in world.known_obstacles
            or stalled >= self.config.worker_escape_replan_ticks
        ):
            waypoint = self._core_disengage_waypoint(world, projection, worker, zone)
            stalled = 0
        lease = WorkerDisengageLease(
            worker_id=worker.id,
            core_id=zone.core_id,
            center=zone.center,
            waypoint=waypoint,
            assigned_tick=(
                world.tick
                if previous is None or previous.waypoint != waypoint
                else previous.assigned_tick
            ),
            safe_ticks=safe_ticks,
            last_distance=distance,
            last_position=worker.position,
            stalled_ticks=stalled,
            abandoned_target=abandoned or (None if previous is None else previous.abandoned_target),
        )
        self.memory.worker_disengage_leases[worker.id] = lease
        return lease

    def _core_disengage_waypoint(
        self,
        world: WorldModel,
        projection: TacticalMap,
        worker: EntitySnapshot,
        zone: EnemyCoreControlZone,
    ) -> Position | None:
        blocked = projection.hostile_occupied | frozenset(world.known_obstacles)
        candidates = tuple(
            cell
            for radius in (zone.clear_radius + 1, zone.clear_radius + 2)
            for cell in manhattan_ring(zone.center, radius)
            if cell in world.known_passable
            and cell not in blocked
            and projection.immediate_attackers(cell) < worker.hp
        )
        rows = []
        for cell in candidates:
            route = route_to(
                world,
                worker.position,
                cell,
                node_limit=min(self.config.path_node_limit, 768),
                blocked=blocked - {worker.position, cell},
            )
            if route is None:
                continue
            rows.append(
                (
                    projection.future_attackers(cell),
                    projection.worker_exposure(cell)[2],
                    route.distance,
                    -manhattan(cell, zone.center),
                    cell,
                )
            )
        return None if not rows else min(rows)[-1]

    def _core_disengage_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        worker: EntitySnapshot,
        lease: WorkerDisengageLease,
    ) -> list[ActionIntent]:
        rows = []
        occupied = dict(world.occupied_cells)
        for index, (direction, destination) in enumerate(cardinal_neighbors(worker.position)):
            if (
                destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or occupied.get(destination, 0) >= 2
            ):
                continue
            immediate = projection.immediate_attackers(destination)
            future = projection.future_attackers(destination)
            if immediate >= worker.hp:
                continue
            viability = move_viability(
                world,
                worker.position,
                destination,
                blocked=projection.hostile_occupied,
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            )
            if not viability.viable:
                continue
            distance = manhattan(destination, lease.center)
            rows.append(
                (
                    (
                        immediate,
                        future,
                        -distance,
                        -viability.forward_exits,
                        projection.worker_exposure(destination)[2],
                        int(
                            destination[0] == lease.center[0]
                            or destination[1] == lease.center[1]
                        ),
                        0 if lease.waypoint is None else manhattan(destination, lease.waypoint),
                        index,
                    ),
                    direction,
                    destination,
                    viability,
                )
            )
        zero_exposure = [row for row in rows if row[0][0] == 0 and row[0][1] == 0]
        if zero_exposure:
            rows = zero_exposure
        outward = [row for row in rows if manhattan(row[2], lease.center) > lease.last_distance]
        if outward:
            rows = outward
        intents = [
            ActionIntent.move(
                worker.id,
                UnitMission.CORE_DISENGAGE,
                19,
                direction,
                destination,
                risk=score[0] * 100 + score[1] * 10 + score[4],
                tie_break=score,
                reason="ENEMY_CORE_CONTROL_DISENGAGE",
                metadata=(
                    ("enemy_core_id", str(lease.core_id)),
                    ("enemy_core_center", lease.center),
                    ("control_distance_before", lease.last_distance),
                    ("control_distance_after", manhattan(destination, lease.center)),
                    ("clear_radius", self.config.enemy_core_worker_clear_radius),
                    ("disengage_waypoint", lease.waypoint),
                    ("abandoned_target", lease.abandoned_target),
                ) + viability.metadata,
            )
            for score, direction, destination, viability in sorted(rows)
        ]
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.CORE_DISENGAGE,
                20,
                target_id=lease.core_id,
                target_position=lease.waypoint,
                reason=(
                    "CORE_DISENGAGE_BLOCKED_THIS_TICK"
                    if rows
                    else "CORE_DISENGAGE_NO_SURVIVABLE_ROUTE"
                ),
            )
        )
        return intents

    def _assign_work(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        available: tuple[EntitySnapshot, ...],
    ) -> tuple[
        dict[UUID, Position],
        dict[UUID, tuple[Position, Route, int]],
    ]:
        assert world.core is not None
        for key, expires_tick in tuple(self.memory.worker_resource_backoff.items()):
            if expires_tick < world.tick:
                self.memory.worker_resource_backoff.pop(key, None)
        resource_workers = tuple(
            worker
            for worker in available
            if worker.position != world.core.position
        )
        service_blocks = frozenset(
            cell
            for cell in (
                world.core.position,
                service.service_core_position,
                service.entrance,
                service.exit_cell,
                *service.queue_cells,
                *dict(service.overflow_slots).values(),
            )
            if cell is not None
        )
        allocation = self.resources.allocate(
            world,
            projection,
            resource_workers,
            hard_blocked=service_blocks,
        )
        self.memory.resource_candidate_counts = dict(allocation.candidate_counts)
        self.memory.resource_rejection_counts = dict(allocation.rejection_counts)
        resources = allocation.as_dict()
        available_by_id = {worker.id: worker for worker in resource_workers}
        for worker_id, order in tuple(self.memory.resource_work_orders.items()):
            if worker_id not in available_by_id:
                self.memory.resource_work_orders.pop(worker_id, None)
                continue
            if (
                worker_id not in resources
                and order.target in self.memory.resource_memory
                and self.memory.worker_resource_backoff.get(
                    (worker_id, order.target), -1
                ) < world.tick
                and order.stalled_ticks < 2
            ):
                resources[worker_id] = order.target
        for worker_id, target in resources.items():
            existing = self.memory.resource_work_orders.get(worker_id)
            confirmed_tick = self.memory.resource_memory.get(
                target,
                existing.last_confirmed_tick
                if existing is not None and existing.target == target
                else world.tick,
            )
            self.memory.resource_work_orders[worker_id] = ResourceWorkOrder(
                worker_id=worker_id,
                target=target,
                assigned_tick=(
                    existing.assigned_tick
                    if existing is not None and existing.target == target
                    else world.tick
                ),
                last_confirmed_tick=confirmed_tick,
                last_route_distance=(
                    existing.last_route_distance
                    if existing is not None and existing.target == target
                    else None
                ),
                stalled_ticks=(
                    existing.stalled_ticks
                    if existing is not None and existing.target == target
                    else 0
                ),
                failures=(
                    existing.failures
                    if existing is not None and existing.target == target
                    else 0
                ),
            )
        explorers = tuple(
            worker for worker in available if worker.id not in resources
        )
        exploration = self._resource_search_assignments(
            world,
            projection,
            explorers,
            service,
        )
        return resources, exploration

    def _routine_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        worker: EntitySnapshot,
        resource: Position | None,
        exploration: tuple[Position, Route, int] | None,
        *,
        allow_harvest: bool = True,
    ) -> list[ActionIntent]:
        assert world.core is not None
        if worker.cargo == 0 and worker.position == world.core.position:
            self.memory.service_egress_worker_ids.add(worker.id)
            return self._clear_core(
                world,
                projection,
                worker,
                service,
                exploration,
            )
        if worker.id in {row[0] for row in service.blocking_units}:
            return self._clear_service_cell(
                world,
                projection,
                worker,
                service,
            )
        if worker.id in self.memory.service_egress_worker_ids and worker.position in {
            *(cell for cell in (service.entrance, service.exit_cell) if cell is not None),
            *service.queue_cells,
        }:
            return self._clear_service_lane(
                world,
                projection,
                worker,
                service,
            )
        if worker.cargo > 0:
            return self._cargo(world, projection, worker, service)
        if allow_harvest and worker.position in world.visible_resources:
            self.memory.unit_missions[worker.id] = MissionState(
                UnitMission.HARVEST,
                worker.position,
                world.tick,
            )
            return [
                ActionIntent.simple(
                    worker.id,
                    IntentAction.HARVEST,
                    UnitMission.HARVEST,
                    50,
                    reason="RESOURCE_UNDERFOOT",
                )
            ]
        assigned = self._resource_work_intents(
            world,
            projection,
            service,
            worker,
            resource,
        )
        if assigned:
            return assigned
        assigned = self._exploration_work_intents(
            world,
            projection,
            service,
            worker,
            exploration,
        )
        if assigned:
            return assigned
        return [
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.EXPLORE,
                71,
                reason=self._exploration_wait_reason(
                    world,
                    projection,
                    worker,
                    service,
                ),
            )
        ]

    def _resource_work_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        worker: EntitySnapshot,
        resource: Position | None,
    ) -> list[ActionIntent]:
        if resource is None:
            return []
        route = self._route(world, projection, worker, resource, service)
        previous_progress = self.memory.worker_task_progress.get(worker.id)
        route_distance = None if route is None else route.distance
        same_target = bool(
            previous_progress is not None
            and previous_progress.target == resource
        )
        improved = bool(
            route_distance is not None
            and (
                not same_target
                or previous_progress.route_distance is None
                or route_distance < previous_progress.route_distance
            )
        )
        stalled = (
            0
            if improved
            else (
                previous_progress.stalled_ticks + 1
                if same_target
                else 1
            )
        )
        progress = WorkerTaskProgress(
            worker_id=worker.id,
            target=resource,
            route_distance=route_distance,
            last_progress_tick=(
                world.tick
                if improved or previous_progress is None
                else previous_progress.last_progress_tick
            ),
            stalled_ticks=stalled,
            rejection_reason=None if route is not None else "NO_SAFE_RESOURCE_ROUTE",
        )
        if stalled >= 2:
            backoff_until = world.tick + 8
            progress = replace(
                progress,
                rejection_reason="RESOURCE_TASK_STALLED",
                backoff_until=backoff_until,
            )
            self.memory.worker_resource_backoff[(worker.id, resource)] = backoff_until
            self.memory.resource_work_orders.pop(worker.id, None)
            self.memory.unit_missions.pop(worker.id, None)
            self.memory.worker_task_progress[worker.id] = progress
            return self._resource_stall_fallback(
                world,
                projection,
                service,
                worker,
                resource,
            )
        existing_order = self.memory.resource_work_orders.get(worker.id)
        self.memory.resource_work_orders[worker.id] = ResourceWorkOrder(
            worker_id=worker.id,
            target=resource,
            assigned_tick=(
                existing_order.assigned_tick
                if existing_order is not None and existing_order.target == resource
                else world.tick
            ),
            last_confirmed_tick=self.memory.resource_memory.get(
                resource,
                existing_order.last_confirmed_tick
                if existing_order is not None and existing_order.target == resource
                else world.tick,
            ),
            last_route_distance=route_distance,
            stalled_ticks=stalled,
            failures=(
                existing_order.failures
                if existing_order is not None and existing_order.target == resource
                else 0
            ),
        )
        self.memory.worker_task_progress[worker.id] = progress
        self.memory.unit_missions[worker.id] = MissionState(
            UnitMission.HARVEST,
            resource,
            self.memory.resource_work_orders[worker.id].assigned_tick,
        )
        if route is None or route.first_direction is None:
            return self._resource_stall_fallback(
                world,
                projection,
                service,
                worker,
                resource,
            )
        if not self._manual_allowed(worker.id, route.first_direction, world.tick):
            return [
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.HARVEST,
                    51,
                    target_position=resource,
                    reason="RESOURCE_MANUAL_LEASE_HOLD",
                )
            ]
        blocked, _ = self._exploration_navigation(
            world,
            projection,
            worker,
            service,
        )
        route_viability = move_viability(
            world,
            worker.position,
            route.first_position,
            target=resource,
            blocked=blocked,
            node_limit=min(self.config.path_node_limit, 512),
            require_continuation=route.first_position != resource,
            terminal_exception=(
                "RESOURCE" if route.first_position == resource else None
            ),
        )
        if not route_viability.viable:
            return self._resource_stall_fallback(
                world,
                projection,
                service,
                worker,
                resource,
            )
        intents = [
            ActionIntent.move(
                worker.id,
                UnitMission.HARVEST,
                50,
                route.first_direction,
                route.first_position,
                risk=self._risk(projection, route.first_position),
                tie_break=(route.distance,),
                reason="GLOBAL_RESOURCE_MATCH",
                metadata=route_viability.metadata,
            ),
        ]
        alternatives = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(worker.position)
        ):
            if (
                direction is route.first_direction
                or destination in blocked
                or destination in world.known_obstacles
                or destination not in world.known_passable
                or projection.immediate_attackers(destination) >= worker.hp
                or not self._manual_allowed(worker.id, direction, world.tick)
            ):
                continue
            viability = move_viability(
                world,
                worker.position,
                destination,
                target=resource,
                blocked=blocked,
                node_limit=min(self.config.path_node_limit, 512),
                require_continuation=destination != resource,
                terminal_exception=(
                    "RESOURCE" if destination == resource else None
                ),
            )
            if not viability.viable:
                continue
            score = (
                projection.immediate_attackers(destination),
                projection.future_attackers(destination),
                projection.worker_exposure(destination)[2],
                manhattan(destination, resource),
                self.memory.congestion_counts.get(destination, 0),
                index,
            )
            alternatives.append((score, direction, destination, viability))
        intents.extend(
            ActionIntent.move(
                worker.id,
                UnitMission.HARVEST,
                51,
                direction,
                destination,
                risk=score[0] * 100 + score[1] * 10 + score[2],
                tie_break=score,
                reason="RESOURCE_SAFE_DETOUR",
                metadata=(("goal", resource),) + viability.metadata,
            )
            for score, direction, destination, viability in sorted(
                alternatives, key=lambda row: row[0]
            )[:3]
        )
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.HARVEST,
                54,
                target_position=resource,
                reason="RESOURCE_ROUTE_BLOCKED_THIS_TICK",
            )
        )
        return intents

    def _resource_stall_fallback(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        worker: EntitySnapshot,
        old_target: Position,
    ) -> list[ActionIntent]:
        blocked, _ = self._exploration_navigation(
            world,
            projection,
            worker,
            service,
        )
        rows = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(worker.position)
        ):
            if (
                destination in blocked
                or destination not in world.known_passable
                or destination in world.known_obstacles
                or projection.immediate_attackers(destination) >= worker.hp
                or not self._manual_allowed(worker.id, direction, world.tick)
            ):
                continue
            viability = move_viability(
                world,
                worker.position,
                destination,
                blocked=blocked,
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            )
            if not viability.viable:
                continue
            gain = information_gain(
                destination,
                tick=world.tick,
                last_visible=dict(world.cell_last_visible),
            )
            score = (
                projection.future_attackers(destination),
                projection.worker_exposure(destination)[2],
                -gain,
                self.memory.congestion_counts.get(destination, 0),
                index,
            )
            rows.append((score, direction, destination, gain, viability))
        mission = UnitMission.RESOURCE_SEARCH
        intents = [
            ActionIntent.move(
                worker.id,
                mission,
                69,
                direction,
                destination,
                risk=score[0] * 10 + score[1],
                tie_break=score,
                reason="RESOURCE_STALL_SCOUT_FALLBACK",
                metadata=(
                    ("released_target", old_target),
                    ("information_gain", gain),
                )
                + viability.metadata,
            )
            for score, direction, destination, gain, viability in sorted(
                rows, key=lambda row: row[0]
            )[:3]
        ]
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.RESOURCE_SEARCH,
                73,
                target_position=old_target,
                reason=(
                    "ALL_SCOUT_TARGETS_BLOCKED"
                    if rows
                    else "NO_SURVIVABLE_MOVE"
                ),
            )
        )
        return intents

    def _exploration_work_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
        worker: EntitySnapshot,
        exploration: tuple[Position, Route, int] | None,
    ) -> list[ActionIntent]:
        if exploration is None:
            return []
        target, route, gain = exploration
        if route.first_direction is None or not self._manual_allowed(
            worker.id,
            route.first_direction,
            world.tick,
        ):
            return []
        scout_state = self.memory.worker_scout_states.get(worker.id)
        search_lease = self.memory.resource_search_leases.get(worker.id)
        mission = (
            UnitMission.RESOURCE_SEARCH
            if search_lease is not None and search_lease.target == target
            else (
                UnitMission.RETURN_TO_SCOUT_BAND
                if scout_state is not None
                and scout_state.phase is WorkerScoutPhase.RETURN_TO_BAND
                else UnitMission.EXPLORE
            )
        )
        blocked, costs = self._exploration_navigation(
            world,
            projection,
            worker,
            service,
        )
        viability = route.viability or self._worker_move_viability(
            world,
            worker.position,
            route.first_position,
            target=target,
            blocked=blocked,
        )
        if not viability.viable:
            if mission is UnitMission.RESOURCE_SEARCH:
                self._invalidate_resource_search_target(
                    worker.id,
                    world.tick,
                    target,
                    edge=(worker.position, route.first_position),
                )
            return []
        route = Route(
            route.distance,
            route.first_direction,
            route.first_position,
            viability,
        )
        intents = [
            ActionIntent.move(
                worker.id,
                mission,
                70,
                route.first_direction,
                route.first_position,
                risk=self._risk(projection, route.first_position),
                tie_break=(-gain, route.distance),
                reason=(
                    "RESOURCE_FRONTIER_SEARCH"
                    if mission is UnitMission.RESOURCE_SEARCH
                    else (
                        "RETURN_TO_SCOUT_BAND"
                        if mission is UnitMission.RETURN_TO_SCOUT_BAND
                        else "INFORMATION_GAIN"
                    )
                ),
                metadata=(("information_gain", gain), ("goal", target))
                + (() if route.viability is None else route.viability.metadata),
            ),
        ]
        previous = None
        history = self.memory.position_history.get(worker.id, ())
        if len(history) >= 2:
            previous = history[-2]
        alternatives = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(worker.position)
        ):
            if (
                direction is route.first_direction
                or destination in blocked
                or destination in world.known_obstacles
                or destination not in world.known_passable
                or not self._manual_allowed(worker.id, direction, world.tick)
                or projection.immediate_attackers(destination) >= worker.hp
            ):
                continue
            return_lease = self.memory.scout_return_route_leases.get(worker.id)
            if (
                mission is UnitMission.RETURN_TO_SCOUT_BAND
                and return_lease is not None
                and return_lease.blocked_edge == (worker.position, destination)
                and world.tick <= return_lease.backoff_until
            ):
                continue
            remaining_route = None
            if mission is UnitMission.RETURN_TO_SCOUT_BAND:
                remaining_route = weighted_route_to(
                    world,
                    destination,
                    target,
                    node_limit=self.config.path_node_limit,
                    blocked=frozenset(set(blocked) - {destination, target}),
                    cell_costs=costs,
                )
                if remaining_route is None:
                    continue
            viability = self._worker_move_viability(
                world,
                worker.position,
                destination,
                target=target,
                blocked=blocked,
                node_limit=(
                    self.config.path_node_limit
                    if mission in {
                        UnitMission.RETURN_TO_SCOUT_BAND,
                        UnitMission.RESOURCE_SEARCH,
                    }
                    else min(self.config.path_node_limit, 512)
                ),
            )
            if not viability.viable:
                continue
            immediate, future, remembered = projection.worker_exposure(destination)
            score = (
                immediate,
                future,
                remembered,
                (
                    remaining_route.distance
                    if remaining_route is not None
                    else manhattan(destination, target)
                ),
                int(destination == previous),
                self.memory.congestion_counts.get(destination, 0),
                index,
            )
            alternatives.append((score, direction, destination, viability))
        intents.extend(
            ActionIntent.move(
                worker.id,
                mission,
                71,
                direction,
                destination,
                risk=score[0] * 100 + score[1] * 10 + score[2],
                tie_break=score,
                reason=(
                    "RESOURCE_FRONTIER_ALTERNATE"
                    if mission is UnitMission.RESOURCE_SEARCH
                    else (
                        "RETURN_TO_SCOUT_BAND_ALTERNATE"
                        if mission is UnitMission.RETURN_TO_SCOUT_BAND
                        else "EXPLORATION_ALTERNATE_STEP"
                    )
                ),
                metadata=(("information_gain", gain), ("goal", target))
                + viability.metadata,
            )
            for score, direction, destination, viability in sorted(
                alternatives, key=lambda row: row[0]
            )[:3]
        )
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                mission,
                72,
                target_position=target,
                reason="EXPLORATION_MOVE_BLOCKED_THIS_TICK",
            )
        )
        return intents

    def _update_escape(self, world, projection, worker):
        wounded = worker.hp < UNIT_MAX_HP[UnitType.WORKER]
        relevant = tuple(
            enemy
            for enemy in projection.enemies
            if enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and enemy.age <= self.config.enemy_track_ttl
        )
        direct_threats = tuple(
            enemy.enemy_id
            for enemy in relevant
            if enemy.visible_now
            and (
                manhattan(enemy.observed_position, worker.position)
                <= self.config.global_worker_threat_awareness_radius
                or worker.position in enemy.immediate_attack_cells
            )
        )
        shared_threats = tuple(
            enemy.enemy_id
            for enemy in relevant
            if (
                max(
                    0,
                    manhattan(enemy.observed_position, worker.position) - enemy.age,
                )
                <= self.config.global_worker_threat_awareness_radius
                or worker.position in enemy.immediate_attack_cells
                or worker.position in enemy.future_attack_cells
                or worker.position in enemy.movement_corridor
            )
        )
        if wounded:
            # A wounded Worker's sticky RECOVER job already uses the shared
            # danger heat-map.  A remote/fog-wide alert must not send it away
            # from treatment; only a genuinely local visible threat may
            # temporarily pre-empt the return route.
            shared_threats = direct_threats
        threats = tuple(
            sorted(set(direct_threats) | set(shared_threats), key=lambda item: item.bytes)
        )
        previous = self.memory.worker_escape_states.get(worker.id)

        current_zones = tuple(
            sorted(
                (
                    zone
                    for zone in self.memory.enemy_core_control_zones.values()
                    if zone.control_level is EnemyCoreControlLevel.HARD
                    and manhattan(worker.position, zone.center) <= zone.clear_radius
                ),
                key=lambda zone: (
                    manhattan(worker.position, zone.center),
                    zone.core_id.bytes,
                ),
            )
        )
        confirmation_zone_ids = frozenset(
            zone.core_id
            for zone in current_zones
            if self._confirmation_observer_allowed(
                world,
                projection,
                worker,
                zone,
            )
        )
        current_zones = tuple(
            zone
            for zone in current_zones
            if zone.core_id not in confirmation_zone_ids
        )
        retained_zone_ids = {
            zone.core_id for zone in current_zones
        } | (
            set()
            if previous is None
            else set(previous.control_core_ids) - set(confirmation_zone_ids)
        )
        retained_zones = tuple(
            sorted(
                (
                    zone
                    for core_id in retained_zone_ids
                    if (zone := self.memory.enemy_core_control_zones.get(core_id))
                    is not None
                    and zone.control_level is EnemyCoreControlLevel.HARD
                    and (
                        zone.expires_tick is None
                        or zone.expires_tick >= world.tick
                    )
                ),
                key=lambda zone: zone.core_id.bytes,
            )
        )
        control_ids = tuple(zone.core_id for zone in retained_zones)
        control_centers = tuple(zone.center for zone in retained_zones)

        # A remembered Core stops being an operational escape constraint once
        # its hard 16-Tick control window expires.  Strategic intelligence is
        # retained for recon/raid planning, but must not keep an otherwise
        # safe Worker in CLEARING for hundreds of Ticks.
        if (
            previous is not None
            and previous.control_core_ids
            and not control_ids
            and not threats
            and not previous.threat_ids
        ):
            self.memory.worker_escape_states.pop(worker.id, None)
            return None

        if current_zones:
            mission = self.memory.unit_missions.get(worker.id)
            if mission is not None and mission.mission in {
                UnitMission.HARVEST,
                UnitMission.EXPLORE,
            }:
                if mission.target is not None:
                    expiry = max(
                        zone.last_seen_tick + self.config.enemy_core_soft_control_ticks
                        for zone in current_zones
                    )
                    self.memory.worker_resource_backoff[(worker.id, mission.target)] = max(
                        self.memory.worker_resource_backoff.get(
                            (worker.id, mission.target),
                            0,
                        ),
                        expiry,
                    )
                    self.memory.target_backoff_until[mission.target] = max(
                        self.memory.target_backoff_until.get(mission.target, 0),
                        expiry,
                    )
                self.memory.unit_missions.pop(worker.id, None)
                self.memory.worker_task_progress.pop(worker.id, None)
                scout = self.memory.worker_scout_states.get(worker.id)
                if scout is not None:
                    self.memory.worker_scout_states[worker.id] = replace(
                        scout,
                        target=None,
                        assigned_tick=world.tick,
                        best_route_cost=None,
                        stalled_ticks=0,
                    )

        def progressed_state(
            phase: str,
            threat_ids: tuple[UUID, ...],
            last_threat_tick: int,
            safe_ticks: int,
        ) -> WorkerSurvivalLease:
            minimum, _ = self._survival_distances(
                projection,
                worker.position,
                threat_ids,
                control_centers,
            )
            same_threats = (
                previous is not None
                and previous.threat_ids == threat_ids
                and previous.control_core_ids == control_ids
            )
            improved = (
                previous is None
                or previous.last_min_enemy_distance is None
                or minimum > previous.last_min_enemy_distance
            )
            waypoint_progressed = bool(
                previous is not None
                and previous.waypoint is not None
                and previous.last_waypoint_distance is not None
                and manhattan(worker.position, previous.waypoint)
                < previous.last_waypoint_distance
            )
            stalled = (
                previous.stalled_ticks + 1
                if same_threats and not improved and not waypoint_progressed
                else 0
            )
            history = self.memory.position_history.get(worker.id, ())
            loop_period = self._escape_loop_period(history)
            waypoint = None if previous is None else previous.waypoint
            route_version = 0 if previous is None else previous.route_version
            waypoint_assigned_tick = (
                None if previous is None else previous.waypoint_assigned_tick
            )
            waypoint_expires_tick = (
                None if previous is None else previous.waypoint_expires_tick
            )
            invalid_reason = None
            lease_mature = bool(
                waypoint_assigned_tick is None
                or world.tick - waypoint_assigned_tick
                >= self.config.worker_escape_waypoint_lease_ticks
            )
            if (
                waypoint == worker.position
                or not same_threats
                or lease_mature
                and stalled >= self.config.worker_escape_replan_ticks
            ):
                invalid_reason = (
                    "WAYPOINT_REACHED"
                    if waypoint == worker.position
                    else "THREAT_SET_CHANGED"
                    if not same_threats
                    else "WAYPOINT_ROUTE_STALLED"
                )
                waypoint = None
                waypoint_assigned_tick = None
                waypoint_expires_tick = None
            return WorkerSurvivalLease(
                phase=phase,
                threat_ids=threat_ids,
                last_threat_tick=last_threat_tick,
                safe_ticks=safe_ticks,
                waypoint=waypoint,
                last_min_enemy_distance=minimum,
                stalled_ticks=stalled,
                loop_period=loop_period,
                route_version=route_version,
                waypoint_assigned_tick=waypoint_assigned_tick,
                waypoint_expires_tick=waypoint_expires_tick,
                waypoint_invalid_reason=invalid_reason,
                last_waypoint_distance=(
                    None
                    if waypoint is None
                    else manhattan(worker.position, waypoint)
                ),
                control_core_ids=control_ids,
                control_centers=control_centers,
            )

        if threats or current_zones:
            retained = {
                threat_id
                for threat_id in (() if previous is None else previous.threat_ids)
                if (enemy := projection.enemy(threat_id)) is not None
                and enemy.age <= self.config.enemy_track_ttl
            }
            retained.update(threats)
            phase = (
                "FLEEING"
                if direct_threats
                else "CORE_DISENGAGE"
                if current_zones
                else "FOG_RETREAT"
            )
            state = progressed_state(
                phase,
                tuple(sorted(retained, key=lambda item: item.bytes)),
                max(
                    (
                        enemy.last_seen_tick
                        for threat_id in retained
                        if (enemy := projection.enemy(threat_id)) is not None
                    ),
                    default=(
                        previous.last_threat_tick
                        if previous is not None
                        else world.tick
                    ),
                ),
                0,
            )
            self.memory.worker_escape_states[worker.id] = state
            if worker.hp >= UNIT_MAX_HP[UnitType.WORKER]:
                self.memory.unit_missions.pop(worker.id, None)
            return state
        if previous is None:
            return None
        fresh = tuple(
            threat_id
            for threat_id in previous.threat_ids
            if (enemy := projection.enemy(threat_id)) is not None
            and enemy.age <= self.config.enemy_track_ttl
        )
        if fresh and wounded:
            fresh = tuple(
                threat_id
                for threat_id in fresh
                if (enemy := projection.enemy(threat_id)) is not None
                and enemy.age <= 2
                and max(
                    0,
                    manhattan(enemy.observed_position, worker.position) - enemy.age,
                ) <= self.config.worker_escape_clearance_radius
            )
            if not fresh:
                self.memory.worker_escape_states.pop(worker.id, None)
                return None
        if fresh:
            state = progressed_state(
                "FOG_RETREAT",
                fresh,
                previous.last_threat_tick,
                0,
            )
            self.memory.worker_escape_states[worker.id] = state
            return state
        exits = self._safe_exits(
            world,
            projection,
            worker.position,
            worker.hp,
            threat_ids=previous.threat_ids,
        )
        outside_controls = all(
            manhattan(worker.position, center)
            > self.config.enemy_core_worker_clear_radius
            for center in previous.control_centers
        )
        safe_ticks = previous.safe_ticks + 1 if exits >= 2 and outside_controls else 0
        if safe_ticks >= self.config.worker_escape_safe_ticks:
            self.memory.worker_escape_states.pop(worker.id, None)
            return None
        state = progressed_state(
            "CLEARING",
            previous.threat_ids,
            previous.last_threat_tick,
            safe_ticks,
        )
        self.memory.worker_escape_states[worker.id] = state
        return state

    def _escape_intents(self, world, projection, worker, state):
        assert world.core is not None
        rows: list[_EscapeCandidate] = []
        history = self.memory.position_history.get(worker.id, ())
        previous = (
            history[-2]
            if len(history) >= 2
            else None
        )
        recent_positions = frozenset(history[-4:-1])
        current_visible_minimum, _ = self._visible_enemy_distances(
            projection,
            worker.position,
            state.threat_ids,
        )
        current_visible_vanguard_minimum = self._visible_vanguard_distance(
            projection,
            worker.position,
            state.threat_ids,
        )
        for index, (direction, destination) in enumerate(cardinal_neighbors(worker.position)):
            if destination in world.known_obstacles or destination in projection.hostile_occupied:
                continue
            # Only a currently visible firing position is authoritative for
            # this Tick.  Fog envelopes remain future risk; treating every
            # uncertain old position as a current attacker can freeze a
            # Worker even when one observable safe retreat exists.
            immediate = projection.immediate_attackers(destination)
            future = max(
                projection.future_attackers(destination),
                self.safety.projected_attackers(
                    world,
                    projection,
                    destination,
                    depth=1,
                    threat_ids=state.threat_ids,
                    depth_limit=self.config.worker_escape_plan_depth,
                ),
            )
            allowed = (
                self.config.worker_escape_nonfatal_hit_budget
                if worker.hp == UNIT_MAX_HP[UnitType.WORKER]
                else 0
            )
            if immediate >= worker.hp or immediate > allowed:
                continue
            exits = self._safe_exits(
                world,
                projection,
                destination,
                worker.hp,
                origin=worker.position,
                threat_ids=state.threat_ids,
            )
            horizon = self._survival_horizon(
                world,
                projection,
                destination,
                worker.hp,
                origin=worker.position,
                threat_ids=state.threat_ids,
            )
            retreat_target = world.core.destination or world.core.position
            viability = move_viability(
                world,
                worker.position,
                destination,
                target=retreat_target,
                blocked=projection.hostile_occupied,
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
                terminal_exception=(
                    "CORE_SERVICE" if destination == retreat_target else None
                ),
            )
            # A full-health Worker may spend its single non-fatal hit budget to
            # cross an exposed cell, but never to enter a cell from which every
            # continuation is fatal.  Treating a full-health unit as an
            # exception here was what allowed the 101946 death pocket.
            if not viability.viable or horizon == 0:
                continue
            recent = self._recent_threat(projection, destination, state.threat_ids)
            heat = projection.worker_exposure(destination)[2]
            minimum, total = self._survival_distances(
                projection,
                destination,
                state.threat_ids,
                state.control_centers,
            )
            visible_minimum, visible_total = self._visible_enemy_distances(
                projection,
                destination,
                state.threat_ids,
            )
            visible_vanguard_minimum = self._visible_vanguard_distance(
                projection,
                destination,
                state.threat_ids,
            )
            waypoint, waypoint_minimum, waypoint_total, waypoint_heat = (
                self._escape_outlook(
                    world,
                    projection,
                    destination,
                    worker.position,
                    worker.hp,
                    state.threat_ids,
                    state.control_centers,
                )
            )
            rows.append(
                _EscapeCandidate(
                    direction=direction,
                    destination=destination,
                    viability=viability,
                    immediate_attackers=immediate,
                    future_attackers=future,
                    recent_threat=recent,
                    heat=heat,
                    minimum_enemy_distance=minimum,
                    total_enemy_distance=total,
                    visible_minimum_distance=visible_minimum,
                    visible_total_distance=visible_total,
                    visible_vanguard_minimum_distance=visible_vanguard_minimum,
                    survival_terminals=horizon,
                    forward_exits=exits,
                    waypoint=waypoint,
                    waypoint_minimum_distance=waypoint_minimum,
                    waypoint_total_distance=waypoint_total,
                    waypoint_heat=waypoint_heat,
                    direction_index=index,
                    recently_visited=destination in recent_positions,
                    backtracking=destination == previous,
                    lease_distance=(
                        0
                        if state.waypoint is None
                        else manhattan(destination, state.waypoint)
                    ),
                )
            )

        filter_rejections: dict[str, int] = {}

        def retain(candidates, predicate, reason):
            kept = [candidate for candidate in candidates if predicate(candidate)]
            if kept:
                rejected = len(candidates) - len(kept)
                if rejected:
                    filter_rejections[reason] = (
                        filter_rejections.get(reason, 0) + rejected
                    )
                return kept
            return candidates

        # The non-fatal hit budget is a last resort, not a licence to ignore a
        # zero-exposure exit.  Apply the safety frontier before distance,
        # novelty or deterministic direction tie-breaks.
        rows = retain(
            rows,
            lambda row: row.immediate_attackers == 0,
            "NONZERO_CURRENT_EXPOSURE",
        )
        rows = retain(
            rows,
            lambda row: row.future_attackers == 0,
            "NONZERO_FUTURE_EXPOSURE",
        )

        if state.phase != "FLEEING" and state.last_min_enemy_distance is not None:
            rows = retain(
                rows,
                lambda row: (
                    row.minimum_enemy_distance >= state.last_min_enemy_distance
                ),
                "DECREASES_CONSERVATIVE_SURVIVAL_DISTANCE",
            )

        if state.control_centers:
            currently_clear = all(
                manhattan(worker.position, center)
                > self.config.enemy_core_worker_clear_radius
                for center in state.control_centers
            )
            if currently_clear:
                rows = retain(
                    rows,
                    lambda row: all(
                        manhattan(row.destination, center)
                        > self.config.enemy_core_worker_clear_radius
                        for center in state.control_centers
                    ),
                    "REENTERS_ENEMY_CORE_CONTROL_ZONE",
                )

        # Recent history may break a tie, but it may never force a Worker
        # closer to an authoritative visible Vanguard threat.  When a
        # non-approaching option exists on the same safety frontier, discard
        # all approaching candidates.
        if current_visible_vanguard_minimum < 99:
            rows = retain(
                rows,
                lambda row: (
                    row.visible_vanguard_minimum_distance
                    >= current_visible_vanguard_minimum
                ),
                "APPROACHES_VISIBLE_VANGUARD",
            )

        core_target = world.core.destination or world.core.position
        current_home_distance = manhattan(worker.position, core_target)
        core_retreat_first_step = None
        core_retreat_route_mode = None
        if state.phase != "FLEEING" and rows:
            # Core remains the strategic retreat target after contact is lost.
            # Threat evidence changes the *route* home; it must not turn into a
            # blanket instruction to keep walking away from friendly cover.
            # Current/future attack cells and remembered corridors are priced
            # into one shared route, while genuinely lethal cells are blocked.
            control_blocked, route_costs = self.safety.navigation_layers(
                projection,
                tuple(
                    zone
                    for core_id in state.control_core_ids
                    if (zone := self.memory.enemy_core_control_zones.get(core_id))
                    is not None
                ),
            )
            lethal_cells = set(projection.hostile_occupied)
            lethal_cells.update(control_blocked)
            for cell, threat in projection.threat_cells.items():
                route_costs[cell] = (
                    route_costs.get(cell, 0)
                    + threat.immediate_attackers * 100
                    + threat.future_attackers * 20
                )
                if threat.immediate_attackers >= worker.hp:
                    lethal_cells.add(cell)
            lethal_cells.discard(worker.position)
            lethal_cells.discard(core_target)
            core_route = weighted_route_to(
                world,
                worker.position,
                core_target,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(lethal_cells),
                cell_costs=route_costs,
            )
            if core_route is not None and core_route.first_position is not None:
                core_retreat_first_step = core_route.first_position
                core_retreat_route_mode = "FULL"
            else:
                segment = weighted_progress_route(
                    world,
                    worker.position,
                    core_target,
                    node_limit=self.config.path_node_limit,
                    blocked=frozenset(lethal_cells),
                    cell_costs=route_costs,
                )
                if segment is not None:
                    core_retreat_first_step = segment[0].first_position
                    core_retreat_route_mode = "SEGMENT"

            routed = [
                row
                for row in rows
                if row.destination == core_retreat_first_step
            ]
            if routed:
                filter_rejections["NOT_THREAT_AWARE_CORE_ROUTE"] = (
                    filter_rejections.get("NOT_THREAT_AWARE_CORE_ROUTE", 0)
                    + len(rows)
                    - len(routed)
                )
                rows = routed
            else:
                # If the complete route's first step was removed by the local
                # survival frontier, prefer a safe direct improvement.  When an
                # enemy actually blocks that direction this set is empty and
                # the normal escape score chooses a lateral detour; it does not
                # mechanically force further outward movement.
                homeward_survivable = [
                    row
                    for row in rows
                    if manhattan(row.destination, core_target)
                    < current_home_distance
                    and (
                        state.last_min_enemy_distance is None
                        or row.minimum_enemy_distance
                        >= state.last_min_enemy_distance
                    )
                ]
                if homeward_survivable:
                    filter_rejections["NOT_SAFE_CORE_PROGRESS"] = (
                        filter_rejections.get("NOT_SAFE_CORE_PROGRESS", 0)
                        + len(rows)
                        - len(homeward_survivable)
                    )
                    rows = homeward_survivable

        ordered = sorted(rows, key=lambda row: row.score)
        if ordered:
            first = ordered[0]
            waypoint = state.waypoint
            should_replan = (
                waypoint is None
                or state.stalled_ticks >= self.config.worker_escape_replan_ticks
            )
            if should_replan:
                waypoint = first.waypoint
            route_version = state.route_version + int(waypoint != state.waypoint)
            state = replace(
                state,
                waypoint=waypoint,
                route_version=route_version,
                waypoint_assigned_tick=(
                    world.tick
                    if waypoint != state.waypoint
                    else state.waypoint_assigned_tick
                ),
                waypoint_expires_tick=(
                    world.tick + self.config.worker_escape_waypoint_lease_ticks
                    if waypoint != state.waypoint
                    else state.waypoint_expires_tick
                ),
                waypoint_invalid_reason=None,
                last_waypoint_distance=(
                    None
                    if waypoint is None
                    else manhattan(worker.position, waypoint)
                ),
            )
            self.memory.worker_escape_states[worker.id] = state

        intents = [
            ActionIntent.move(
                worker.id,
                UnitMission.ESCAPE,
                20,
                row.direction,
                row.destination,
                risk=(
                    row.immediate_attackers * 100
                    + row.future_attackers * 10
                    + row.recent_threat
                    + row.heat
                ),
                tie_break=row.score,
                reason=state.phase,
                metadata=(
                    ("escape_phase", state.phase),
                    (
                        "safe_horizon",
                        self.config.worker_escape_plan_depth
                        if row.survival_terminals > 0
                        else 0,
                    ),
                    ("survival_terminals", row.survival_terminals),
                    ("first_step_heat", row.heat),
                    ("enemy_distance_before", state.last_min_enemy_distance),
                    ("enemy_distance_after", row.minimum_enemy_distance),
                    ("visible_enemy_distance_before", current_visible_minimum),
                    ("visible_enemy_distance_after", row.visible_minimum_distance),
                    (
                        "visible_vanguard_distance_before",
                        current_visible_vanguard_minimum,
                    ),
                    (
                        "visible_vanguard_distance_after",
                        row.visible_vanguard_minimum_distance,
                    ),
                    (
                        "nonfatal_budget_used",
                        row.immediate_attackers > 0,
                    ),
                    ("escape_loop_period", state.loop_period),
                    ("escape_waypoint", row.waypoint),
                    ("escape_waypoint_lease", state.waypoint),
                    ("escape_waypoint_assigned_tick", state.waypoint_assigned_tick),
                    ("escape_waypoint_expires_tick", state.waypoint_expires_tick),
                    ("escape_waypoint_distance", state.last_waypoint_distance),
                    (
                        "fog_homeward_allowed",
                        state.phase == "FLEEING"
                        or row.destination == core_retreat_first_step
                        or manhattan(row.destination, core_target)
                        < current_home_distance,
                    ),
                    ("core_retreat_target", core_target),
                    ("core_retreat_first_step", core_retreat_first_step),
                    ("core_retreat_route_mode", core_retreat_route_mode),
                    ("escape_route_version", state.route_version),
                    (
                        "escape_filter_rejections",
                        tuple(sorted(filter_rejections.items())),
                    ),
                ) + row.viability.metadata,
            )
            for row in ordered
        ]
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.ESCAPE,
                21 if rows else 20,
                reason=(
                    "ESCAPE_BLOCKED_THIS_TICK"
                    if rows
                    else "NO_SURVIVABLE_ROUTE"
                ),
            )
        )
        return intents

    def _safe_exits(
        self,
        world,
        projection,
        position,
        hp,
        *,
        origin=None,
        threat_ids=(),
    ):
        return self.safety.forward_safe_exits(
            world,
            projection,
            position,
            hp,
            origin=origin,
            threat_ids=threat_ids,
        )

    def _survival_horizon(
        self,
        world,
        projection,
        start,
        hp,
        *,
        origin=None,
        threat_ids=(),
    ):
        return self.safety.survival_terminals(
            world,
            projection,
            start,
            hp,
            origin=origin,
            depth_limit=self.config.worker_escape_plan_depth,
            node_limit=self.config.worker_escape_plan_node_limit,
            threat_ids=threat_ids,
        )

    def _escape_outlook(
        self,
        world: WorldModel,
        projection: TacticalMap,
        start: Position,
        origin: Position,
        hp: int,
        threat_ids: tuple[UUID, ...],
        control_centers: tuple[Position, ...] = (),
    ) -> tuple[Position, int, int, int]:
        """Choose a bounded, zero-controller egress waypoint for one first step."""

        frontier: list[tuple[Position, int, int, int, int]] = [
            (
                start,
                1,
                projection.immediate_attackers(start),
                projection.future_attackers(start),
                projection.worker_exposure(start)[2],
            )
        ]
        visited = {origin, start}
        terminals: list[tuple[tuple[object, ...], Position, int, int, int]] = []
        while frontier and len(visited) <= self.config.worker_escape_plan_node_limit:
            cell, depth, max_immediate, max_future, accumulated_heat = frontier.pop(0)
            minimum, total = self._survival_distances(
                projection,
                cell,
                threat_ids,
                control_centers,
            )
            visible_minimum, visible_total = self._visible_enemy_distances(
                projection,
                cell,
                threat_ids,
            )
            if depth >= self.config.worker_escape_plan_depth:
                terminals.append(
                    (
                        (
                            max_immediate,
                            max_future,
                            -visible_minimum,
                            -visible_total,
                            -minimum,
                            -total,
                            accumulated_heat,
                            manhattan(
                                cell,
                                world.core.destination or world.core.position,
                            ),
                            cell,
                        ),
                        cell,
                        minimum,
                        total,
                        accumulated_heat,
                    )
                )
                continue
            onward = []
            for _, neighbor in cardinal_neighbors(cell):
                if (
                    neighbor in visited
                    or neighbor in world.known_obstacles
                    or neighbor in projection.hostile_occupied
                    or neighbor not in world.known_passable
                    or projection.immediate_attackers(neighbor) >= hp
                    or projection.future_attackers(neighbor) >= hp
                ):
                    continue
                onward.append(neighbor)
            for neighbor in onward:
                visited.add(neighbor)
                frontier.append(
                    (
                        neighbor,
                        depth + 1,
                        max(max_immediate, projection.immediate_attackers(neighbor)),
                        max(max_future, projection.future_attackers(neighbor)),
                        accumulated_heat + projection.worker_exposure(neighbor)[2],
                    )
                )
        if not terminals:
            minimum, total = self._survival_distances(
                projection,
                start,
                threat_ids,
                control_centers,
            )
            return start, minimum, total, projection.worker_exposure(start)[2]
        _, waypoint, minimum, total, heat = min(terminals, key=lambda row: row[0])
        return waypoint, minimum, total, heat

    def _recent_threat(self, projection, position, ids):
        score = 0
        for threat_id in ids:
            enemy = projection.enemy(threat_id)
            if enemy is None:
                continue
            effective = max(
                0,
                manhattan(position, enemy.observed_position) - enemy.age,
            )
            if effective <= self.config.worker_escape_clearance_radius:
                score += self.config.worker_escape_clearance_radius - effective + 1
        return score

    @staticmethod
    def _enemy_distances(projection, position, ids):
        distances: list[int] = []
        for threat_id in ids:
            enemy = projection.enemy(threat_id)
            if enemy is not None:
                distances.append(
                    max(0, manhattan(position, enemy.observed_position) - enemy.age)
                )
        return min(distances, default=99), sum(distances)

    @classmethod
    def _survival_distances(
        cls,
        projection,
        position: Position,
        ids: tuple[UUID, ...],
        control_centers: tuple[Position, ...],
    ) -> tuple[int, int]:
        """Return one conservative distance frontier for every survival cause.

        Enemy tracks shrink with age while enemy-Core control zones remain
        spatially exact.  Combining them prevents FOG_RETREAT and
        CORE_DISENGAGE from alternately pulling a Worker back into the other
        hazard.
        """

        distances: list[int] = []
        for threat_id in ids:
            enemy = projection.enemy(threat_id)
            if enemy is not None:
                distances.append(
                    max(0, manhattan(position, enemy.observed_position) - enemy.age)
                )
        distances.extend(manhattan(position, center) for center in control_centers)
        return min(distances, default=99), sum(distances)

    @staticmethod
    def _visible_enemy_distances(projection, position, ids):
        distances: list[int] = []
        for threat_id in ids:
            enemy = projection.enemy(threat_id)
            if enemy is not None and enemy.visible_now:
                distances.append(manhattan(position, enemy.observed_position))
        return min(distances, default=99), sum(distances)

    @staticmethod
    def _visible_vanguard_distance(projection, position, ids):
        distances: list[int] = []
        for threat_id in ids:
            enemy = projection.enemy(threat_id)
            if (
                enemy is not None
                and enemy.visible_now
                and enemy.unit_type is UnitType.VANGUARD
            ):
                distances.append(manhattan(position, enemy.observed_position))
        return min(distances, default=99)

    def _escape_loop_period(self, history: tuple[Position, ...]) -> int | None:
        for period in range(1, self.config.worker_escape_max_loop_period + 1):
            required = period * 2
            if len(history) < required:
                continue
            previous = history[-required:-period]
            current = history[-period:]
            if previous == current and len(set(current)) > 1:
                return period
        return None

    def _cargo(self, world, projection, worker, service):
        assert world.core is not None
        if worker.id in {row[0] for row in service.blocking_units}:
            return self._clear_service_cell(
                world,
                projection,
                worker,
                service,
            )
        if service.paused_reason == "LANE_THREATENED":
            return [
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.RETURN_CARGO,
                    49,
                    reason=f"SERVICE_PAUSED_{service.paused_reason}",
                )
            ]
        reservation = next(
            (
                row
                for row in service.return_reservations
                if row.worker_id == worker.id
            ),
            None,
        )
        if (
            worker.position == world.core.position
            and world.resources >= world.resource_capacity
            and service.exit_cell is not None
        ):
            route = self._route(
                world,
                projection,
                worker,
                service.exit_cell,
                service,
                logistics=True,
                allow_directional_fallback=True,
            )
            if route is not None and route.first_direction is not None:
                viability = move_viability(
                    world,
                    worker.position,
                    route.first_position,
                    target=service.exit_cell,
                    node_limit=min(self.config.path_node_limit, 256),
                    require_open_area=True,
                )
                if not viability.viable:
                    return [
                        ActionIntent.simple(
                            worker.id,
                            IntentAction.WAIT,
                            UnitMission.RETURN_CARGO,
                            50,
                            reason="NO_VIABLE_CONTINUATION",
                            metadata=viability.metadata,
                        )
                    ]
                return [
                    ActionIntent.move(
                        worker.id,
                        UnitMission.RETURN_CARGO,
                        50,
                        route.first_direction,
                        route.first_position,
                        reason="CORE_FULL_RELEASE_SLOT",
                        metadata=(("allow_protected", True),)
                        + viability.metadata,
                    )
                ]
        if reservation is None or reservation.status == "UNROUTABLE":
            return [
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.RETURN_CARGO,
                    51,
                    reason=(
                        "SEGMENT_ROUTE_EXHAUSTED"
                        if reservation is not None
                        and reservation.delay_reason == "SEGMENT_ROUTE_EXHAUSTED"
                        else "FULL_ROUTE_EXHAUSTED"
                    ),
                )
            ]
        if reservation.status == "WAIT_FOR_DEPARTURE":
            active_cells = {
                lease.cell
                for lease in service.service_cell_leases
                if lease.active and lease.owner_id != worker.id
            }
            if worker.position in active_cells:
                return self._clear_service_cell(
                    world,
                    projection,
                    worker,
                    service,
                )
            transit_hold = dict(service.overflow_slots).get(worker.id)
            if transit_hold is not None and worker.position == transit_hold:
                return [
                    ActionIntent.simple(
                        worker.id,
                        IntentAction.WAIT,
                        UnitMission.RETURN_CARGO,
                        51,
                        target_position=transit_hold,
                        reason="WAIT_AT_TRANSIT_HOLD",
                        metadata=(
                            ("transit_hold", transit_hold),
                            ("scheduled_deposit_tick", reservation.scheduled_deposit_tick),
                            ("departure_tick", reservation.departure_tick),
                            ("slack_ticks", reservation.slack_ticks),
                            ("remaining_distance", reservation.route_distance),
                        ),
                    )
                ]
            if transit_hold is not None:
                hold_route = self._route(
                    world,
                    projection,
                    worker,
                    transit_hold,
                    service,
                    logistics=True,
                )
                if hold_route is not None and hold_route.first_direction is not None:
                    return self._cargo_route_intents(
                        world,
                        projection,
                        worker,
                        service,
                        reservation,
                        transit_hold,
                        hold_route,
                        priority=51,
                        reason="SERVICE_TRANSIT_HOLD_APPROACH",
                    )
            # No reachable transit hold exists.  Continue along the exact
            # return route instead of freezing at a remote origin; the rolling
            # calendar will compress or move the appointment on the next Tick.
        if reservation.first_direction is not None and reservation.first_position is not None:
            route_target = reservation.route_target or world.core.position
            ready = worker.id in service.ready_depositors
            priority = 49 if service.admission_id == worker.id else (50 if ready else 51)
            reason = "SERVICE_ADMISSION" if reservation.first_position == world.core.position else (
                "SERVICE_PIPELINE_ADVANCE" if ready else "SERVICE_QUEUE_APPROACH"
            )
            primary = Route(
                reservation.route_distance or 1,
                reservation.first_direction,
                reservation.first_position,
            )
            return self._cargo_route_intents(
                world,
                projection,
                worker,
                service,
                reservation,
                route_target,
                primary,
                priority=priority,
                reason=reason,
            )
        return [
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.RETURN_CARGO,
                49 if service.admission_id == worker.id else 51,
                reason="WAITING_FOR_DEPOSIT_ACTION",
            )
        ]

    def _cargo_route_intents(
        self,
        world,
        projection,
        worker,
        service,
        reservation,
        route_target,
        primary_route,
        *,
        priority,
        reason,
    ):
        """Return the shared service step plus at most two safe alternatives."""

        assert world.core is not None
        routes: list[Route] = [primary_route]
        blocked_first_steps: set[Position] = set()
        if primary_route.first_position is not None:
            blocked_first_steps.add(primary_route.first_position)
        if worker.position not in service.queue_cells:
            while len(routes) < 3:
                alternate = self._route(
                    world,
                    projection,
                    worker,
                    route_target,
                    service,
                    logistics=True,
                    extra_blocked=frozenset(blocked_first_steps),
                )
                if (
                    alternate is None
                    or alternate.first_direction is None
                    or alternate.first_position is None
                    or alternate.first_position in blocked_first_steps
                ):
                    break
                blocked_first_steps.add(alternate.first_position)
                routes.append(alternate)

        service_job = next(
            (job for job in service.jobs if job.actor_id == worker.id),
            None,
        )
        kind = self.transit.kind_for(service_job, worker)
        descriptor, intents, rejected_first_steps = self.transit.intents(
            world,
            projection,
            worker,
            route_target,
            tuple(routes),
            mission=UnitMission.RETURN_CARGO,
            priority=priority,
            reason=reason,
            kind=kind,
            job=service_job,
            metadata=(
                ("service_slot", route_target),
                ("scheduled_deposit_tick", reservation.scheduled_deposit_tick),
                ("departure_tick", reservation.departure_tick),
                ("route_mode", reservation.route_mode),
                ("lane_version", reservation.lane_version),
                ("waypoint", reservation.waypoint),
            ),
        )
        self.memory.service_transit_routes[worker.id] = descriptor
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.RETURN_CARGO,
                priority + 3,
                reason=(
                    "WAITING_FOR_CORE_SLOT"
                    if service.admission_id == worker.id
                    else "WAITING_FOR_SERVICE_SLOT"
                ) if intents else (
                    "NO_SAFE_FIRST_STEP"
                    if rejected_first_steps
                    else "SEGMENT_ROUTE_EXHAUSTED"
                    if reservation.route_mode == "SEGMENTED"
                    else "FULL_ROUTE_EXHAUSTED"
                ),
                metadata=(
                    ("route_mode", reservation.route_mode),
                    ("lane_version", reservation.lane_version),
                    ("waypoint", reservation.waypoint),
                ),
            )
        )
        return intents

    def _clear_core(self, world, projection, unit, service, exploration=None):
        assert world.core is not None
        occupied = dict(world.occupied_cells)
        protected = {
            cell
            for cell in (service.entrance, *service.queue_cells)
            if cell is not None
        }
        preferred = service.exit_cell
        if preferred is not None and manhattan(preferred, world.core.position) != 1:
            preferred = None
        candidates = []
        scout_step = None if exploration is None else exploration[1].first_position
        scout_target = None if exploration is None else exploration[0]
        for index, (direction, destination) in enumerate(cardinal_neighbors(unit.position)):
            if destination in world.known_obstacles or destination in projection.hostile_occupied:
                continue
            if destination in protected and destination != preferred:
                continue
            viability = move_viability(
                world,
                unit.position,
                destination,
                # Clearing the physical Core slot is a local operation.  A
                # durable scout target may be hundreds of cells away and must
                # not make a safe adjacent exit fail merely because this
                # bounded check cannot prove the whole expedition route.
                target=None,
                blocked=frozenset(protected - {destination}),
                node_limit=min(self.config.path_node_limit, 128),
                require_continuation=False,
                require_open_area=True,
            )
            if not viability.viable:
                continue
            score = (
                projection.immediate_attackers(destination),
                projection.future_attackers(destination),
                int(destination != scout_step),
                int(destination != preferred),
                occupied.get(destination, 0),
                index,
            )
            candidates.append((score, direction, destination, viability))
        intents = [
            ActionIntent.move(
                unit.id,
                UnitMission.CLEAR_CORE,
                45,
                direction,
                destination,
                risk=score[0] * 100 + score[1] * 10,
                exclusive_destination=False,
                tie_break=score,
                reason=(
                    "SCOUT_CORE_EXIT"
                    if destination == scout_step
                    else (
                        "CORE_SERVICE_EXIT"
                        if destination == preferred
                        else "CORE_EXIT_ALTERNATE"
                    )
                ),
                metadata=(
                    ("allow_protected", destination == preferred),
                    ("allow_head_on_swap", True),
                    (
                        "allow_service_overlap",
                        occupied.get(destination, 0) == 1,
                    ),
                    ("scout_target", scout_target),
                    ("local_exit_only", True),
                ) + viability.metadata,
            )
            for score, direction, destination, viability in sorted(
                candidates, key=lambda row: row[0]
            )
        ]
        intents.append(
            ActionIntent.simple(
                unit.id,
                IntentAction.WAIT,
                UnitMission.CLEAR_CORE,
                46,
                reason="CORE_EXIT_BLOCKED_THIS_TICK" if candidates else "CORE_EXIT_BLOCKED",
            )
        )
        return intents

    def _clear_service_lane(self, world, projection, unit, service):
        """Continue egress until an empty Worker is outside service cells."""

        assert world.core is not None
        occupied = dict(world.occupied_cells)
        protected = {
            world.core.position,
            *(cell for cell in (service.entrance, service.exit_cell) if cell is not None),
            *service.queue_cells,
        }
        rows = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(unit.position)
        ):
            if (
                destination in protected
                or destination in world.known_obstacles
                or destination in projection.hostile_occupied
            ):
                continue
            viability = move_viability(
                world,
                unit.position,
                destination,
                blocked=frozenset(protected),
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            )
            if not viability.viable:
                continue
            score = (
                projection.immediate_attackers(destination),
                projection.future_attackers(destination),
                projection.worker_exposure(destination)[2],
                int(destination in world.visible_resources),
                occupied.get(destination, 0),
                -manhattan(destination, world.core.position),
                index,
            )
            rows.append((score, direction, destination, viability))
        intents = [
            ActionIntent.move(
                unit.id,
                UnitMission.CLEAR_CORE,
                45,
                direction,
                destination,
                risk=score[0] * 100 + score[1] * 10 + score[2],
                tie_break=score,
                reason=(
                    "CLEAR_SERVICE_EXIT"
                    if unit.position == service.exit_cell
                    else "CLEAR_SERVICE_LANE"
                ),
                metadata=(
                    ("allow_protected", True),
                    ("allow_head_on_swap", False),
                    (
                        "allow_service_overlap",
                        occupied.get(destination, 0) == 1,
                    ),
                ) + viability.metadata,
            )
            for score, direction, destination, viability in sorted(
                rows, key=lambda row: row[0]
            )
        ]
        intents.append(
            ActionIntent.simple(
                unit.id,
                IntentAction.WAIT,
                UnitMission.CLEAR_CORE,
                46,
                reason=(
                    "SERVICE_EGRESS_BLOCKED_THIS_TICK"
                    if rows
                    else "SERVICE_EGRESS_BLOCKED"
                ),
            )
        )
        return intents

    def _clear_service_cell(self, world, projection, unit, service):
        """Move a non-owner out of a service lease before its live window."""

        assert world.core is not None
        occupied = dict(world.occupied_cells)
        leased = {
            lease.cell
            for lease in service.service_cell_leases
            if lease.active and lease.owner_id != unit.id
        }
        infrastructure = {
            world.core.position,
            *(cell for cell in (service.entrance, service.exit_cell) if cell is not None),
            *service.queue_cells,
        }
        rows = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(unit.position)
        ):
            if (
                destination in leased
                or destination in infrastructure
                or destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or projection.immediate_attackers(destination) >= unit.hp
            ):
                continue
            viability = move_viability(
                world,
                unit.position,
                destination,
                blocked=frozenset(leased | infrastructure),
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            )
            if not viability.viable:
                continue
            score = (
                projection.future_attackers(destination),
                projection.worker_exposure(destination)[2],
                occupied.get(destination, 0),
                -manhattan(destination, world.core.position),
                index,
            )
            rows.append((score, direction, destination, viability))
        intents = [
            ActionIntent.move(
                unit.id,
                UnitMission.CLEAR_SERVICE_CELL,
                44,
                direction,
                destination,
                risk=score[0] * 10 + score[1],
                exclusive_destination=True,
                tie_break=score,
                reason="CLEAR_FUTURE_SERVICE_CELL",
                metadata=(
                    ("allow_protected", False),
                    ("allow_head_on_swap", True),
                ) + viability.metadata,
            )
            for score, direction, destination, viability in sorted(
                rows, key=lambda row: row[0]
            )
        ]
        intents.append(
            ActionIntent.simple(
                unit.id,
                IntentAction.WAIT,
                UnitMission.CLEAR_SERVICE_CELL,
                45,
                reason=(
                    "CLEAR_SERVICE_CELL_BLOCKED_THIS_TICK"
                    if rows
                    else "NO_SAFE_SERVICE_CELL_EXIT"
                ),
            )
        )
        return intents

    def _hold_cargo_outside_service_lane(self, world, projection, worker, service):
        """Stage non-active carriers near home without freezing them in place."""

        assert world.core is not None
        protected = {
            world.core.position,
            *(cell for cell in (service.entrance, service.exit_cell) if cell is not None),
            *service.queue_cells,
        }
        scheduled_tick = dict(service.scheduled_deposits).get(worker.id)
        assigned_overflow = dict(service.overflow_slots).get(worker.id)
        if assigned_overflow is not None:
            if worker.position == assigned_overflow:
                return [
                    ActionIntent.simple(
                        worker.id,
                        IntentAction.WAIT,
                        UnitMission.RETURN_CARGO,
                        52,
                        target_position=assigned_overflow,
                        reason="WAITING_FOR_SCHEDULED_DEPOSIT",
                        metadata=(
                            ("scheduled_deposit_tick", scheduled_tick),
                            ("staging_cell", assigned_overflow),
                        ),
                    )
                ]
            overflow_route = self._route(
                world,
                projection,
                worker,
                assigned_overflow,
                service,
                logistics=True,
                allow_directional_fallback=True,
                extra_blocked=frozenset(protected - {assigned_overflow}),
            )
            if overflow_route is not None and overflow_route.first_direction is not None:
                viability = move_viability(
                    world,
                    worker.position,
                    overflow_route.first_position,
                    target=assigned_overflow,
                    blocked=frozenset(protected - {assigned_overflow}),
                    node_limit=min(self.config.path_node_limit, 512),
                    require_continuation=(
                        overflow_route.first_position != assigned_overflow
                    ),
                )
                if not viability.viable:
                    overflow_route = None
            if overflow_route is not None and overflow_route.first_direction is not None:
                return [
                    ActionIntent.move(
                        worker.id,
                        UnitMission.RETURN_CARGO,
                        51,
                        overflow_route.first_direction,
                        overflow_route.first_position,
                        risk=self._risk(projection, overflow_route.first_position),
                        exclusive_destination=True,
                        tie_break=(overflow_route.distance,),
                        reason="MOVE_TO_SCHEDULED_STAGING",
                        metadata=(
                            ("overflow_slot", assigned_overflow),
                            ("scheduled_deposit_tick", scheduled_tick),
                        ) + viability.metadata,
                    ),
                    ActionIntent.simple(
                        worker.id,
                        IntentAction.WAIT,
                        UnitMission.RETURN_CARGO,
                        53,
                        target_position=assigned_overflow,
                        reason="OVERFLOW_MOVE_BLOCKED_THIS_TICK",
                    ),
                ]
        current_distance = manhattan(worker.position, world.core.position)
        staging_radius = self.config.service_lane_depth + 2
        if (
            worker.position not in protected
            and current_distance <= staging_radius
        ):
            return [
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.RETURN_CARGO,
                    52,
                    target_position=worker.position,
                    reason="WAITING_AT_SERVICE_STAGING",
                )
            ]
        occupied = dict(world.occupied_cells)
        rows = []
        route = self._route(
            world,
            projection,
            worker,
            service.queue_cells[-1]
            if service.queue_cells
            else world.core.position,
            service,
            logistics=True,
            allow_directional_fallback=True,
        )
        preferred_direction = None if route is None else route.first_direction
        returning_to_stage = current_distance > staging_radius
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(worker.position)
        ):
            if (
                destination in protected
                or destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or projection.immediate_attackers(destination) >= worker.hp
            ):
                continue
            stage_target = (
                service.queue_cells[-1]
                if service.queue_cells
                else world.core.position
            )
            viability = move_viability(
                world,
                worker.position,
                destination,
                target=stage_target,
                blocked=frozenset(protected - {stage_target}),
                node_limit=min(self.config.path_node_limit, 512),
                require_continuation=returning_to_stage,
                require_open_area=not returning_to_stage,
                terminal_exception=(
                    "CORE_SERVICE" if destination == world.core.position else None
                ),
            )
            if not viability.viable:
                continue
            destination_distance = manhattan(destination, world.core.position)
            score = (
                projection.future_attackers(destination),
                projection.worker_exposure(destination)[2],
                int(
                    destination_distance >= current_distance
                    if returning_to_stage
                    else destination_distance <= current_distance
                ),
                int(
                    returning_to_stage
                    and preferred_direction is not None
                    and direction is not preferred_direction
                ),
                occupied.get(destination, 0),
                (
                    destination_distance
                    if returning_to_stage
                    else -destination_distance
                ),
                index,
            )
            rows.append((score, direction, destination, viability))
        intents = [
            ActionIntent.move(
                worker.id,
                UnitMission.RETURN_CARGO,
                48,
                direction,
                destination,
                risk=score[0] * 10 + score[1],
                tie_break=score,
                reason=(
                    "RETURN_TO_SERVICE_STAGING"
                    if returning_to_stage
                    else "CLEAR_SERVICE_APPROACH"
                ),
                metadata=(("allow_protected", False),) + viability.metadata,
            )
            for score, direction, destination, viability in sorted(
                rows, key=lambda row: row[0]
            )
        ]
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.RETURN_CARGO,
                49 if not rows else 53,
                reason=(
                    "SERVICE_APPROACH_DRAIN_BLOCKED"
                    if not rows
                    else (
                        "WAITING_FOR_STAGING_STEP"
                        if returning_to_stage
                        else "WAITING_OUTSIDE_SERVICE_LANE"
                    )
                ),
            )
        )
        return intents

    def _suspend_resource_search_leases(
        self,
        workers: tuple[EntitySnapshot, ...],
    ) -> None:
        """Keep angular identity while removing non-saturated search targets."""

        living = {worker.id for worker in workers}
        for worker_id, lease in tuple(self.memory.resource_search_leases.items()):
            if worker_id not in living:
                self.memory.resource_search_leases.pop(worker_id, None)
                continue
            self.memory.resource_search_leases[worker_id] = replace(
                lease,
                target=None,
                waypoint=None,
                last_route_distance=None,
                stalled_ticks=0,
                blocked_edge=None,
                backoff_until=0,
                information_gain=0,
                visible_gain=0,
                overlap_cells=0,
            )
            mission = self.memory.unit_missions.get(worker_id)
            if mission is not None and mission.mission is UnitMission.RESOURCE_SEARCH:
                self.memory.unit_missions.pop(worker_id, None)

    def _resource_search_assignments(
        self,
        world: WorldModel,
        projection: TacticalMap,
        workers: tuple[EntitySnapshot, ...],
        service: CoreServiceQueue,
    ) -> dict[UUID, tuple[Position, Route, int]]:
        """Assign sticky, unbounded and angularly dispersed frontier work.

        This is intentionally independent from ``_exploration_assignments``:
        the latter is the bounded mature-stockpile patrol.  A resource search
        advances from one proven reachable frontier to the next and therefore
        has no Core-radius ceiling.
        """

        assert world.core is not None
        self._sync_resource_search_leases(
            workers,
            world.tick,
            world.core.position,
        )
        if not workers:
            return {}

        assignments: dict[UUID, tuple[Position, Route, int]] = {}
        claimed: set[Position] = set()
        claimed_visible: set[Position] = set()
        last_visible = dict(world.cell_last_visible)
        global_frontiers_by_slot: dict[int, list[Position]] | None = None

        for worker in sorted(
            workers,
            key=lambda item: (
                self.memory.resource_search_leases[item.id].direction_slot,
                item.id.bytes,
            ),
        ):
            lease = self.memory.resource_search_leases[worker.id]
            blocked, costs = self._exploration_navigation(
                world,
                projection,
                worker,
                service,
            )
            history = self.memory.position_history.get(worker.id, ())
            loop_period = self._loop_period(history)
            if (
                loop_period is not None
                and lease.target is not None
                and (loop_period > 1 or lease.stalled_ticks >= 1)
            ):
                edge = (
                    (history[-1], history[-2])
                    if len(history) >= 2
                    else None
                )
                self._invalidate_resource_search_target(
                    worker.id,
                    world.tick,
                    lease.target,
                    edge=edge,
                )
                lease = self.memory.resource_search_leases[worker.id]

            if lease.target is not None:
                target_observed = (
                    lease.target in world.visible_cells
                    and lease.target not in world.visible_resources
                )
                target_exhausted = (
                    lease.target in world.known_passable
                    and information_gain(
                        lease.target,
                        tick=world.tick,
                        last_visible=last_visible,
                        refresh_ticks=self.config.exploration_refresh_ticks,
                    ) <= 0
                )
                if target_observed or target_exhausted:
                    self._invalidate_resource_search_target(
                        worker.id,
                        world.tick,
                        lease.target,
                    )
                    lease = self.memory.resource_search_leases[worker.id]

            routed = None
            if (
                lease.target is not None
                and lease.target not in claimed
                and self.memory.target_backoff_until.get(lease.target, -1) < world.tick
            ):
                routed = self._resource_search_route(
                    world,
                    worker,
                    lease.target,
                    blocked,
                    costs,
                )
                if routed is not None:
                    route, waypoint, estimate = routed
                    progressed = bool(
                        lease.last_route_distance is None
                        or estimate < lease.last_route_distance
                    )
                    stalled = 0 if progressed else lease.stalled_ticks + 1
                    if stalled >= self.config.resource_search_stall_ticks:
                        self._invalidate_resource_search_target(
                            worker.id,
                            world.tick,
                            lease.target,
                            edge=(worker.position, route.first_position),
                        )
                        lease = self.memory.resource_search_leases[worker.id]
                        routed = None
                    else:
                        lease = replace(
                            lease,
                            waypoint=waypoint,
                            last_position=worker.position,
                            last_route_distance=estimate,
                            stalled_ticks=stalled,
                        )

            if routed is None:
                distances, parents = weighted_distance_field(
                    world,
                    worker.position,
                    node_limit=min(
                        self.config.path_node_limit,
                        self.config.distance_field_node_limit,
                    ),
                    blocked=blocked,
                    cell_costs=costs,
                )
                local = exploration_candidates(
                    world,
                    worker.position,
                    distances=distances,
                    # Search every cell actually reached by this bounded field;
                    # the Core distance is deliberately irrelevant.
                    search_radius=1 << 20,
                    limit=max(32, self.config.exploration_candidate_limit),
                    backoff=frozenset(self.memory.target_backoff_until),
                )
                search_candidates = local
                if not search_candidates:
                    if global_frontiers_by_slot is None:
                        global_frontiers_by_slot = {}
                        for frontier in self._global_resource_frontiers(world):
                            slot = self._resource_search_direction_slot(
                                world.core.position,
                                frontier,
                                self.config.resource_search_direction_slots,
                            )
                            global_frontiers_by_slot.setdefault(slot, []).append(
                                frontier
                            )
                    rows: list[Position] = []
                    slot_count = self.config.resource_search_direction_slots
                    for offset in range(slot_count // 2 + 1):
                        slot_rows = (
                            (lease.direction_slot + offset) % slot_count,
                            (lease.direction_slot - offset) % slot_count,
                        )
                        for slot in dict.fromkeys(slot_rows):
                            rows.extend(global_frontiers_by_slot.get(slot, ())[:16])
                        if len(rows) >= 64:
                            break
                    search_candidates = tuple(rows[:64])
                ordered_candidates = self._rank_resource_frontiers(
                    world,
                    worker,
                    lease,
                    search_candidates,
                    claimed,
                    claimed_visible,
                    last_visible,
                )
                chosen = None
                for target, gain, visible_gain, overlap in ordered_candidates[:12]:
                    route = route_from_field(
                        worker.position,
                        target,
                        distances,
                        parents,
                        obstacles=world.known_obstacles,
                        allow_unknown_endpoint=True,
                    )
                    routed_candidate = None
                    if route is not None and route.first_direction is not None:
                        viability = self._worker_move_viability(
                            world,
                            worker.position,
                            route.first_position,
                            target=target,
                            blocked=blocked,
                        )
                        if viability.viable:
                            routed_candidate = (
                                Route(
                                    route.distance,
                                    route.first_direction,
                                    route.first_position,
                                    viability,
                                ),
                                target,
                                route.distance,
                            )
                    if routed_candidate is None:
                        routed_candidate = self._resource_search_route(
                            world,
                            worker,
                            target,
                            blocked,
                            costs,
                        )
                    if routed_candidate is None:
                        continue
                    chosen = (
                        target,
                        gain,
                        visible_gain,
                        overlap,
                        routed_candidate,
                    )
                    break
                if chosen is None:
                    local_rows = []
                    for index, (direction, target) in enumerate(
                        cardinal_neighbors(worker.position)
                    ):
                        if (
                            target in blocked
                            or target in world.known_obstacles
                            or target not in world.known_passable
                            or target in claimed
                        ):
                            continue
                        gain = information_gain(
                            target,
                            tick=world.tick,
                            last_visible=last_visible,
                            refresh_ticks=self.config.exploration_refresh_ticks,
                        )
                        if gain <= 0:
                            continue
                        viability = self._worker_move_viability(
                            world,
                            worker.position,
                            target,
                            target=target,
                            blocked=blocked,
                            node_limit=max(1, self.config.path_node_limit),
                        )
                        if not viability.viable:
                            continue
                        visible = scout_visible_cells(
                            target,
                            world.known_obstacles,
                        )
                        visible_gain = len(visible - claimed_visible)
                        overlap = len(visible & claimed_visible)
                        candidate_slot = self._resource_search_direction_slot(
                            world.core.position,
                            target,
                            self.config.resource_search_direction_slots,
                        )
                        angular_gap = min(
                            (
                                candidate_slot - lease.direction_slot
                            ) % self.config.resource_search_direction_slots,
                            (
                                lease.direction_slot - candidate_slot
                            ) % self.config.resource_search_direction_slots,
                        )
                        local_rows.append(
                            (
                                (
                                    angular_gap,
                                    -visible_gain,
                                    overlap,
                                    -gain,
                                    index,
                                ),
                                target,
                                gain,
                                visible_gain,
                                overlap,
                                Route(1, direction, target, viability),
                            )
                        )
                    if local_rows:
                        _, target, gain, visible_gain, overlap, local_route = min(
                            local_rows,
                            key=lambda row: row[0],
                        )
                        chosen = (
                            target,
                            gain,
                            visible_gain,
                            overlap,
                            (local_route, target, 1),
                        )
                if chosen is None:
                    self.memory.resource_search_leases[worker.id] = replace(
                        lease,
                        target=None,
                        waypoint=None,
                        last_position=worker.position,
                        last_route_distance=None,
                        stalled_ticks=lease.stalled_ticks + 1,
                    )
                    continue
                target, gain, visible_gain, overlap, routed = chosen
                route, waypoint, estimate = routed
                lease = replace(
                    lease,
                    target=target,
                    waypoint=waypoint,
                    assigned_tick=world.tick,
                    last_position=worker.position,
                    last_route_distance=estimate,
                    stalled_ticks=0,
                    route_version=lease.route_version + 1,
                    blocked_edge=None,
                    backoff_until=0,
                    information_gain=gain,
                    visible_gain=visible_gain,
                    overlap_cells=overlap,
                )

            if routed is None or lease.target is None:
                self.memory.resource_search_leases[worker.id] = lease
                continue
            route, waypoint, estimate = routed
            lease = replace(
                lease,
                waypoint=waypoint,
                last_position=worker.position,
                last_route_distance=estimate,
            )
            self.memory.resource_search_leases[worker.id] = lease
            self.memory.unit_missions[worker.id] = MissionState(
                UnitMission.RESOURCE_SEARCH,
                lease.target,
                lease.assigned_tick,
                failures=lease.stalled_ticks,
            )
            claimed.add(lease.target)
            claimed_visible.update(
                scout_visible_cells(lease.target, world.known_obstacles)
            )
            assignments[worker.id] = (
                lease.target,
                route,
                lease.information_gain,
            )
        return assignments

    def _sync_resource_search_leases(
        self,
        workers: tuple[EntitySnapshot, ...],
        tick: int,
        core_position: Position,
    ) -> None:
        active = tuple(sorted(workers, key=lambda item: item.id.bytes))
        active_ids = {worker.id for worker in active}
        for worker_id in tuple(self.memory.resource_search_leases):
            if worker_id not in active_ids:
                self.memory.resource_search_leases.pop(worker_id, None)
                mission = self.memory.unit_missions.get(worker_id)
                if mission is not None and mission.mission is UnitMission.RESOURCE_SEARCH:
                    self.memory.unit_missions.pop(worker_id, None)

        slots = self.config.resource_search_direction_slots
        used = {
            lease.direction_slot % slots
            for lease in self.memory.resource_search_leases.values()
        }
        for worker in active:
            if worker.id in self.memory.resource_search_leases:
                continue
            slot = (
                self._resource_search_direction_slot(
                    core_position,
                    worker.position,
                    slots,
                )
                if not used and worker.position != core_position
                else self._largest_search_gap_slot(used, slots)
            )
            used.add(slot)
            self.memory.resource_search_leases[worker.id] = ResourceSearchLease(
                worker_id=worker.id,
                direction_slot=slot,
                target=None,
                waypoint=None,
                assigned_tick=tick,
                last_position=worker.position,
            )

    @staticmethod
    def _largest_search_gap_slot(used: set[int], slots: int) -> int:
        if not used:
            return 0
        ordered = sorted(used)
        rows = []
        for index, start in enumerate(ordered):
            end = ordered[(index + 1) % len(ordered)]
            gap = (end - start) % slots
            if len(ordered) == 1:
                gap = slots
            rows.append((-gap, start, (start + max(1, gap // 2)) % slots))
        for _, _, candidate in sorted(rows):
            if candidate not in used:
                return candidate
        return next(slot for slot in range(slots) if slot not in used)

    def _global_resource_frontiers(self, world: WorldModel) -> tuple[Position, ...]:
        rows: set[Position] = set()
        for cell in world.known_passable:
            for _, neighbor in cardinal_neighbors(cell):
                if (
                    neighbor not in world.known_passable
                    and neighbor not in world.known_obstacles
                    and neighbor not in world.visible_cells
                    and self.memory.target_backoff_until.get(neighbor, -1) < world.tick
                ):
                    rows.add(neighbor)
        return tuple(sorted(rows))

    def _rank_resource_frontiers(
        self,
        world: WorldModel,
        worker: EntitySnapshot,
        lease: ResourceSearchLease,
        candidates: tuple[Position, ...],
        claimed: set[Position],
        claimed_visible: set[Position],
        last_visible: dict[Position, int],
    ) -> tuple[tuple[Position, int, int, int], ...]:
        assert world.core is not None
        slots = self.config.resource_search_direction_slots
        rows = []
        for candidate in candidates:
            if (
                candidate in claimed
                or candidate in world.known_obstacles
                or self.memory.target_backoff_until.get(candidate, -1) >= world.tick
            ):
                continue
            gain = information_gain(
                candidate,
                tick=world.tick,
                last_visible=last_visible,
                refresh_ticks=self.config.exploration_refresh_ticks,
            )
            if gain <= 0:
                continue
            visible = scout_visible_cells(candidate, world.known_obstacles)
            visible_gain = len(visible - claimed_visible)
            overlap = len(visible & claimed_visible)
            candidate_slot = self._resource_search_direction_slot(
                world.core.position,
                candidate,
                slots,
            )
            angular_gap = min(
                (candidate_slot - lease.direction_slot) % slots,
                (lease.direction_slot - candidate_slot) % slots,
            )
            rows.append(
                (
                    (
                        angular_gap,
                        -visible_gain,
                        overlap,
                        -gain,
                        manhattan(worker.position, candidate),
                        # Core range is a weak deterministic tie break only.
                        -manhattan(world.core.position, candidate),
                        candidate,
                    ),
                    candidate,
                    gain,
                    visible_gain,
                    overlap,
                )
            )
        rows.sort(key=lambda row: row[0])
        return tuple((target, gain, visible_gain, overlap) for _, target, gain, visible_gain, overlap in rows)

    @staticmethod
    def _resource_search_direction_slot(
        core: Position,
        target: Position,
        slots: int,
    ) -> int:
        if target == core:
            return 0
        angle = (atan2(target[1] - core[1], target[0] - core[0]) + pi / 2) % (
            2 * pi
        )
        width = 2 * pi / slots
        return int((angle + width / 2) // width) % slots

    def _resource_search_route(
        self,
        world: WorldModel,
        worker: EntitySnapshot,
        target: Position,
        blocked: frozenset[Position],
        costs: dict[Position, int],
    ) -> tuple[Route, Position, int] | None:
        lease = self.memory.resource_search_leases.get(worker.id)
        effective = set(blocked)
        if (
            lease is not None
            and lease.blocked_edge is not None
            and worker.position == lease.blocked_edge[0]
            and world.tick <= lease.backoff_until
        ):
            effective.add(lease.blocked_edge[1])
        effective.discard(worker.position)
        route = weighted_route_to(
            world,
            worker.position,
            target,
            node_limit=self.config.path_node_limit,
            blocked=frozenset(effective),
            cell_costs=costs,
            allow_unknown_endpoint=True,
        )
        waypoint = target
        segmented = False
        if route is None:
            progress = weighted_progress_route(
                world,
                worker.position,
                target,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(effective),
                cell_costs=costs,
            )
            if progress is None:
                return None
            route, waypoint = progress
            segmented = True
        if route.first_direction is None or route.first_position is None:
            return None
        viability = self._worker_move_viability(
            world,
            worker.position,
            route.first_position,
            target=waypoint,
            blocked=frozenset(effective),
        )
        if not viability.viable:
            return None
        proved = Route(
            route.distance,
            route.first_direction,
            route.first_position,
            viability,
        )
        estimate = route.distance + (manhattan(waypoint, target) if segmented else 0)
        return proved, waypoint, estimate

    def _invalidate_resource_search_target(
        self,
        worker_id: UUID,
        tick: int,
        target: Position,
        *,
        edge: tuple[Position, Position] | None = None,
    ) -> None:
        lease = self.memory.resource_search_leases.get(worker_id)
        if lease is None:
            return
        backoff_until = tick + self.config.resource_search_edge_backoff_ticks
        self.memory.target_backoff_until[target] = backoff_until
        self.memory.resource_search_leases[worker_id] = replace(
            lease,
            target=None,
            waypoint=None,
            assigned_tick=tick,
            last_route_distance=None,
            stalled_ticks=0,
            route_version=lease.route_version + 1,
            blocked_edge=edge,
            backoff_until=backoff_until,
            information_gain=0,
            visible_gain=0,
            overlap_cells=0,
        )
        mission = self.memory.unit_missions.get(worker_id)
        if mission is not None and mission.mission is UnitMission.RESOURCE_SEARCH:
            self.memory.unit_missions.pop(worker_id, None)

    @staticmethod
    def _loop_period(history: tuple[Position, ...]) -> int | None:
        for period in range(1, min(4, len(history) // 2) + 1):
            if history[-period:] == history[-2 * period : -period]:
                return period
        return None

    def _worker_move_viability(
        self,
        world: WorldModel,
        origin: Position,
        destination: Position,
        *,
        target: Position,
        blocked: frozenset[Position],
        node_limit: int | None = None,
        terminal_exception: str | None = None,
    ) -> MoveViability:
        """Single proof gate used by primary and alternate Worker routes."""

        return move_viability(
            world,
            origin,
            destination,
            target=target,
            blocked=blocked,
            node_limit=(
                min(self.config.path_node_limit, 512)
                if node_limit is None
                else node_limit
            ),
            require_continuation=terminal_exception is None,
            terminal_exception=terminal_exception,
        )

    def _exploration_assignments(self, world, projection, workers, service):
        assert world.core is not None
        self._ensure_scout_states(
            workers,
            world.tick,
            world.core.destination or world.core.position,
        )
        assignments: dict[UUID, tuple[Position, Route, int]] = {}
        claimed: set[Position] = set()
        last_visible = dict(world.cell_last_visible)
        home_alert = self._home_alert(world, projection)
        ordered = sorted(
            workers,
            key=lambda worker: (
                self.memory.worker_scout_states[worker.id].slot,
                self.memory.worker_scout_states[worker.id].target is not None,
                self.memory.worker_scout_states[worker.id].last_scan_tick
                if self.memory.worker_scout_states[worker.id].last_scan_tick is not None
                else -1,
                worker.id.bytes,
            ),
        )
        scan_attempts = 0
        for worker in ordered:
            state = self.memory.worker_scout_states[worker.id]
            outside_scout_band = (
                manhattan(worker.position, world.core.position)
                > self.config.exploration_sector_radii[-1]
            )
            if outside_scout_band and state.phase is not WorkerScoutPhase.RETURN_TO_BAND:
                state = replace(
                    state,
                    phase=WorkerScoutPhase.RETURN_TO_BAND,
                    target=None,
                    assigned_tick=world.tick,
                    best_route_cost=None,
                    stalled_ticks=0,
                )
                self.memory.worker_scout_states[worker.id] = state
                self.memory.unit_missions.pop(worker.id, None)
            mission = self.memory.unit_missions.get(worker.id)
            if (
                state.target is None
                and mission is not None
                and mission.mission is UnitMission.EXPLORE
                and mission.target is not None
            ):
                state = replace(
                    state,
                    target=mission.target,
                    assigned_tick=mission.assigned_tick,
                )
                self.memory.worker_scout_states[worker.id] = state

            returning_to_band = state.phase is WorkerScoutPhase.RETURN_TO_BAND
            target_expired = (
                state.target is not None
                and not returning_to_band
                and world.tick - state.assigned_tick >= self.config.exploration_scout_hold_ticks
            )
            target_observed = (
                state.target is not None
                and worker.position == state.target
                and worker.id not in self.memory.service_egress_worker_ids
            )
            looping = self._looping(worker.id)
            if target_observed and returning_to_band:
                self.memory.scout_return_route_leases.pop(worker.id, None)
                state = replace(
                    state,
                    phase=WorkerScoutPhase.SECTOR_SCOUT,
                    target=None,
                    assigned_tick=world.tick,
                    best_route_cost=None,
                    stalled_ticks=0,
                    backoff_until=0,
                    reachable_candidates=0,
                )
                self.memory.worker_scout_states[worker.id] = state
                self.memory.unit_missions.pop(worker.id, None)
                returning_to_band = False
                target_observed = False
            if looping or target_expired or target_observed:
                backoff_until = 0
                if state.target is not None and (looping or target_expired):
                    backoff_until = (
                        world.tick
                        + (
                            self.config.exploration_return_loop_backoff_ticks
                            if returning_to_band and looping
                            else self.config.exploration_target_backoff_ticks
                        )
                    )
                    self.memory.target_backoff_until[state.target] = backoff_until
                if looping and returning_to_band:
                    history = self.memory.position_history.get(worker.id, ())
                    edge = (
                        # Block the *next* reversal from the authoritative
                        # current cell back to the previous cell.  Recording
                        # the already-traversed edge in its forward direction
                        # leaves an A-B-A-B cycle completely unconstrained.
                        (history[-1], history[-2])
                        if len(history) >= 2
                        else None
                    )
                    old_lease = self.memory.scout_return_route_leases.get(worker.id)
                    if old_lease is not None:
                        self.memory.scout_return_route_leases[worker.id] = replace(
                            old_lease,
                            blocked_edge=edge,
                            backoff_until=backoff_until,
                            route_version=old_lease.route_version + 1,
                        )
                state = self._clear_scout_target(
                    state,
                    world.tick,
                    advance=state.phase is WorkerScoutPhase.SECTOR_SCOUT,
                    backoff_until=backoff_until,
                )
                self.memory.unit_missions.pop(worker.id, None)
                returning_to_band = state.phase is WorkerScoutPhase.RETURN_TO_BAND

            blocked, costs = self._exploration_navigation(
                world,
                projection,
                worker,
                service,
            )
            if returning_to_band or outside_scout_band:
                fallback = self._return_to_scout_band_assignment(
                    world,
                    projection,
                    worker,
                    state,
                    claimed,
                    blocked,
                    costs,
                )
                if fallback is not None:
                    state, route = fallback
                    self._commit_scout_assignment(
                        world,
                        worker,
                        state,
                        route,
                        claimed,
                        assignments,
                        last_visible,
                    )
                else:
                    self.memory.worker_scout_states[worker.id] = state
                continue
            if (
                state.target is not None
                and state.target not in claimed
                and self.memory.target_backoff_until.get(state.target, -1) < world.tick
            ):
                if (
                    worker.id in self.memory.service_egress_worker_ids
                    and worker.position == state.target
                ):
                    # CLEAR_CORE/CLEAR_SERVICE_LANE is merely the first leg of
                    # the already assigned scout mission.  Do not replace its
                    # durable target while the service choreography is still
                    # moving the Worker out of the protected cells.
                    self.memory.worker_scout_states[worker.id] = state
                    continue
                use_local_segment = (
                    home_alert
                    or manhattan(worker.position, state.target)
                    > self.config.exploration_search_radius
                )
                route = (
                    self._alert_exploration_step(
                        world, projection, worker, state.target, blocked
                    )
                    if use_local_segment
                    else weighted_route_to(
                        world,
                        worker.position,
                        state.target,
                        node_limit=min(self.config.path_node_limit, 512),
                        blocked=blocked,
                        cell_costs=costs,
                        allow_unknown_endpoint=True,
                    )
                )
                if route is None and not use_local_segment:
                    # A far persistent target must not require one search to
                    # span the whole map.  Advance by a safe, target-improving
                    # local step and retry bounded A* from the new position on
                    # the next Tick.
                    route = self._alert_exploration_step(
                        world,
                        projection,
                        worker,
                        state.target,
                        blocked,
                    )
                if route is not None and route.first_direction is not None:
                    state = self._record_scout_progress(
                        state,
                        worker,
                        route,
                        world.tick,
                    )
                    if state.stalled_ticks < self.config.exploration_stall_ticks:
                        self._commit_scout_assignment(
                            world,
                            worker,
                            state,
                            route,
                            claimed,
                            assignments,
                            last_visible,
                        )
                        continue
                if state.target is not None:
                    backoff_until = (
                        world.tick + self.config.exploration_target_backoff_ticks
                    )
                    self.memory.target_backoff_until[state.target] = backoff_until
                else:
                    backoff_until = 0
                state = self._clear_scout_target(
                    state,
                    world.tick,
                    advance=state.phase is WorkerScoutPhase.SECTOR_SCOUT,
                    backoff_until=backoff_until,
                )
                self.memory.unit_missions.pop(worker.id, None)

            if (
                not returning_to_band
                and not home_alert
                and scan_attempts < self.config.exploration_new_goal_budget
            ):
                scan_attempts += 1
                distances, parents = weighted_distance_field(
                    world,
                    worker.position,
                    node_limit=min(
                        self.config.path_node_limit,
                        self.config.distance_field_node_limit,
                        1_024,
                    ),
                    blocked=blocked,
                    cell_costs=costs,
                )
                candidates = exploration_candidates(
                    world,
                    worker.position,
                    distances=distances,
                    search_radius=self.config.exploration_search_radius,
                    limit=self.config.exploration_candidate_limit,
                    backoff=frozenset(self.memory.target_backoff_until),
                )
                rows = []
                for candidate in candidates:
                    band_radius = self.config.exploration_sector_radii[state.stage]
                    candidate_radius = manhattan(candidate, world.core.position)
                    if (
                        candidate in claimed
                        or candidate_radius > self.config.exploration_sector_radii[-1]
                        or abs(candidate_radius - band_radius) > 3
                        or scout_sector_index(
                            world.core.position,
                            candidate,
                        )
                        != state.sector_index
                    ):
                        continue
                    route = route_from_field(
                        worker.position,
                        candidate,
                        distances,
                        parents,
                        obstacles=world.known_obstacles,
                        allow_unknown_endpoint=True,
                    )
                    if route is None or route.first_direction is None:
                        continue
                    gain = information_gain(
                        candidate,
                        tick=world.tick,
                        last_visible=last_visible,
                        refresh_ticks=self.config.exploration_refresh_ticks,
                    )
                    overlap = sum(
                        max(0, 7 - manhattan(candidate, other))
                        for other in claimed
                    )
                    score = (
                        -gain,
                        route.distance,
                        overlap,
                        self.memory.visit_counts.get(candidate, 0),
                        candidate,
                    )
                    rows.append((score, candidate, route, gain))
                if rows:
                    _, target, route, gain = min(rows, key=lambda row: row[0])
                    phase = (
                        WorkerScoutPhase.FRONTIER
                        if target not in world.known_passable
                        else WorkerScoutPhase.STALE_REVISIT
                    )
                    state = replace(
                        state,
                        phase=phase,
                        target=target,
                        assigned_tick=world.tick,
                        best_route_cost=route.distance,
                        stalled_ticks=0,
                        backoff_until=0,
                        last_scan_tick=world.tick,
                        reachable_candidates=len(rows),
                    )
                    self._commit_scout_assignment(
                        world,
                        worker,
                        state,
                        route,
                        claimed,
                        assignments,
                        last_visible,
                        gain=gain,
                    )
                    continue
                state = replace(
                    state,
                    last_scan_tick=world.tick,
                    reachable_candidates=0,
                )

            fallback = self._sector_scout_assignment(
                world,
                projection,
                worker,
                service,
                state,
                claimed,
                blocked,
                costs,
                home_alert,
            )
            if fallback is None:
                fallback = self._local_scout_assignment(
                    world,
                    projection,
                    worker,
                    service,
                    state,
                    claimed,
                )
            if fallback is not None:
                state, route = fallback
                self._commit_scout_assignment(
                    world,
                    worker,
                    state,
                    route,
                    claimed,
                    assignments,
                    last_visible,
                )
            else:
                self.memory.worker_scout_states[worker.id] = state
        return assignments

    def _ensure_scout_states(
        self,
        workers,
        tick: int,
        core_position: Position,
    ) -> None:
        """Synchronise balanced slots for the *active empty scout pool* only."""

        active = tuple(sorted(workers, key=lambda item: item.id.bytes))
        active_ids = {worker.id for worker in active}
        changed = False
        for worker_id, state in tuple(self.memory.worker_scout_states.items()):
            if worker_id in active_ids:
                continue
            if state.scout_eligible or state.target is not None:
                self.memory.worker_scout_states[worker_id] = replace(
                    state,
                    scout_eligible=False,
                    target=None,
                    best_route_cost=None,
                    stalled_ticks=0,
                    reachable_candidates=0,
                )
                changed = True
            self.memory.scout_return_route_leases.pop(worker_id, None)
            mission = self.memory.unit_missions.get(worker_id)
            if mission is not None and mission.mission in {
                UnitMission.EXPLORE,
                UnitMission.RETURN_TO_SCOUT_BAND,
            }:
                self.memory.unit_missions.pop(worker_id, None)

        desired_slots = set(range(len(active)))
        retained: dict[UUID, int] = {}
        used_slots: set[int] = set()
        for worker in active:
            existing = self.memory.worker_scout_states.get(worker.id)
            if (
                existing is not None
                and existing.coverage_version >= 1
                and existing.slot in desired_slots
                and existing.slot not in used_slots
            ):
                retained[worker.id] = existing.slot
                used_slots.add(existing.slot)

        available_slots = sorted(desired_slots - used_slots)
        band_count = len(self.config.exploration_sector_radii)
        max_radius = self.config.exploration_sector_radii[-1]
        for worker in active:
            existing = self.memory.worker_scout_states.get(worker.id)
            slot = retained.get(worker.id)
            if slot is None:
                if existing is not None and available_slots:
                    slot = min(
                        available_slots,
                        key=lambda candidate: (
                            int(candidate % self.config.exploration_sector_count != existing.sector_index),
                            abs(candidate % band_count - existing.stage),
                            candidate,
                        ),
                    )
                    available_slots.remove(slot)
                else:
                    slot = available_slots.pop(0)
                changed = True
            sector_index = slot % self.config.exploration_sector_count
            stage = slot % band_count
            mission = self.memory.unit_missions.get(worker.id)
            target = (
                mission.target
                if mission is not None and mission.mission is UnitMission.EXPLORE
                else None if existing is None else existing.target
            )
            slot_changed = bool(
                existing is None
                or existing.slot != slot
                or existing.sector_index != sector_index
                or existing.stage != stage
                or existing.coverage_version < 1
                or not existing.scout_eligible
            )
            if (
                slot_changed
                or target is not None
                and manhattan(target, core_position) > max_radius
                or target is not None
                and scout_sector_index(core_position, target) != sector_index
                and (
                    existing is None
                    or existing.phase is not WorkerScoutPhase.RETURN_TO_BAND
                )
            ):
                target = None
            phase = (
                WorkerScoutPhase.RETURN_TO_BAND
                if manhattan(worker.position, core_position) > max_radius
                else (
                    WorkerScoutPhase.SECTOR_SCOUT
                    if existing is None
                    else existing.phase
                )
            )
            if existing is None:
                self.memory.worker_scout_states[worker.id] = WorkerScoutState(
                    worker_id=worker.id,
                    slot=slot,
                    sector_index=sector_index,
                    stage=stage,
                    phase=phase,
                    target=target,
                    assigned_tick=tick if mission is None else mission.assigned_tick,
                    scout_eligible=True,
                    coverage_version=1,
                    lease_until=tick + self.config.exploration_assignment_lease_ticks,
                )
            else:
                self.memory.worker_scout_states[worker.id] = replace(
                    existing,
                    slot=slot,
                    sector_index=sector_index,
                    stage=stage,
                    phase=phase,
                    target=target,
                    assigned_tick=(
                        tick if target is None else existing.assigned_tick
                    ),
                    best_route_cost=(
                        None if target is None else existing.best_route_cost
                    ),
                    stalled_ticks=(0 if target is None else existing.stalled_ticks),
                    scout_eligible=True,
                    coverage_version=1,
                    lease_until=(
                        tick + self.config.exploration_assignment_lease_ticks
                        if slot_changed
                        else max(existing.lease_until, tick)
                    ),
                )
            if slot_changed:
                self.memory.scout_return_route_leases.pop(worker.id, None)
        if changed:
            self.memory.scout_assignment_last_rebalance_tick = tick

    @staticmethod
    def _clear_scout_target(
        state: WorkerScoutState,
        tick: int,
        *,
        advance: bool,
        backoff_until: int = 0,
    ) -> WorkerScoutState:
        return replace(
            state,
            # Stable scout slots rotate targets within one band.  Reaching a
            # target never promotes the Worker to a larger radius.
            stage=state.stage,
            target=None,
            assigned_tick=tick,
            best_route_cost=None,
            stalled_ticks=0,
            backoff_until=backoff_until,
            reachable_candidates=0,
        )

    def _record_scout_progress(
        self,
        state: WorkerScoutState,
        worker: EntitySnapshot,
        route: Route,
        tick: int,
    ) -> WorkerScoutState:
        history = self.memory.position_history.get(worker.id, ())
        position_changed = len(history) < 2 or history[-2] != worker.position
        improved = (
            state.best_route_cost is None
            or route.distance < state.best_route_cost
        )
        stalled = 0 if improved else state.stalled_ticks + int(not position_changed or not improved)
        return replace(
            state,
            best_route_cost=(
                route.distance
                if state.best_route_cost is None
                else min(state.best_route_cost, route.distance)
            ),
            stalled_ticks=stalled,
            last_scan_tick=state.last_scan_tick,
        )

    def _commit_scout_assignment(
        self,
        world: WorldModel,
        worker: EntitySnapshot,
        state: WorkerScoutState,
        route: Route,
        claimed: set[Position],
        assignments: dict[UUID, tuple[Position, Route, int]],
        last_visible: dict[Position, int],
        *,
        gain: int | None = None,
    ) -> None:
        if state.target is None:
            return
        target_visible = scout_visible_cells(
            state.target,
            world.known_obstacles,
        )
        claimed_visible: set[Position] = set()
        for post in claimed:
            claimed_visible.update(
                scout_visible_cells(post, world.known_obstacles)
            )
        state = replace(
            state,
            visible_gain=len(target_visible - claimed_visible),
            overlap_cells=len(target_visible & claimed_visible),
        )
        claimed.add(state.target)
        self.memory.worker_scout_states[worker.id] = state
        mission = (
            UnitMission.RETURN_TO_SCOUT_BAND
            if state.phase is WorkerScoutPhase.RETURN_TO_BAND
            else UnitMission.EXPLORE
        )
        self.memory.unit_missions[worker.id] = MissionState(
            mission,
            state.target,
            state.assigned_tick,
            failures=state.stalled_ticks,
        )
        assignments[worker.id] = (
            state.target,
            route,
            information_gain(
                state.target,
                tick=world.tick,
                last_visible=last_visible,
                refresh_ticks=self.config.exploration_refresh_ticks,
            )
            if gain is None
            else gain,
        )

    def _sector_scout_assignment(
        self,
        world,
        projection,
        worker,
        service,
        state,
        claimed,
        blocked,
        costs,
        home_alert,
    ):
        assert world.core is not None
        # One nominal layer is probed per Tick.  Candidate failures within
        # that layer are retried immediately; if the whole ray is unavailable
        # LOCAL_DISPERSAL still gives the Worker a real task and the next Tick
        # gets the fair precision-scan opportunity.  Scanning three empty
        # 60-cell rays per Worker only multiplied latency without improving
        # the chosen first step.
        for stage_offset in range(1):
            stage = state.stage % len(self.config.exploration_sector_radii)
            radius = self.config.exploration_sector_radii[stage]
            candidates = sector_scout_candidates(
                world,
                world.core.destination or world.core.position,
                sector_index=state.sector_index,
                radius=radius,
                tick=world.tick,
                refresh_ticks=self.config.exploration_refresh_ticks,
                limit=min(12, self.config.exploration_candidate_limit),
                backoff=frozenset(self.memory.target_backoff_until),
                claimed=frozenset(claimed),
                max_radius=self.config.exploration_sector_radii[-1],
            )
            for candidate in candidates[:3]:
                use_local_segment = (
                    home_alert
                    or manhattan(worker.position, candidate)
                    > self.config.exploration_search_radius
                )
                route = (
                    self._alert_exploration_step(
                        world, projection, worker, candidate, blocked
                    )
                    if use_local_segment
                    else weighted_route_to(
                        world,
                        worker.position,
                        candidate,
                        node_limit=min(
                            self.config.path_node_limit,
                            max(256, self.config.exploration_search_radius * 16),
                        ),
                        blocked=blocked,
                        cell_costs=costs,
                        allow_unknown_endpoint=True,
                    )
                )
                if route is None or route.first_direction is None:
                    continue
                return (
                    replace(
                        state,
                        stage=stage,
                        phase=(
                            WorkerScoutPhase.RETURN_TO_BAND
                            if state.phase is WorkerScoutPhase.RETURN_TO_BAND
                            else WorkerScoutPhase.SECTOR_SCOUT
                        ),
                        target=candidate,
                        assigned_tick=world.tick,
                        best_route_cost=route.distance,
                        stalled_ticks=0,
                        backoff_until=0,
                        reachable_candidates=1,
                    ),
                    route,
                )
        return None

    def _return_to_scout_band_assignment(
        self,
        world: WorldModel,
        projection: TacticalMap,
        worker: EntitySnapshot,
        state: WorkerScoutState,
        claimed: set[Position],
        blocked: frozenset[Position],
        costs: dict[Position, int],
    ):
        """Choose a proved risk-weighted route back to the assigned ring."""

        assert world.core is not None
        radius = self.config.exploration_sector_radii[state.stage]
        core = world.core.destination or world.core.position
        max_radius = self.config.exploration_sector_radii[-1]
        previous = self.memory.scout_return_route_leases.get(worker.id)
        effective_blocked = set(blocked)
        if (
            previous is not None
            and previous.blocked_edge is not None
            and world.tick <= previous.backoff_until
            and worker.position == previous.blocked_edge[0]
        ):
            effective_blocked.add(previous.blocked_edge[1])
        effective_blocked.discard(worker.position)

        candidates: list[Position] = []
        if (
            state.target is not None
            and state.target in world.known_passable
            and state.target not in effective_blocked
            and state.target not in claimed
            and manhattan(core, state.target) <= max_radius
            and abs(manhattan(core, state.target) - radius) <= 3
            and self.memory.target_backoff_until.get(state.target, -1) < world.tick
        ):
            candidates.append(state.target)
        sector_order = [state.sector_index]
        for offset in range(1, self.config.exploration_sector_count // 2 + 1):
            sector_order.extend(
                (
                    (state.sector_index + offset)
                    % self.config.exploration_sector_count,
                    (state.sector_index - offset)
                    % self.config.exploration_sector_count,
                )
            )
        for sector_index in sector_order:
            rows = sector_scout_candidates(
                world,
                core,
                sector_index=sector_index,
                radius=radius,
                tick=world.tick,
                refresh_ticks=self.config.exploration_refresh_ticks,
                limit=min(16, self.config.exploration_candidate_limit),
                backoff=frozenset(self.memory.target_backoff_until),
                claimed=frozenset(claimed),
                max_radius=max_radius,
            )
            candidates.extend(cell for cell in rows if cell not in candidates)
            if len(candidates) >= 16:
                break
        rows = []
        sticky_target_ready = bool(
            state.target is not None
            and candidates
            and candidates[0] == state.target
            and (
                previous is None
                or previous.stalled_ticks
                < self.config.exploration_return_stall_ticks
            )
        )
        candidate_scan_limit = 1 if sticky_target_ready else 6
        for target in candidates[:candidate_scan_limit]:
            full_route = weighted_route_to(
                world,
                worker.position,
                target,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(effective_blocked),
                cell_costs=costs,
            )
            waypoint = target
            segmented = False
            route = full_route
            if route is None:
                progress = weighted_progress_route(
                    world,
                    worker.position,
                    target,
                    node_limit=self.config.path_node_limit,
                    blocked=frozenset(effective_blocked),
                    cell_costs=costs,
                )
                if progress is None:
                    continue
                route, waypoint = progress
                segmented = True
            if route.first_direction is None or route.first_position is None:
                continue
            viability = move_viability(
                world,
                worker.position,
                route.first_position,
                target=waypoint,
                blocked=frozenset(effective_blocked),
                node_limit=self.config.path_node_limit,
                require_continuation=True,
            )
            if not viability.viable:
                continue
            estimate = (
                route.distance + manhattan(waypoint, target)
                if segmented
                else route.distance
            )
            route = Route(
                route.distance,
                route.first_direction,
                route.first_position,
                viability,
            )
            rows.append(
                (
                    (
                        int(state.target is None or target != state.target),
                        int(segmented),
                        estimate,
                        self.memory.congestion_counts.get(target, 0),
                        target,
                    ),
                    target,
                    route,
                    waypoint,
                    estimate,
                )
            )
        if not rows:
            return None
        rows.sort(key=lambda row: row[0])
        _, target, route, waypoint, estimate = rows[0]
        stalled = 0
        route_version = 0 if previous is None else previous.route_version
        if previous is not None and previous.target == target:
            progressed = bool(
                previous.last_route_distance is None
                or estimate < previous.last_route_distance
            )
            stalled = 0 if progressed else previous.stalled_ticks + 1
            if (
                stalled >= self.config.exploration_return_stall_ticks
                and len(rows) > 1
            ):
                self.memory.target_backoff_until[target] = (
                    world.tick + self.config.exploration_return_loop_backoff_ticks
                )
                _, target, route, waypoint, estimate = rows[1]
                stalled = 0
                route_version += 1
        lease = ScoutReturnRouteLease(
            worker_id=worker.id,
            target=target,
            waypoint=waypoint,
            assigned_tick=(
                previous.assigned_tick
                if previous is not None and previous.target == target
                else world.tick
            ),
            last_position=worker.position,
            last_route_distance=estimate,
            stalled_ticks=stalled,
            route_version=route_version,
            blocked_edge=(None if previous is None else previous.blocked_edge),
            backoff_until=(0 if previous is None else previous.backoff_until),
        )
        self.memory.scout_return_route_leases[worker.id] = lease
        return (
            replace(
                state,
                phase=WorkerScoutPhase.RETURN_TO_BAND,
                target=target,
                assigned_tick=lease.assigned_tick,
                best_route_cost=estimate,
                stalled_ticks=stalled,
                backoff_until=0,
                reachable_candidates=len(rows),
            ),
            route,
        )

    def _local_scout_assignment(
        self,
        world,
        projection,
        worker,
        service,
        state,
        claimed,
    ):
        blocked, _ = self._exploration_navigation(
            world, projection, worker, service
        )
        previous = None
        history = self.memory.position_history.get(worker.id, ())
        if len(history) >= 2:
            previous = history[-2]
        last_visible = dict(world.cell_last_visible)
        options = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(worker.position)
        ):
            if (
                destination in blocked
                or destination in world.known_obstacles
                or destination in claimed
                or destination not in world.known_passable
                or (
                    world.core is not None
                    and manhattan(destination, world.core.position)
                    > self.config.exploration_sector_radii[-1]
                )
            ):
                continue
            immediate, future, remembered = projection.worker_exposure(destination)
            if immediate >= worker.hp:
                continue
            viability = move_viability(
                world,
                worker.position,
                destination,
                blocked=blocked,
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            )
            if not viability.viable:
                continue
            gain = information_gain(
                destination,
                tick=world.tick,
                last_visible=last_visible,
                refresh_ticks=self.config.exploration_refresh_ticks,
            )
            score = (
                immediate,
                future,
                remembered,
                int(
                    world.core is not None
                    and scout_sector_index(world.core.position, destination)
                    != state.sector_index
                ),
                -gain,
                int(destination == previous),
                self.memory.congestion_counts.get(destination, 0),
                self.memory.visit_counts.get(destination, 0),
                index,
            )
            options.append((score, direction, destination, viability))
        if not options:
            return None
        _, direction, target, viability = min(options, key=lambda row: row[0])
        return (
            replace(
                state,
                phase=WorkerScoutPhase.LOCAL_DISPERSAL,
                target=target,
                assigned_tick=world.tick,
                best_route_cost=1,
                stalled_ticks=0,
                backoff_until=0,
                reachable_candidates=len(options),
            ),
            Route(1, direction, target, viability),
        )

    def _exploration_wait_reason(self, world, projection, worker, service):
        protected = {
            cell
            for cell in (
                service.service_core_position,
                service.entrance,
                service.exit_cell,
                *service.queue_cells,
            )
            if cell is not None and cell != worker.position
        }
        terrain_legal = [
            cell
            for _, cell in cardinal_neighbors(worker.position)
            if cell not in world.known_obstacles
            and cell not in projection.hostile_occupied
            and cell not in protected
        ]
        if not terrain_legal:
            return "ALL_SCOUT_TARGETS_BLOCKED"
        if all(projection.immediate_attackers(cell) >= worker.hp for cell in terrain_legal):
            return "NO_SURVIVABLE_MOVE"
        if all(
            not move_viability(
                world,
                worker.position,
                cell,
                blocked=frozenset(
                    set(projection.hostile_occupied)
                    | protected
                    | set(projection.immediate_damage)
                ),
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            ).viable
            for cell in terrain_legal
        ):
            return "NO_VIABLE_CONTINUATION"
        return "NO_REACHABLE_FRONTIER"

    def _home_alert(self, world: WorldModel, projection: TacticalMap) -> bool:
        if world.core is None:
            return False
        return self.memory.home_defense_alert_until >= world.tick or any(
            enemy.visible_now
            and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.observed_position, world.core.position)
            <= self.config.home_engage_radius + 4
            for enemy in projection.enemies
        )

    def _alert_exploration_step(
        self,
        world: WorldModel,
        projection: TacticalMap,
        worker: EntitySnapshot,
        target: Position,
        blocked: frozenset[Position],
    ) -> Route | None:
        if worker.position == target:
            return Route(0, None, None)
        options: list[
            tuple[tuple[int, ...], Direction, Position, MoveViability]
        ] = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(worker.position)
        ):
            if (
                destination in world.known_obstacles
                or destination in blocked
                or (
                    destination not in world.known_passable
                    and destination != target
                )
            ):
                continue
            viability = move_viability(
                world,
                worker.position,
                destination,
                target=target,
                blocked=blocked,
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            )
            if not viability.viable:
                continue
            immediate, future, remembered = projection.worker_exposure(destination)
            options.append(
                (
                    (
                        immediate,
                        future,
                        remembered,
                        manhattan(destination, target),
                        self.memory.congestion_counts.get(destination, 0),
                        index,
                    ),
                    direction,
                    destination,
                    viability,
                )
            )
        if not options:
            return None
        _, direction, destination, viability = min(
            options, key=lambda row: row[0]
        )
        return Route(
            manhattan(worker.position, target),
            direction,
            destination,
            viability,
        )

    def _exploration_navigation(self, world, projection, actor, service):
        protected = {
            cell
            for cell in (
                world.core.position if world.core is not None else None,
                service.service_core_position,
                service.entrance,
                service.exit_cell,
                *service.queue_cells,
            )
            if cell is not None
        }
        control_blocked, costs = self.safety.navigation_layers(
            projection,
            tuple(self.memory.enemy_core_control_zones.values()),
        )
        blocked = set(projection.hostile_occupied)
        blocked.update(control_blocked)
        # Once a Worker is outside a currently hard enemy-Core control zone,
        # returning/exploring may route around it but may not cut back through
        # the radius-8 clearing belt.  Workers already inside keep the normal
        # radius-6 hard block so their disengage planner can lead them out.
        for zone in self.memory.enemy_core_control_zones.values():
            if (
                zone.control_level is EnemyCoreControlLevel.HARD
                and manhattan(actor.position, zone.center) > zone.clear_radius
            ):
                blocked.update(diamond(zone.center, zone.clear_radius))
        blocked.update(protected)
        blocked.update(projection.immediate_damage)
        blocked.discard(actor.position)
        if (
            world.core is not None
            and actor.position == world.core.position
            and service.exit_cell is not None
        ):
            blocked.discard(service.exit_cell)
        return frozenset(blocked), costs

    def _route(
        self,
        world,
        projection,
        actor,
        target,
        service,
        *,
        logistics=False,
        allow_directional_fallback=False,
        allow_unknown=False,
        extra_blocked=frozenset(),
    ):
        protected = {
            cell
            for cell in (
                world.core.position if world.core is not None else None,
                service.entrance,
                service.exit_cell,
                *service.queue_cells,
            )
            if cell is not None
        }
        control_blocked, costs = self.safety.navigation_layers(
            projection,
            tuple(self.memory.enemy_core_control_zones.values()),
        )
        blocked = set(projection.hostile_occupied)
        blocked.update(control_blocked)
        blocked.update(extra_blocked)
        if not logistics:
            blocked.update(protected - {actor.position, target})
        # Ordinary Worker tasks never volunteer for a current firing cell.
        # The escape planner is the sole place allowed to spend its explicit
        # non-fatal-hit budget after two-step dead-end analysis.
        blocked.update(projection.immediate_damage)
        blocked.discard(actor.position)
        route = weighted_route_to(
            world,
            actor.position,
            target,
            node_limit=self.config.path_node_limit,
            blocked=frozenset(blocked),
            cell_costs=costs,
            allow_unknown_endpoint=allow_unknown,
        )
        return route

    def _looping(self, unit_id):
        history = self.memory.position_history.get(unit_id, ())
        for period in range(
            1,
            min(
                self.config.worker_escape_max_loop_period,
                len(history) // self.config.loop_repeat_limit,
            )
            + 1,
        ):
            required = period * self.config.loop_repeat_limit
            if len(history) < required:
                continue
            pattern = history[-period:]
            window = history[-required:]
            if len(set(window)) <= 1:
                continue
            if all(
                window[index * period : (index + 1) * period] == pattern
                for index in range(self.config.loop_repeat_limit)
            ):
                return True
        return False

    def _manual_allowed(self, unit_id, direction, tick):
        lease = self.memory.manual_move_leases.get(unit_id)
        if lease is None or tick > lease.expires_tick:
            return True
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }[lease.direction]
        return direction is not opposite

    @staticmethod
    def _risk(projection, cell):
        if cell is None:
            return 0
        immediate, future, remembered = projection.worker_exposure(cell)
        return immediate * 100 + future * 10 + remembered
