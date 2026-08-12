from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from arena_hero import CoreState, Direction, Position, UnitType

from .config import TacticConfig
from .geometry import cardinal_neighbors, manhattan, manhattan_ring
from .models import (
    ActionIntent,
    CoreServiceQueue,
    EntitySnapshot,
    IntentAction,
    MissionState,
    UnitMission,
    WorkerEscapeState,
    WorkerScoutPhase,
    WorkerScoutState,
    WorkerTaskProgress,
    WorldModel,
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
    weighted_route_to,
)
from .projection import TacticalMap
from .resource_allocator import ResourceAllocator
from .rules import UNIT_MAX_HP
from .state import TacticMemory
from .worker_safety import WorkerSafetyEvaluator


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

    def intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
    ) -> list[ActionIntent]:
        if world.core is None:
            return []
        workers = tuple(
            unit for unit in world.friendlies if unit.unit_type is UnitType.WORKER
        )
        self._ensure_scout_states(workers, world.tick)
        intents, escaping = self._survival_intents(world, projection, service, workers)
        guarding: dict[UUID, Position] = {}
        if self.memory.storage_saturated:
            service_ids = set(service.depositors)
            guard_workers = tuple(
                worker
                for worker in workers
                if worker.id not in escaping
                and worker.hp >= UNIT_MAX_HP[UnitType.WORKER]
                and worker.id != world.beacon.carrier_id
                and worker.id not in service_ids
            )
            guarding = self._home_guard_assignments(
                world,
                projection,
                service,
                guard_workers,
            )
            resource_assignments: dict[UUID, Position] = {}
            exploration_assignments: dict[UUID, tuple[Position, Route, int]] = {}
        else:
            self.memory.worker_home_guard_targets.clear()
            available = tuple(
                worker
                for worker in workers
                if worker.id not in escaping
                and worker.hp >= UNIT_MAX_HP[UnitType.WORKER]
                and worker.cargo == 0
                and worker.id != world.beacon.carrier_id
            )
            resource_assignments, exploration_assignments = self._assign_work(
                world, projection, service, available
            )

        for worker in workers:
            if worker.id in escaping:
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
        """Spread saturated-economy Workers near home without blocking it."""

        assert world.core is not None
        living = {worker.id for worker in workers}
        for worker_id in tuple(self.memory.worker_home_guard_targets):
            if worker_id not in living:
                self.memory.worker_home_guard_targets.pop(worker_id, None)
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
        combat_guard = (
            self.memory.home_defense_alert_until >= world.tick
            or any(
                enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                and manhattan(enemy.position, core) <= self.config.home_warning_radius
                for enemy in world.enemies
            )
        )
        guard_radii = (
            self.config.worker_full_storage_combat_guard_radii
            if combat_guard
            else self.config.worker_full_storage_guard_radii
        )
        candidates = tuple(
            cell
            for radius in guard_radii
            for cell in manhattan_ring(core, radius)
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in projection.hostile_occupied
            and cell not in service_cells
            and cell not in combat_positions
            and projection.immediate_attackers(cell) == 0
        )
        if len(candidates) < len(workers):
            # Dense or heavily obstructed homes may not have enough legal posts
            # on the preferred rings.  Expand only as far as needed, skipping
            # the combat patrol rings so Workers remain near home without
            # becoming part of the fighting formation.
            preferred = set(guard_radii)
            combat_rings = set(self.config.peaceful_squad_radii)
            expanded = list(candidates)
            seen = set(expanded)
            outer = max(guard_radii)
            for radius in range(outer + 1, outer + 9):
                if radius in preferred or radius in combat_rings:
                    continue
                for cell in manhattan_ring(core, radius):
                    if (
                        cell in seen
                        or cell not in world.known_passable
                        or cell in world.known_obstacles
                        or cell in projection.hostile_occupied
                        or cell in service_cells
                        or cell in combat_positions
                        or projection.immediate_attackers(cell) > 0
                    ):
                        continue
                    seen.add(cell)
                    expanded.append(cell)
                if len(expanded) >= len(workers):
                    break
            candidates = tuple(expanded)
        if not candidates:
            self.memory.worker_home_guard_targets.clear()
            return {}

        ordered_workers = tuple(sorted(workers, key=lambda unit: unit.id.bytes))
        used: set[Position] = set()
        assignments: dict[UUID, Position] = {}
        candidate_set = set(candidates)
        for worker in ordered_workers:
            previous = self.memory.worker_home_guard_targets.get(worker.id)
            if previous in candidate_set and previous not in used:
                assignments[worker.id] = previous
                used.add(previous)

        count = len(ordered_workers)
        for rank, worker in enumerate(ordered_workers):
            if worker.id in assignments:
                continue
            desired = rank * len(candidates) // count

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

            available = (
                (index, cell)
                for index, cell in enumerate(candidates)
                if cell not in used
            )
            _, target = min(available, key=score, default=(0, worker.position))
            assignments[worker.id] = target
            used.add(target)

        self.memory.worker_home_guard_targets = dict(assignments)
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
            UnitMission.HOME_GUARD,
            target,
            world.tick,
        )
        if worker.position == target:
            return [
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.HOME_GUARD,
                    68,
                    target_position=target,
                    reason="FULL_STORAGE_HOME_GUARD_HOLD",
                    metadata=(("guard_post", target),),
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
        if (
            route is not None
            and route.first_direction is not None
            and self._manual_allowed(worker.id, route.first_direction, world.tick)
        ):
            intents = [
                ActionIntent.move(
                    worker.id,
                    UnitMission.HOME_GUARD,
                    68,
                    route.first_direction,
                    route.first_position,
                    risk=self._risk(projection, route.first_position),
                    exclusive_destination=True,
                    tie_break=(route.distance,),
                    reason="FULL_STORAGE_RETURN_TO_HOME_GUARD",
                    metadata=(
                        ("guard_post", target),
                        ("allow_protected", True),
                    ),
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
                )
            )
        preferred_step = None if route is None else route.first_position
        for score, direction, destination in sorted(options)[:4]:
            if destination == preferred_step:
                continue
            intents.append(
                ActionIntent.move(
                    worker.id,
                    UnitMission.HOME_GUARD,
                    69,
                    direction,
                    destination,
                    risk=score[0] * 100 + score[1] * 10 + score[2],
                    exclusive_destination=True,
                    tie_break=score,
                    reason="FULL_STORAGE_HOME_GUARD_FALLBACK",
                    metadata=(
                        ("guard_post", target),
                        ("allow_protected", destination in service_cells),
                    ),
                )
            )
        intents.append(
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.HOME_GUARD,
                71,
                target_position=target,
                reason=(
                    "HOME_GUARD_CONGESTION_HOLD"
                    if options
                    else "HOME_GUARD_NO_SAFE_STEP"
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
            if projection.immediate_attackers(worker.position) >= worker.hp:
                escape = self._update_escape(world, projection, worker)
                if escape is not None:
                    escaping.add(worker.id)
                    intents.extend(
                        self._escape_intents(world, projection, worker, escape)
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
            worker for worker in available if worker.position != world.core.position
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
        resources = self.resources.allocate(
            world,
            projection,
            resource_workers,
            hard_blocked=service_blocks,
        ).as_dict()
        explorers = tuple(
            worker for worker in available if worker.id not in resources
        )
        exploration = self._exploration_assignments(
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
        if worker.position in world.visible_resources:
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
        improved = bool(
            previous_progress is None
            or previous_progress.target != resource
            or route_distance is not None
            and (
                previous_progress.route_distance is None
                or route_distance < previous_progress.route_distance
            )
        )
        stalled = 0 if improved else previous_progress.stalled_ticks + 1
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
            self.memory.unit_missions.pop(worker.id, None)
            self.memory.worker_task_progress[worker.id] = progress
            return self._resource_stall_fallback(
                world,
                projection,
                service,
                worker,
                resource,
            )
        self.memory.worker_task_progress[worker.id] = progress
        self.memory.unit_missions[worker.id] = MissionState(
            UnitMission.HARVEST,
            resource,
            world.tick,
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
        intents = [
            ActionIntent.move(
                worker.id,
                UnitMission.EXPLORE,
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
                UnitMission.EXPLORE,
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
        intents = [
            ActionIntent.move(
                worker.id,
                UnitMission.EXPLORE,
                70,
                route.first_direction,
                route.first_position,
                risk=self._risk(projection, route.first_position),
                tie_break=(-gain, route.distance),
                reason="INFORMATION_GAIN",
                metadata=(("information_gain", gain), ("goal", target))
                + (() if route.viability is None else route.viability.metadata),
            ),
        ]
        blocked, _ = self._exploration_navigation(
            world,
            projection,
            worker,
            service,
        )
        previous = None
        history = self.memory.position_history.get(worker.id, ())
        if len(history) >= 2:
            previous = history[-2]
        alternatives = []
        require_continuation = (
            manhattan(worker.position, target)
            <= self.config.exploration_search_radius
        )
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
            viability = move_viability(
                world,
                worker.position,
                destination,
                target=target,
                blocked=blocked,
                node_limit=min(self.config.path_node_limit, 512),
                require_continuation=require_continuation,
                require_open_area=not require_continuation,
            )
            if not viability.viable:
                continue
            immediate, future, remembered = projection.worker_exposure(destination)
            score = (
                immediate,
                future,
                remembered,
                manhattan(destination, target),
                int(destination == previous),
                self.memory.congestion_counts.get(destination, 0),
                index,
            )
            alternatives.append((score, direction, destination, viability))
        intents.extend(
            ActionIntent.move(
                worker.id,
                UnitMission.EXPLORE,
                71,
                direction,
                destination,
                risk=score[0] * 100 + score[1] * 10 + score[2],
                tie_break=score,
                reason="EXPLORATION_ALTERNATE_STEP",
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
                UnitMission.EXPLORE,
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
                <= self.config.worker_escape_trigger_radius
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
        if threats:
            retained = {
                threat_id
                for threat_id in (() if previous is None else previous.threat_ids)
                if (enemy := projection.enemy(threat_id)) is not None
                and enemy.age <= self.config.enemy_track_ttl
            }
            retained.update(threats)
            state = WorkerEscapeState(
                "FLEEING" if direct_threats else "GLOBAL_ALERT_RETREAT",
                tuple(sorted(retained, key=lambda item: item.bytes)),
                world.tick,
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
            state = WorkerEscapeState(
                "FOG_RETREAT",
                fresh,
                previous.last_threat_tick,
                0,
            )
            self.memory.worker_escape_states[worker.id] = state
            return state
        exits = self._safe_exits(world, projection, worker.position, worker.hp)
        safe_ticks = previous.safe_ticks + 1 if exits >= 2 else 0
        if safe_ticks >= self.config.worker_escape_safe_ticks:
            self.memory.worker_escape_states.pop(worker.id, None)
            return None
        state = WorkerEscapeState(
            "CLEARING",
            previous.threat_ids,
            previous.last_threat_tick,
            safe_ticks,
        )
        self.memory.worker_escape_states[worker.id] = state
        return state

    def _escape_intents(self, world, projection, worker, state):
        assert world.core is not None
        rows: list[tuple[tuple[int, ...], Direction, Position]] = []
        history = self.memory.position_history.get(worker.id, ())
        previous = (
            history[-2]
            if len(history) >= 2
            else None
        )
        recent_positions = frozenset(history[-4:-1])
        for index, (direction, destination) in enumerate(cardinal_neighbors(worker.position)):
            if destination in world.known_obstacles or destination in projection.hostile_occupied:
                continue
            immediate = projection.immediate_attackers(destination)
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
            )
            horizon = self._survival_horizon(
                world,
                projection,
                destination,
                worker.hp,
                origin=worker.position,
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
            if not viability.viable or horizon == 0:
                continue
            recent = self._recent_threat(projection, destination, state.threat_ids)
            heat = projection.worker_exposure(destination)[2]
            minimum, total = self._enemy_distances(
                projection, destination, state.threat_ids
            )
            score = (
                int(horizon == 0),
                int(exits == 0),
                immediate,
                projection.future_attackers(destination),
                recent,
                heat,
                -minimum,
                -total,
                manhattan(destination, world.core.destination or world.core.position),
                int(destination == previous),
                index,
            )
            rows.append((score, direction, destination, viability))
        # A fresh threat lease must produce spatial progress.  Merely making
        # the previous cell look slightly cooler caused the live A-B-A-B
        # oscillation: every fogged Tick undid the preceding escape step.  A
        # recent cell remains available only when every novel option has no
        # survivable two-step continuation.
        novel_survivable = [
            row
            for row in rows
            if row[2] not in recent_positions and row[0][0] == 0
        ]
        if novel_survivable:
            rows = [row for row in rows if row[2] not in recent_positions]
        elif previous is not None:
            non_backtracking_survivable = [
                row
                for row in rows
                if row[2] != previous and row[0][0] == 0
            ]
            if non_backtracking_survivable:
                rows = [row for row in rows if row[2] != previous]
        if state.phase != "FLEEING":
            current_home_distance = manhattan(
                worker.position,
                world.core.destination or world.core.position,
            )
            homeward_survivable = [
                row
                for row in rows
                if row[0][0] == 0
                and row[0][1] == 0
                and row[0][2] == 0
                and row[0][3] == 0
                and manhattan(
                    row[2],
                    world.core.destination or world.core.position,
                )
                <= current_home_distance
            ]
            if homeward_survivable:
                rows = homeward_survivable
        intents = [
            ActionIntent.move(
                worker.id,
                UnitMission.ESCAPE,
                20,
                direction,
                destination,
                risk=score[2] * 100 + score[3] * 10 + score[4] + score[5],
                tie_break=score,
                reason=state.phase,
                metadata=(
                    ("escape_phase", state.phase),
                    ("safe_horizon", -score[0]),
                    ("first_step_heat", score[5]),
                ) + viability.metadata,
            )
            for score, direction, destination, viability in sorted(
                rows, key=lambda row: row[0]
            )
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

    def _safe_exits(self, world, projection, position, hp, *, origin=None):
        return self.safety.forward_safe_exits(
            world,
            projection,
            position,
            hp,
            origin=origin,
        )

    def _survival_horizon(self, world, projection, start, hp, *, origin=None):
        return self.safety.survival_terminals(
            world,
            projection,
            start,
            hp,
            origin=origin,
            node_limit=self.config.worker_escape_lookahead_nodes,
        )

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
                    reason="NO_RETURN_ROUTE",
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
            return [
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.RETURN_CARGO,
                    51,
                    target_position=worker.position,
                    reason="WAIT_FOR_DEPARTURE_TICK",
                    metadata=(
                        ("scheduled_deposit_tick", reservation.scheduled_deposit_tick),
                        ("departure_tick", reservation.departure_tick),
                        ("slack_ticks", reservation.slack_ticks),
                    ),
                )
            ]
        if reservation.first_direction is not None and reservation.first_position is not None:
            route_target = reservation.route_target or world.core.position
            terminal_exception = (
                "CORE_SERVICE"
                if reservation.first_position == world.core.position
                else None
            )
            viability = move_viability(
                world,
                worker.position,
                reservation.first_position,
                target=route_target,
                node_limit=min(self.config.path_node_limit, 512),
                require_continuation=(
                    terminal_exception is None
                    and reservation.first_position != route_target
                ),
                require_open_area=terminal_exception is None,
                terminal_exception=terminal_exception,
            )
            if not viability.viable:
                return [
                    ActionIntent.simple(
                        worker.id,
                        IntentAction.WAIT,
                        UnitMission.RETURN_CARGO,
                        49 if service.admission_id == worker.id else 51,
                        reason="NO_RETURN_ROUTE",
                        metadata=viability.metadata,
                    )
                ]
            ready = worker.id in service.ready_depositors
            priority = 49 if service.admission_id == worker.id else (50 if ready else 51)
            reason = "SERVICE_ADMISSION" if reservation.first_position == world.core.position else (
                "SERVICE_PIPELINE_ADVANCE" if ready else "SERVICE_QUEUE_APPROACH"
            )
            return [
                ActionIntent.move(
                    worker.id,
                    UnitMission.RETURN_CARGO,
                    priority,
                    reservation.first_direction,
                    reservation.first_position,
                    risk=self._risk(projection, reservation.first_position),
                    exclusive_destination=reservation.first_position == world.core.position,
                    tie_break=(reservation.route_distance,),
                    reason=reason,
                    metadata=(
                        ("allow_protected", True),
                        (
                            "allow_head_on_swap",
                            reservation.first_position == world.core.position
                            or worker.position == world.core.position,
                        ),
                        ("service_slot", reservation.route_target),
                        (
                            "allow_service_overlap",
                            reservation.first_position in service.queue_cells,
                        ),
                        ("scheduled_deposit_tick", reservation.scheduled_deposit_tick),
                        ("departure_tick", reservation.departure_tick),
                    ) + viability.metadata,
                ),
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.RETURN_CARGO,
                    priority + 1,
                    reason=(
                        "WAITING_FOR_CORE_SLOT"
                        if service.admission_id == worker.id
                        else "WAITING_FOR_SERVICE_SLOT"
                    ),
                ),
            ]
        return [
            ActionIntent.simple(
                worker.id,
                IntentAction.WAIT,
                UnitMission.RETURN_CARGO,
                49 if service.admission_id == worker.id else 51,
                reason="WAITING_FOR_DEPOSIT_ACTION",
            )
        ]

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
                target=scout_target,
                blocked=frozenset(protected - {destination}),
                node_limit=min(self.config.path_node_limit, 256),
                require_continuation=(
                    scout_target is not None and destination != scout_target
                ),
                require_open_area=scout_target is None,
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

    def _exploration_assignments(self, world, projection, workers, service):
        assignments: dict[UUID, tuple[Position, Route, int]] = {}
        claimed: set[Position] = set()
        last_visible = dict(world.cell_last_visible)
        home_alert = self._home_alert(world, projection)
        ordered = sorted(
            workers,
            key=lambda worker: (
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

            target_expired = (
                state.target is not None
                and world.tick - state.assigned_tick >= self.config.exploration_scout_hold_ticks
            )
            target_observed = (
                state.target is not None
                and last_visible.get(state.target) == world.tick
            )
            looping = self._looping(worker.id)
            if looping or target_expired or target_observed:
                backoff_until = 0
                if state.target is not None and (looping or target_expired):
                    backoff_until = (
                        world.tick + self.config.exploration_target_backoff_ticks
                    )
                    self.memory.target_backoff_until[state.target] = backoff_until
                state = self._clear_scout_target(
                    state,
                    world.tick,
                    advance=state.phase is WorkerScoutPhase.SECTOR_SCOUT,
                    backoff_until=backoff_until,
                )
                self.memory.unit_missions.pop(worker.id, None)

            blocked, costs = self._exploration_navigation(
                world,
                projection,
                worker,
                service,
            )
            if (
                state.target is not None
                and state.target not in claimed
                and self.memory.target_backoff_until.get(state.target, -1) < world.tick
            ):
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

            if not home_alert and scan_attempts < self.config.exploration_new_goal_budget:
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
                    if candidate in claimed:
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

    def _ensure_scout_states(self, workers, tick: int) -> None:
        used_slots: set[int] = set()
        next_slot = 0
        for worker in sorted(workers, key=lambda item: item.id.bytes):
            existing = self.memory.worker_scout_states.get(worker.id)
            if existing is not None and existing.slot not in used_slots:
                used_slots.add(existing.slot)
                continue
            while next_slot in used_slots:
                next_slot += 1
            mission = self.memory.unit_missions.get(worker.id)
            target = (
                mission.target
                if mission is not None and mission.mission is UnitMission.EXPLORE
                else None
            )
            if existing is None:
                self.memory.worker_scout_states[worker.id] = WorkerScoutState(
                    worker_id=worker.id,
                    slot=next_slot,
                    sector_index=next_slot % 8,
                    stage=next_slot // 8,
                    phase=WorkerScoutPhase.SECTOR_SCOUT,
                    target=target,
                    assigned_tick=tick if mission is None else mission.assigned_tick,
                )
            else:
                self.memory.worker_scout_states[worker.id] = replace(
                    existing,
                    slot=next_slot,
                    sector_index=next_slot % 8,
                    stage=max(existing.stage, next_slot // 8),
                    target=None,
                    assigned_tick=tick,
                    best_route_cost=None,
                    stalled_ticks=0,
                )
            used_slots.add(next_slot)

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
            stage=state.stage + int(advance),
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
        claimed.add(state.target)
        self.memory.worker_scout_states[worker.id] = state
        self.memory.unit_missions[worker.id] = MissionState(
            UnitMission.EXPLORE,
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
            stage = state.stage + stage_offset
            if stage < len(self.config.exploration_sector_radii):
                radius = self.config.exploration_sector_radii[stage]
            else:
                radius = (
                    self.config.exploration_sector_radii[-1]
                    + (stage - len(self.config.exploration_sector_radii) + 1)
                    * self.config.exploration_sector_step
                )
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
                        phase=WorkerScoutPhase.SECTOR_SCOUT,
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
            <= self.config.home_warning_radius
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
        blocked = set(projection.hostile_occupied)
        blocked.update(protected)
        blocked.update(projection.immediate_damage)
        blocked.discard(actor.position)
        if (
            world.core is not None
            and actor.position == world.core.position
            and service.exit_cell is not None
        ):
            blocked.discard(service.exit_cell)
        costs = self.safety.route_costs(projection)
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
        blocked = set(projection.hostile_occupied)
        blocked.update(extra_blocked)
        if not logistics:
            blocked.update(protected - {actor.position, target})
        # Ordinary Worker tasks never volunteer for a current firing cell.
        # The escape planner is the sole place allowed to spend its explicit
        # non-fatal-hit budget after two-step dead-end analysis.
        blocked.update(projection.immediate_damage)
        blocked.discard(actor.position)
        costs = self.safety.route_costs(projection)
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
        for period in range(1, len(history) // self.config.loop_repeat_limit + 1):
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
