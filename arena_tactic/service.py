from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable
from uuid import UUID

from arena_hero import CoreState, Direction, Position, UnitType

from .config import TacticConfig
from .geometry import DIRECTION_ORDER, add_direction, cardinal_neighbors, manhattan, manhattan_ring
from .models import (
    ActionIntent,
    CargoReturnReservation,
    CoreOperationRequest,
    CoreOperationTimeline,
    CoreServiceJob,
    CoreServicePhase,
    CoreSlotSchedule,
    CoreServiceWindow,
    CoreServiceQueue,
    EntitySnapshot,
    IntentAction,
    PatientQueueEntry,
    PatientAdmissionProgress,
    ServiceCellLease,
    UnitMission,
    WorldModel,
)
from .projection import TacticalMap
from .planning import route_to, weighted_distance_field, weighted_route_to
from .rules import UNIT_MAX_HP
from .state import TacticMemory


_SERVICE_MISSIONS = frozenset(
    {
        UnitMission.DEPOSIT,
        UnitMission.RETURN_CARGO,
        UnitMission.CLEAR_CORE,
        UnitMission.CLEAR_SERVICE_CELL,
        UnitMission.RECOVER,
    }
)


def _cardinal_direction(start: Position, destination: Position) -> Direction | None:
    return next(
        (
            direction
            for direction, neighbor in cardinal_neighbors(start)
            if neighbor == destination
        ),
        None,
    )


def cargo_return_route(
    world: WorldModel,
    projection: TacticalMap,
    carrier: EntitySnapshot,
    core_position: Position,
    queue_cells: tuple[Position, ...],
    *,
    node_limit: int,
    exit_cell: Position | None = None,
    direct_core: bool = False,
) -> CargoReturnReservation:
    """Build the one authoritative spatial route used by calendar and actor.

    Visible hostile occupancy and actual current firing cells are hard
    blockers.  Fogged tracks and durable threat heat remain weighted costs;
    they may cause a local detour but never an infinite directional ban.
    """

    if carrier.position == core_position:
        return CargoReturnReservation(
            worker_id=carrier.id,
            route_target=core_position,
            route_distance=0,
            first_direction=None,
            first_position=None,
            earliest_deposit_tick=world.tick,
            scheduled_deposit_tick=None,
            departure_tick=None,
            slack_ticks=None,
            status="ON_CORE",
        )
    if direct_core and manhattan(carrier.position, core_position) == 1:
        direction = _cardinal_direction(carrier.position, core_position)
        return CargoReturnReservation(
            worker_id=carrier.id,
            route_target=core_position,
            route_distance=1,
            first_direction=direction,
            first_position=core_position,
            earliest_deposit_tick=world.tick + 1,
            scheduled_deposit_tick=None,
            departure_tick=None,
            slack_ticks=None,
            status="RETURNING",
        )
    lane_index = {cell: index for index, cell in enumerate(queue_cells)}
    if carrier.position in lane_index:
        index = lane_index[carrier.position]
        destination = core_position if index == 0 else queue_cells[index - 1]
        direction = _cardinal_direction(carrier.position, destination)
        distance = index + 1
        return CargoReturnReservation(
            worker_id=carrier.id,
            route_target=destination,
            route_distance=distance,
            first_direction=direction,
            first_position=destination if direction is not None else None,
            earliest_deposit_tick=world.tick + distance,
            scheduled_deposit_tick=None,
            departure_tick=None,
            slack_ticks=None,
            status="RETURNING" if direction is not None else "UNROUTABLE",
            delay_reason=None if direction is not None else "INVALID_SERVICE_LANE",
        )

    target = queue_cells[-1] if queue_cells else core_position
    blocked = set(projection.hostile_occupied)
    blocked.update(projection.immediate_damage)
    blocked.update({core_position, *queue_cells[:-1]})
    if exit_cell is not None:
        blocked.add(exit_cell)
    blocked.discard(carrier.position)
    blocked.discard(target)
    route = weighted_route_to(
        world,
        carrier.position,
        target,
        node_limit=node_limit,
        blocked=frozenset(blocked),
        cell_costs=projection.route_costs_for(UnitType.WORKER),
    )
    if route is None:
        return CargoReturnReservation(
            worker_id=carrier.id,
            route_target=target,
            route_distance=None,
            first_direction=None,
            first_position=None,
            earliest_deposit_tick=None,
            scheduled_deposit_tick=None,
            departure_tick=None,
            slack_ticks=None,
            status="UNROUTABLE",
            delay_reason="NO_RETURN_ROUTE",
        )
    distance = route.distance + len(queue_cells)
    return CargoReturnReservation(
        worker_id=carrier.id,
        route_target=target,
        route_distance=distance,
        first_direction=route.first_direction,
        first_position=route.first_position,
        earliest_deposit_tick=world.tick + distance,
        scheduled_deposit_tick=None,
        departure_tick=None,
        slack_ticks=None,
        status="RETURNING",
    )


def cargo_return_routes(
    world: WorldModel,
    projection: TacticalMap,
    carriers: tuple[EntitySnapshot, ...],
    core_position: Position,
    queue_cells: tuple[Position, ...],
    *,
    node_limit: int,
    exit_cell: Position | None = None,
    direct_core_id: UUID | None = None,
) -> tuple[CargoReturnReservation, ...]:
    """Route a cargo wave through one shared danger-weighted return field.

    Workers already on the service line retain its deterministic one-step
    choreography.  Remote Workers reuse a field rooted at the queue tail;
    only actors outside the bounded field fall back to a single-target A*.
    """

    target = queue_cells[-1] if queue_cells else core_position
    blocked = set(projection.hostile_occupied)
    blocked.update(projection.immediate_damage)
    blocked.update({core_position, *queue_cells[:-1]})
    if exit_cell is not None:
        blocked.add(exit_cell)
    blocked.discard(target)
    distances, parents = weighted_distance_field(
        world,
        target,
        node_limit=node_limit,
        blocked=frozenset(blocked),
        cell_costs=projection.route_costs_for(UnitType.WORKER),
    )

    def shared_route(carrier: EntitySnapshot) -> CargoReturnReservation:
        if (
            carrier.position == core_position
            or carrier.position in queue_cells
            or (
                carrier.id == direct_core_id
                and manhattan(carrier.position, core_position) == 1
            )
        ):
            return cargo_return_route(
                world,
                projection,
                carrier,
                core_position,
                queue_cells,
                node_limit=node_limit,
                exit_cell=exit_cell,
                direct_core=carrier.id == direct_core_id,
            )
        parent = parents.get(carrier.position)
        if carrier.position not in distances or parent is None:
            return cargo_return_route(
                world,
                projection,
                carrier,
                core_position,
                queue_cells,
                node_limit=node_limit,
                exit_cell=exit_cell,
                direct_core=carrier.id == direct_core_id,
            )
        first_position = parent[0]
        first_direction = _cardinal_direction(carrier.position, first_position)
        steps = 0
        cursor = carrier.position
        seen: set[Position] = set()
        while cursor != target and cursor not in seen:
            seen.add(cursor)
            predecessor = parents.get(cursor)
            if predecessor is None:
                break
            cursor = predecessor[0]
            steps += 1
        if cursor != target or first_direction is None:
            return cargo_return_route(
                world,
                projection,
                carrier,
                core_position,
                queue_cells,
                node_limit=node_limit,
                exit_cell=exit_cell,
                direct_core=carrier.id == direct_core_id,
            )
        distance = steps + len(queue_cells)
        return CargoReturnReservation(
            worker_id=carrier.id,
            route_target=target,
            route_distance=distance,
            first_direction=first_direction,
            first_position=first_position,
            earliest_deposit_tick=world.tick + distance,
            scheduled_deposit_tick=None,
            departure_tick=None,
            slack_ticks=None,
            status="RETURNING",
        )

    return tuple(shared_route(carrier) for carrier in carriers)


def service_protected_positions(
    world: WorldModel,
    queue: CoreServiceQueue,
) -> frozenset[Position]:
    """Return cells reserved for Core service, including the Core itself.

    The Core cell is intentionally part of the zone.  Cargo return, healing
    and explicit egress opt into it through intent metadata; ordinary patrol,
    exploration and formation routing must treat it as infrastructure rather
    than a shortcut.
    """

    core_cells: tuple[Position | None, ...]
    if world.core is None:
        core_cells = ()
    else:
        core_cells = (
            world.core.position,
            world.core.destination,
            queue.service_core_position,
        )
    return frozenset(
        cell
        for cell in (
            *core_cells,
            queue.entrance,
            queue.exit_cell,
            *queue.queue_cells,
        )
        if cell is not None
    )


@dataclass(frozen=True, slots=True)
class CoreServiceChoreography:
    """Pure-value pre-plan that protects Core logistics for one Tick.

    Repositories with reliable delivery behavior plan Core egress and cargo
    handoff before ordinary unit tasks, update projected occupancy, and then
    prevent later planners from reassigning those actors.  This record gives
    the intent-based kernel the same invariant without retaining SDK Turn or
    Controller objects.
    """

    protected_positions: frozenset[Position]
    actor_priority_ceilings: tuple[tuple[UUID, int], ...]

    @property
    def preplanned_actor_ids(self) -> tuple[UUID, ...]:
        return tuple(actor_id for actor_id, _ in self.actor_priority_ceilings)

    def priority_ceiling_map(self) -> dict[UUID, int]:
        return dict(self.actor_priority_ceilings)

    @classmethod
    def build(
        cls,
        world: WorldModel,
        queue: CoreServiceQueue,
        intents: Iterable[ActionIntent],
    ) -> CoreServiceChoreography:
        ceilings: dict[UUID, int] = {}
        for intent in intents:
            if intent.actor_id is None or intent.mission not in _SERVICE_MISSIONS:
                continue
            ceilings[intent.actor_id] = max(
                ceilings.get(intent.actor_id, 0),
                intent.priority,
            )
        return cls(
            protected_positions=service_protected_positions(world, queue),
            actor_priority_ceilings=tuple(
                sorted(ceilings.items(), key=lambda item: item[0].bytes)
            ),
        )


class CoreServicePlanner:
    """Plan a stable near-Core logistics corridor and service admission.

    Geometry, FIFO state and treatment arbitration live here so ordinary
    economy and combat planners cannot independently redefine the Core slot.
    The corridor may bend around obstacles; it is not restricted to a straight
    two-cell ray.
    """

    def __init__(self, config: TacticConfig, memory: TacticMemory) -> None:
        self.config = config
        self.memory = memory

    def plan(
        self,
        world: WorldModel,
        projection: TacticalMap,
        *,
        core_starting_move: bool = False,
        projected_core_destination: Position | None = None,
    ) -> CoreServiceQueue:
        if world.core is None:
            return CoreServiceQueue(service="NONE", admission_id=None)
        core = world.core
        self._sync_storage_saturation(world)
        previous_admission = self.memory.service_admission_id
        carriers, wounded, urgent = self._service_actors(world)
        living_carrier_ids = {unit.id for unit in carriers}
        for worker_id in tuple(self.memory.service_worker_progress):
            if worker_id not in living_carrier_ids:
                self.memory.service_worker_progress.pop(worker_id, None)
        for worker_id in tuple(self.memory.service_return_progress):
            if worker_id not in living_carrier_ids:
                self.memory.service_return_progress.pop(worker_id, None)
        for worker_id in tuple(self.memory.service_cargo_first_seen_ticks):
            if worker_id not in living_carrier_ids:
                self.memory.service_cargo_first_seen_ticks.pop(worker_id, None)
        for worker_id in tuple(self.memory.service_deposit_ticks):
            if worker_id not in living_carrier_ids:
                self.memory.service_deposit_ticks.pop(worker_id, None)
        for carrier in carriers:
            self.memory.service_cargo_first_seen_ticks.setdefault(
                carrier.id,
                world.tick,
            )
            previous = self.memory.service_worker_progress.get(carrier.id)
            stalled = (
                previous[1] + 1
                if previous is not None and previous[0] == carrier.position
                else 0
            )
            self.memory.service_worker_progress[carrier.id] = carrier.position, stalled
        if self.memory.patient_admission_progress is not None and wounded:
            sticky_id = self.memory.patient_admission_progress.patient_id
            if (
                self.memory.patient_admission_progress.stalled_ticks >= 2
                and len(wounded) > 1
            ):
                wounded = tuple(unit for unit in wounded if unit.id != sticky_id) + tuple(
                    unit for unit in wounded if unit.id == sticky_id
                )
            else:
                wounded = tuple(unit for unit in wounded if unit.id == sticky_id) + tuple(
                    unit for unit in wounded if unit.id != sticky_id
                )
            urgent = tuple(
                unit
                for unit in wounded
                if unit.hp * 100
                <= UNIT_MAX_HP[unit.unit_type] * self.config.recovery_urgent_percent
            )

        previous_lane = self.memory.service_entrance, self.memory.service_queue_cells
        service_core_position = (
            projected_core_destination
            or core.destination
            or core.position
        )
        entrance, queue_cells, exit_cell = self._choose_lane(
            world,
            projection,
            carriers,
            core_position=service_core_position,
            force_replan=bool(
                any(
                    stalled >= 2
                    for _, stalled in self.memory.service_return_progress.values()
                )
                or any(
                    feedback.stalled_ticks >= 2
                    for feedback in self.memory.service_move_feedback.values()
                )
                or any(
                    carrier.id in self.memory.failed_unit_moves
                    for carrier in carriers
                )
            ),
        )
        if previous_lane != (entrance, queue_cells):
            self.memory.cargo_arrival_ticks.clear()
        core_wounded = next(
            (
                unit
                for unit in wounded
                if unit.position == service_core_position
            ),
            None,
        )
        selected_patient = (
            core_wounded
            or (urgent[0] if urgent else None)
            or (wounded[0] if wounded else None)
        )
        patient_gateway = self._choose_patient_gateway(
            world,
            projection,
            selected_patient,
            entrance,
            exit_cell,
            service_core_position,
        )
        emergency_ready_id = self._emergency_funding_carrier(
            world,
            carriers,
            urgent,
        )
        return_reservations = self._sync_deposit_schedule(
            world,
            projection,
            carriers,
            service_core_position,
            queue_cells,
            exit_cell,
            emergency_ready_id=emergency_ready_id,
        )
        deposit_schedule = tuple(
            (row.worker_id, row.scheduled_deposit_tick)
            for row in return_reservations
            if row.scheduled_deposit_tick is not None
        )
        reservation_by_id = {row.worker_id: row for row in return_reservations}
        for row in return_reservations:
            previous = self.memory.service_return_progress.get(row.worker_id)
            stalled = 0
            if row.status == "RETURNING" and row.route_distance is not None:
                stalled = (
                    previous[1] + 1
                    if previous is not None
                    and previous[0] is not None
                    and row.route_distance >= previous[0]
                    else 0
                )
            self.memory.service_return_progress[row.worker_id] = (
                row.route_distance,
                stalled,
            )

        ready, outside_line, lane_index, head = self._sync_ready_line(
            world,
            carriers,
            queue_cells,
            emergency_ready_id=emergency_ready_id,
        )
        active_approaching = tuple(
            unit
            for unit in outside_line
            if reservation_by_id.get(unit.id) is not None
            and reservation_by_id[unit.id].status == "RETURNING"
        )
        future_remote = tuple(
            unit
            for unit in outside_line
            if reservation_by_id.get(unit.id) is not None
            and reservation_by_id[unit.id].status == "WAIT_FOR_DEPARTURE"
        )
        overflow_slots = self._overflow_slots(
            world,
            projection,
            future_remote,
            service_core_position,
            entrance,
            queue_cells,
            exit_cell,
            deposit_schedule,
        )
        overflow_by_id = dict(overflow_slots)
        holding = tuple(
            unit
            for unit in future_remote
            if overflow_by_id.get(unit.id) == unit.position
        )
        active_approaching = tuple(
            unit for unit in outside_line if unit not in holding
        )
        approaching = outside_line
        ready_ids = {unit.id for unit in ready}

        # START_MOVE is validated before harvest/deposit and turns the Core
        # migration-restricted for the Tick.  Unit healing resolves even later
        # (after combat), so neither service action may share a START_MOVE Tick.
        if core.state is CoreState.MOVING or core_starting_move:
            self._clear_admission()
            target = queue_cells[-1] if queue_cells else service_core_position
            return CoreServiceQueue(
                service="PAUSED",
                admission_id=None,
                service_core_position=service_core_position,
                depositors=tuple(unit.id for unit in (*ready, *approaching)),
                approaching_depositors=tuple(unit.id for unit in active_approaching),
                holding_depositors=tuple(unit.id for unit in (*ready, *holding)),
                overflow_slots=overflow_slots,
                queue_slots=tuple((unit.id, target) for unit in active_approaching),
                scheduled_deposits=deposit_schedule,
                return_reservations=return_reservations,
                worker_progress=tuple(
                    (
                        unit.id,
                        unit.position,
                        self.memory.service_return_progress.get(unit.id, (None, 0))[1],
                    )
                    for unit in (*ready, *approaching)
                ),
                wounded=tuple(unit.id for unit in wounded),
                entrance=entrance,
                queue_cells=queue_cells,
                exit_cell=exit_cell,
                patient_gateway=patient_gateway,
                core_slot_reserved=False,
                paused_reason=(
                    "CORE_MOVING" if core.state is CoreState.MOVING else "CORE_STARTING_MOVE"
                ),
                previous_admission_id=previous_admission,
                release_reason=(
                    (
                        "CORE_MOVING"
                        if core.state is CoreState.MOVING
                        else "CORE_STARTING_MOVE"
                    )
                    if previous_admission is not None
                    else None
                ),
            )

        lane_cells = (core.position, *queue_cells)
        minimum_carrier_hp = min((unit.hp for unit in carriers), default=1)
        lane_threatened = bool(carriers) and any(
            projection.immediate_attackers(cell) >= minimum_carrier_hp
            for cell in lane_cells
        )
        units_by_id = {unit.id: unit for unit in world.friendlies}
        # Resource reservation is immediate, but admission and all physical
        # service timing are decided exactly once by the unified calendar
        # below.  The former patient selector and cargo selector must not
        # compete before that calendar exists.
        admission: UUID | None = None
        service = "IDLE"
        reserved = sum(
            UNIT_MAX_HP[unit.unit_type] - unit.hp
            for unit in wounded
            if (
                unit.unit_type is UnitType.WORKER
                or (
                    unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                    and unit.hp * 2 <= UNIT_MAX_HP[unit.unit_type]
                )
            )
        )

        admitted_unit = units_by_id.get(admission) if admission is not None else None
        patient_progress = self._patient_progress(
            world,
            projection,
            selected_patient,
            patient_gateway,
        )
        patient_queue = self._patient_queue_entries(
            world,
            projection,
            wounded,
            entrance,
            exit_cell,
            service_core_position,
            patient_progress,
        )
        (
            jobs,
            slot_schedule,
            admission,
            service,
            patient_gateway,
            return_reservations,
            timeline,
            service_windows,
        ) = self._unified_service_schedule(
            world,
            projection,
            carriers,
            wounded,
            patient_queue,
            return_reservations,
            service_core_position,
            entrance,
            exit_cell,
            reserved,
        )
        admitted_unit = units_by_id.get(admission) if admission is not None else None
        treatment_service = service in {"EMERGENCY_HEAL", "MAINTENANCE_HEAL"}
        patient_threatened = bool(
            treatment_service
            and admitted_unit is not None
            and (
                projection.immediate_attackers(core.position) >= admitted_unit.hp
                or (
                    patient_gateway is not None
                    and projection.immediate_attackers(patient_gateway) >= admitted_unit.hp
                )
            )
        )
        paused_reason = (
            "PATIENT_GATEWAY_LETHAL"
            if patient_threatened
            else "LANE_THREATENED"
            if lane_threatened and not treatment_service
            else None
        )
        deposit_schedule = tuple(
            (row.worker_id, row.scheduled_deposit_tick)
            for row in return_reservations
            if row.scheduled_deposit_tick is not None
        )
        reservation_by_id = {row.worker_id: row for row in return_reservations}
        active_approaching = tuple(
            unit
            for unit in approaching
            if not (
                reservation_by_id.get(unit.id) is not None
                and reservation_by_id[unit.id].status == "WAIT_FOR_DEPARTURE"
                and overflow_by_id.get(unit.id) == unit.position
            )
        )
        holding = tuple(
            unit for unit in approaching if unit not in active_approaching
        )
        service_cell_leases = self._service_cell_leases(
            world,
            service_windows,
            entrance,
            queue_cells,
            exit_cell,
        )
        blocking_units = self._service_blockers(
            world,
            service_cell_leases,
        )
        core_slot_reserved = slot_schedule.slot_reserved

        release_reason = self._release_reason(
            previous_admission,
            admission,
            units_by_id,
            ready_ids,
            head,
        )
        if admission != self.memory.service_admission_id:
            self.memory.service_started_tick = world.tick
        self.memory.service_admission_id = admission
        self.memory.service_kind = service
        admission_reason = self._admission_reason(
            previous_admission,
            admission,
            service,
            ready,
            head,
        )
        allow_advance = not lane_threatened and service not in {
            "EMERGENCY_HEAL",
            "MAINTENANCE_HEAL",
        }
        return CoreServiceQueue(
            service="PAUSED" if paused_reason is not None else service,
            admission_id=admission,
            service_core_position=service_core_position,
            depositors=tuple(unit.id for unit in (*ready, *approaching)),
            ready_depositors=tuple(unit.id for unit in ready),
            approaching_depositors=tuple(unit.id for unit in active_approaching),
            holding_depositors=tuple(unit.id for unit in holding),
            overflow_slots=overflow_slots,
            scheduled_deposits=tuple(
                sorted(
                    self.memory.service_deposit_ticks.items(),
                    key=lambda item: (item[1], item[0].bytes),
                )
            ),
            return_reservations=return_reservations,
            ready_ticks=tuple(
                (unit.id, self.memory.cargo_arrival_ticks[unit.id]) for unit in ready
            ),
            queue_slots=self._queue_slots(
                core.position,
                ready,
                active_approaching,
                lane_index,
                queue_cells,
                exit_cell,
                admission,
                allow_advance,
            ),
            worker_progress=tuple(
                (
                    unit.id,
                    unit.position,
                    self.memory.service_return_progress.get(unit.id, (None, 0))[1],
                )
                for unit in (*ready, *approaching)
            ),
            wounded=tuple(unit.id for unit in wounded),
            entrance=entrance,
            queue_cells=queue_cells,
            exit_cell=exit_cell,
            patient_gateway=patient_gateway,
            core_slot_reserved=core_slot_reserved,
            patient_progress=patient_progress,
            service_windows=service_windows,
            patient_queue=patient_queue,
            service_cell_leases=service_cell_leases,
            jobs=jobs,
            slot_schedule=slot_schedule,
            blocking_units=blocking_units,
            reschedule_reasons=tuple(
                sorted(
                    {
                        row.delay_reason
                        for row in return_reservations
                        if row.delay_reason is not None
                    }
                )
            ),
            timeline=timeline,
            reserved_resources=reserved,
            paused_reason=paused_reason,
            previous_admission_id=previous_admission,
            admission_reason=admission_reason,
            release_reason=release_reason,
        )

    def _overflow_slots(
        self,
        world: WorldModel,
        projection: TacticalMap,
        holding: tuple[EntitySnapshot, ...],
        core_position: Position,
        queue_cells: tuple[Position, ...],
        entrance: Position | None,
        exit_cell: Position | None,
        deposit_schedule: tuple[tuple[UUID, int], ...],
    ) -> tuple[tuple[UUID, Position], ...]:
        if not holding:
            return ()
        protected = {
            core_position,
            *queue_cells,
            *(cell for cell in (entrance, exit_cell) if cell is not None),
        }
        occupied = dict(world.occupied_cells)
        candidates = tuple(
            cell
            for radius in range(
                self.config.service_lane_depth + 2,
                self.config.service_lane_depth + 7,
            )
            for cell in manhattan_ring(core_position, radius)
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in projection.hostile_occupied
            and cell not in protected
            and occupied.get(cell, 0) < 2
            and projection.immediate_attackers(cell) == 0
        )
        available = set(candidates)
        schedule = dict(deposit_schedule)
        minimum_radius = self.config.service_lane_depth + 2
        maximum_radius = self.config.service_lane_depth + 6
        rows: list[tuple[UUID, Position]] = []
        for worker in sorted(
            holding,
            key=lambda unit: (
                schedule.get(unit.id, world.tick),
                self.memory.service_cargo_first_seen_ticks.get(unit.id, world.tick),
                unit.id.bytes,
            ),
        ):
            remaining = max(0, schedule.get(worker.id, world.tick) - world.tick)
            preferred_radius = max(
                minimum_radius,
                min(maximum_radius, remaining - self.config.service_lane_depth),
            )
            target = min(
                available,
                key=lambda cell: (
                    abs(manhattan(core_position, cell) - preferred_radius),
                    manhattan(worker.position, cell),
                    projection.future_attackers(cell),
                    projection.threat_heat.get(cell, 0),
                    cell,
                ),
                default=worker.position,
            )
            rows.append((worker.id, target))
            available.discard(target)
        return tuple(rows)

    def _sync_deposit_schedule(
        self,
        world: WorldModel,
        projection: TacticalMap,
        carriers: tuple[EntitySnapshot, ...],
        core_position: Position,
        queue_cells: tuple[Position, ...],
        exit_cell: Position | None,
        *,
        emergency_ready_id: UUID | None = None,
    ) -> tuple[CargoReturnReservation, ...]:
        """Build executable routes and a stable, feasibility-correct calendar."""

        drafts = cargo_return_routes(
            world,
            projection,
            carriers,
            core_position,
            queue_cells,
            node_limit=self.config.path_node_limit,
            exit_cell=exit_cell,
            direct_core_id=emergency_ready_id,
        )
        reachable = tuple(
            row
            for row in drafts
            if row.earliest_deposit_tick is not None
            and row.route_distance is not None
        )
        used_ticks: set[int] = set()

        def reservation_status(
            row: CargoReturnReservation,
            departure: int,
        ) -> str:
            if row.route_distance == 0:
                return "ON_CORE"
            if world.tick >= departure:
                return "RETURNING"
            return "WAIT_FOR_DEPARTURE"

        def next_available(requested: int) -> int:
            assigned = requested
            while any(abs(assigned - other) < 2 for other in used_ticks):
                assigned += 1
            return assigned

        # Rebuild a compressed rolling calendar from current physical ETAs.
        # Previous ticks only stabilize equal-ETA ordering; they never pin an
        # unreachable absolute appointment or leave a stale hole at the head.
        ordered = sorted(
            reachable,
            key=lambda row: (
                row.earliest_deposit_tick,
                self.memory.service_cargo_first_seen_ticks.get(row.worker_id, world.tick),
                self.memory.service_deposit_ticks.get(row.worker_id, 1 << 60),
                row.worker_id.bytes,
            ),
        )
        assigned_rows: dict[UUID, CargoReturnReservation] = {}
        for row in ordered:
            assert row.earliest_deposit_tick is not None
            requested = self.memory.service_deposit_ticks.get(row.worker_id)
            assigned = next_available(row.earliest_deposit_tick)
            used_ticks.add(assigned)
            departure = assigned - row.route_distance
            assigned_rows[row.worker_id] = replace(
                row,
                scheduled_deposit_tick=assigned,
                departure_tick=departure,
                slack_ticks=max(0, departure - world.tick),
                status=reservation_status(row, departure),
                delay_reason=(
                    "MISSED_APPOINTMENT"
                    if requested is not None and requested < row.earliest_deposit_tick
                    else "ROLLING_COMPRESSION"
                    if requested is not None and assigned < requested
                    else "SERVICE_CONFLICT"
                    if assigned > row.earliest_deposit_tick
                    else None
                ),
            )

        result = tuple(
            sorted(
                (
                    assigned_rows.get(row.worker_id, row)
                    for row in drafts
                ),
                key=lambda row: (
                    row.scheduled_deposit_tick
                    if row.scheduled_deposit_tick is not None
                    else 1 << 60,
                    row.worker_id.bytes,
                ),
            )
        )
        self.memory.service_deposit_ticks = {
            row.worker_id: row.scheduled_deposit_tick
            for row in result
            if row.scheduled_deposit_tick is not None
        }
        return result

    def _unified_service_schedule(
        self,
        world: WorldModel,
        projection: TacticalMap,
        carriers: tuple[EntitySnapshot, ...],
        wounded: tuple[EntitySnapshot, ...],
        patient_queue: tuple[PatientQueueEntry, ...],
        reservations: tuple[CargoReturnReservation, ...],
        core_position: Position,
        entrance: Position | None,
        exit_cell: Position | None,
        reserved_resources: int,
    ) -> tuple[
        tuple[CoreServiceJob, ...],
        CoreSlotSchedule,
        UUID | None,
        str,
        Position | None,
        tuple[CargoReturnReservation, ...],
        CoreOperationTimeline,
        tuple[CoreServiceWindow, ...],
    ]:
        """Schedule every Core visitor on one work-conserving calendar.

        The old implementation selected one patient before building an
        unrelated cargo calendar.  That allowed a distant casualty to freeze
        the slot while nearby maintenance and deposits waited.  Here every
        actor owns one job; a wounded loaded Worker therefore has a single
        ordered ``DEPOSIT, HEAL`` (or ``HEAL, DEPOSIT``) visit.
        """

        assert world.core is not None
        units = {unit.id: unit for unit in world.friendlies}
        patient_rows = {row.patient_id: row for row in patient_queue}
        reservation_rows = {row.worker_id: row for row in reservations}
        actor_ids = {unit.id for unit in carriers} | {unit.id for unit in wounded}
        for actor_id in tuple(self.memory.service_ready_since_ticks):
            if actor_id not in actor_ids:
                self.memory.service_ready_since_ticks.pop(actor_id, None)
        for actor_id in tuple(self.memory.service_patient_gateways):
            if actor_id not in {unit.id for unit in wounded}:
                self.memory.service_patient_gateways.pop(actor_id, None)

        urgent_resource_need = sum(
            row.resource_cost for row in patient_queue if row.urgent
        )
        drafts: list[CoreServiceJob] = []
        for actor_id in sorted(actor_ids, key=lambda value: value.bytes):
            actor = units.get(actor_id)
            if actor is None:
                continue
            patient = patient_rows.get(actor_id)
            cargo = reservation_rows.get(actor_id)
            missing = UNIT_MAX_HP[actor.unit_type] - actor.hp
            can_deposit = bool(
                actor.unit_type is UnitType.WORKER
                and actor.cargo > 0
                and (
                    world.resources < world.resource_capacity
                    or missing > 0
                )
            )
            can_heal = missing > 0
            if not can_deposit and not can_heal:
                continue

            if can_deposit and can_heal:
                operations = (
                    ("HEAL", "DEPOSIT")
                    if world.resources >= world.resource_capacity
                    and world.resources >= missing
                    else ("DEPOSIT", "HEAL")
                )
            elif can_deposit:
                operations = ("DEPOSIT",)
            else:
                operations = ("HEAL",)

            # Loaded Workers use the exact cargo route consumed by WorkerPlanner.
            # Other patients receive their own gateway and route; no patient is
            # sent to a generic recovery ring merely because another patient is
            # currently more urgent.
            gateway = None
            route_distance: int | None = None
            first_direction: Direction | None = None
            first_position: Position | None = None
            if actor.position == core_position:
                route_distance = 0
            elif cargo is not None and can_deposit:
                gateway = cargo.route_target
                route_distance = cargo.route_distance
                first_direction = cargo.first_direction
                first_position = cargo.first_position
            elif patient is not None:
                gateway = patient.gateway
                route_distance = patient.eta
                if gateway is not None:
                    if actor.position == gateway:
                        first_direction = _cardinal_direction(actor.position, core_position)
                        first_position = core_position
                    else:
                        route = weighted_route_to(
                            world,
                            actor.position,
                            gateway,
                            node_limit=self.config.path_node_limit,
                            blocked=frozenset(
                                projection.hostile_occupied
                                - {actor.position, gateway}
                            ),
                            cell_costs=dict(
                                projection.route_costs_for(actor.unit_type)
                            ),
                        )
                        if route is not None:
                            first_direction = route.first_direction
                            first_position = route.first_position

            earliest = (
                None
                if route_distance is None
                else world.tick + route_distance
            )
            if route_distance is not None and route_distance <= 1:
                ready_since = self.memory.service_ready_since_ticks.setdefault(
                    actor_id, world.tick
                )
            else:
                self.memory.service_ready_since_ticks.pop(actor_id, None)
                ready_since = None
            urgent = bool(patient is not None and patient.urgent)
            aged_maintenance = bool(
                patient is not None
                and not urgent
                and ready_since is not None
                and world.tick - ready_since >= 8
            )
            priority = (
                -100
                if actor.position == core_position
                else 0
                if urgent
                else 15
                if aged_maintenance
                else 20
                if can_deposit
                else 30
            )
            phase = (
                CoreServicePhase.SERVICE
                if route_distance == 0
                else CoreServicePhase.ENTRY
                if route_distance == 1
                else CoreServicePhase.APPROACHING
            )
            drafts.append(
                CoreServiceJob(
                    actor_id=actor_id,
                    operations=operations,
                    phase=phase,
                    route_distance=route_distance,
                    first_direction=first_direction,
                    first_position=first_position,
                    gateway=gateway,
                    earliest_service_tick=earliest,
                    service_tick=None,
                    exit_tick=None,
                    priority=priority,
                    ready_since_tick=ready_since,
                    resource_cost=missing if can_heal else 0,
                    resource_gain=actor.cargo if can_deposit else 0,
                    reason=(
                        "COMPOUND_WOUNDED_CARGO"
                        if can_deposit and can_heal
                        else "PATIENT"
                        if can_heal
                        else "CARGO"
                    ),
                )
            )

        reachable = [job for job in drafts if job.earliest_service_tick is not None]
        unreachable = [job for job in drafts if job.earliest_service_tick is None]
        scheduled: list[CoreServiceJob] = []
        cursor = world.tick
        projected_resources = world.resources
        remaining_urgent_need = urgent_resource_need
        while reachable:
            ready_jobs = [
                job for job in reachable if job.earliest_service_tick <= cursor
            ]
            if not ready_jobs:
                cursor = min(job.earliest_service_tick for job in reachable)
                ready_jobs = [
                    job for job in reachable if job.earliest_service_tick <= cursor
                ]

            def executable(job: CoreServiceJob) -> bool:
                first = job.operations[0]
                if first != "HEAL":
                    return True
                # Maintenance may fill a real idle gap, but never consume funds
                # already reserved for an emergency patient arriving later.
                actor = units[job.actor_id]
                patient = patient_rows.get(job.actor_id)
                if projected_resources < job.resource_cost:
                    return False
                return bool(
                    patient is not None
                    and patient.urgent
                    or projected_resources - job.resource_cost
                    >= remaining_urgent_need
                    or actor.position == core_position
                    and not any(row.urgent for row in patient_queue)
                )

            executable_jobs = [job for job in ready_jobs if executable(job)]
            if not executable_jobs:
                funding = [
                    job for job in reachable if job.operations[0] == "DEPOSIT"
                ]
                if funding:
                    cursor = max(
                        cursor,
                        min(job.earliest_service_tick for job in funding),
                    )
                    executable_jobs = [
                        job
                        for job in funding
                        if job.earliest_service_tick <= cursor
                    ]
                else:
                    # Keep an underfunded patient visible in the soft calendar;
                    # RecoveryPlanner will approach while the Core remains free.
                    executable_jobs = ready_jobs

            funding_needed = projected_resources < remaining_urgent_need
            chosen = min(
                executable_jobs,
                key=lambda job: (
                    0
                    if funding_needed and job.operations[0] == "DEPOSIT"
                    else 1,
                    job.priority,
                    job.ready_since_tick
                    if job.ready_since_tick is not None
                    else 1 << 60,
                    job.earliest_service_tick,
                    b"" if job.actor_id is None else job.actor_id.bytes,
                ),
            )
            start = max(cursor, chosen.earliest_service_tick)
            actor = units[chosen.actor_id]
            phase = (
                CoreServicePhase.SERVICE
                if actor.position == core_position and start == world.tick
                else CoreServicePhase.ENTRY
                if chosen.route_distance == 1 and start <= world.tick + 1
                else CoreServicePhase.APPROACHING
            )
            final = start + len(chosen.operations) - 1
            chosen = replace(
                chosen,
                phase=phase,
                service_tick=start,
                exit_tick=final + 1,
            )
            scheduled.append(chosen)
            reachable.remove(next(job for job in reachable if job.actor_id == chosen.actor_id))
            for operation in chosen.operations:
                if operation == "DEPOSIT":
                    projected_resources = min(
                        world.resource_capacity,
                        projected_resources + chosen.resource_gain,
                    )
                elif operation == "HEAL" and projected_resources >= chosen.resource_cost:
                    projected_resources -= chosen.resource_cost
                    patient = patient_rows.get(chosen.actor_id)
                    if patient is not None and patient.urgent:
                        remaining_urgent_need = max(
                            0, remaining_urgent_need - chosen.resource_cost
                        )
            # Final service at ``final``; egress and next entry share final+1,
            # so the next actor can perform service at final+2.
            cursor = final + 2

        jobs = tuple(
            sorted(
                (*scheduled, *unreachable),
                key=lambda job: (
                    job.service_tick if job.service_tick is not None else 1 << 60,
                    job.priority,
                    b"" if job.actor_id is None else job.actor_id.bytes,
                ),
            )
        )
        jobs_by_id = {job.actor_id: job for job in jobs}

        updated_reservations: list[CargoReturnReservation] = []
        for row in reservations:
            job = jobs_by_id.get(row.worker_id)
            if job is None or "DEPOSIT" not in job.operations or job.service_tick is None:
                updated_reservations.append(row)
                continue
            deposit_tick = job.service_tick + job.operations.index("DEPOSIT")
            distance = row.route_distance
            departure = (
                None if distance is None else deposit_tick - distance
            )
            status = (
                "UNROUTABLE"
                if distance is None
                else "ON_CORE"
                if distance == 0
                else "RETURNING"
                if "HEAL" in job.operations or departure is None or world.tick >= departure
                else "WAIT_FOR_DEPARTURE"
            )
            updated_reservations.append(
                replace(
                    row,
                    scheduled_deposit_tick=deposit_tick,
                    departure_tick=departure,
                    slack_ticks=(
                        None if departure is None else max(0, departure - world.tick)
                    ),
                    status=status,
                    delay_reason=(
                        "UNIFIED_SERVICE_CALENDAR"
                        if row.scheduled_deposit_tick != deposit_tick
                        else row.delay_reason
                    ),
                )
            )
        reservations = tuple(
            sorted(
                updated_reservations,
                key=lambda row: (
                    row.scheduled_deposit_tick
                    if row.scheduled_deposit_tick is not None
                    else 1 << 60,
                    row.worker_id.bytes,
                ),
            )
        )
        self.memory.service_deposit_ticks = {
            row.worker_id: row.scheduled_deposit_tick
            for row in reservations
            if row.scheduled_deposit_tick is not None
        }

        occupant = next(
            (unit for unit in world.friendlies if unit.position == core_position),
            None,
        )
        current_job = jobs_by_id.get(occupant.id) if occupant is not None else None
        admission: UUID | None = None
        if current_job is not None and current_job.service_tick == world.tick:
            admission = current_job.actor_id
        elif occupant is None:
            incoming = next(
                (
                    job
                    for job in jobs
                    if job.route_distance == 1
                    and job.service_tick == world.tick + 1
                ),
                None,
            )
            admission = None if incoming is None else incoming.actor_id
        else:
            # A healthy/finished occupant can leave while an adjacent actor
            # enters through the movement dependency chain.
            incoming = next(
                (
                    job
                    for job in jobs
                    if job.route_distance == 1
                    and job.service_tick == world.tick + 1
                ),
                None,
            )
            if current_job is None and incoming is not None:
                admission = incoming.actor_id

        admitted_job = jobs_by_id.get(admission)
        current_operation = None
        if current_job is not None and current_job.service_tick is not None:
            operation_index = world.tick - current_job.service_tick
            if 0 <= operation_index < len(current_job.operations):
                current_operation = current_job.operations[operation_index]
        next_operation = (
            admitted_job.operations[0]
            if admitted_job is not None and current_operation is None
            else current_operation
        )
        admitted_patient = patient_rows.get(admission) if admission is not None else None
        service = (
            "HEAL_FUNDING"
            if next_operation == "DEPOSIT"
            and urgent_resource_need > world.resources
            else "DEPOSIT"
            if next_operation == "DEPOSIT"
            else "EMERGENCY_HEAL"
            if next_operation == "HEAL" and admitted_patient is not None and admitted_patient.urgent
            else "MAINTENANCE_HEAL"
            if next_operation == "HEAL"
            else "DEPOSIT_APPROACH"
            if carriers
            else "RECOVERY_APPROACH"
            if wounded
            else "IDLE"
        )
        patient_gateway = (
            admitted_job.gateway
            if admitted_job is not None and "HEAL" in admitted_job.operations
            else next(
                (job.gateway for job in jobs if "HEAL" in job.operations),
                None,
            )
        )

        egress_rows = tuple(
            cell
            for _, cell in cardinal_neighbors(core_position)
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in projection.hostile_occupied
            and projection.immediate_attackers(cell) == 0
            and cell not in {entrance}
        )
        next_job = next(
            (job for job in jobs if job.service_tick is not None and job.service_tick >= world.tick),
            None,
        )
        slot_reserved = bool(occupant is not None or admission is not None)
        production_allowed = bool(egress_rows) and admission is None
        if occupant is not None and current_job is not None:
            production_allowed = False
        elif next_job is not None and next_job.service_tick <= world.tick + 1:
            production_allowed = False
        reason = (
            "CURRENT_SERVICE"
            if current_job is not None
            else "SERVICE_DUE_THIS_TICK"
            if admission is not None
            else "SAFE_BEFORE_FUTURE_SERVICE"
            if production_allowed
            else "NO_SAFE_SPAWN_EGRESS"
            if not egress_rows
            else "SERVICE_DUE_NEXT_TICK"
        )
        schedule = CoreSlotSchedule(
            tick=world.tick,
            jobs=jobs,
            current_job_id=None if current_job is None else current_job.actor_id,
            next_job_id=None if next_job is None else next_job.actor_id,
            slot_owner_id=None if occupant is None else occupant.id,
            slot_reserved=slot_reserved,
            production_allowed=production_allowed,
            spawn_egress_cell=min(egress_rows, default=exit_cell),
            reason=reason,
        )
        requests: list[CoreOperationRequest] = []
        windows: list[CoreServiceWindow] = []
        for job in jobs:
            if job.actor_id is None or job.service_tick is None or job.exit_tick is None:
                continue
            for index, operation in enumerate(job.operations):
                service_tick = job.service_tick + index
                requests.append(
                    CoreOperationRequest(
                        actor_id=job.actor_id,
                        operation=operation,
                        eta=max(0, service_tick - world.tick),
                        occupy_tick=service_tick,
                        release_tick=job.exit_tick,
                        priority=job.priority,
                        resource_cost=job.resource_cost if operation == "HEAL" else 0,
                        resource_gain=job.resource_gain if operation == "DEPOSIT" else 0,
                        gateway=job.gateway,
                    )
                )
                windows.append(
                    CoreServiceWindow(
                        actor_id=job.actor_id,
                        operation=operation,
                        enter_tick=max(world.tick, job.service_tick - 1),
                        service_tick=service_tick,
                        exit_tick=job.exit_tick,
                        gateway=job.gateway,
                        status=job.phase.value,
                    )
                )
        timeline = CoreOperationTimeline(
            tick=world.tick,
            requests=tuple(requests),
            current_slot_owner=None if occupant is None else occupant.id,
            current_slot_reserved=slot_reserved,
            next_service_eta=(
                None
                if next_job is None or next_job.service_tick is None
                else max(0, next_job.service_tick - world.tick)
            ),
            next_service_tick=None if next_job is None else next_job.service_tick,
            next_release_tick=None if next_job is None else next_job.exit_tick,
            production_allowed=production_allowed,
            spawn_egress_cell=schedule.spawn_egress_cell,
            reason=reason,
        )
        return (
            jobs,
            schedule,
            admission,
            service,
            patient_gateway,
            reservations,
            timeline,
            tuple(windows),
        )

    def _shift_reservations_for_patient(
        self,
        world: WorldModel,
        reservations: tuple[CargoReturnReservation, ...],
        patient_progress: PatientAdmissionProgress | None,
        service: str,
    ) -> tuple[CargoReturnReservation, ...]:
        if (
            patient_progress is None
            or patient_progress.entry_distance is None
            or patient_progress.entry_distance > 2
        ):
            return reservations
        patient_tick = world.tick + patient_progress.entry_distance
        if (
            patient_progress.entry_distance == 0
            and service in {
                "DEPOSIT",
                "DEPOSIT_BEFORE_PATIENT",
                "WOUNDED_CARGO_DEPOSIT",
            }
        ):
            patient_tick += 1
        used: set[int] = set()
        rows: list[CargoReturnReservation] = []
        for row in sorted(
            reservations,
            key=lambda item: (
                item.scheduled_deposit_tick
                if item.scheduled_deposit_tick is not None
                else 1 << 60,
                item.worker_id.bytes,
            ),
        ):
            if row.scheduled_deposit_tick is None or row.route_distance is None:
                rows.append(row)
                continue
            assigned = row.scheduled_deposit_tick
            original = assigned
            while (
                assigned in {patient_tick, patient_tick + 1}
                or any(abs(assigned - other) < 2 for other in used)
            ):
                assigned += 1
            used.add(assigned)
            departure = assigned - row.route_distance
            status = (
                "ON_CORE"
                if row.route_distance == 0
                else "RETURNING"
                if world.tick >= departure
                else "WAIT_FOR_DEPARTURE"
            )
            rows.append(
                replace(
                    row,
                    scheduled_deposit_tick=assigned,
                    departure_tick=departure,
                    slack_ticks=max(0, departure - world.tick),
                    status=status,
                    delay_reason=(
                        "PATIENT_WINDOW" if assigned != original else row.delay_reason
                    ),
                )
            )
        self.memory.service_deposit_ticks = {
            row.worker_id: row.scheduled_deposit_tick
            for row in rows
            if row.scheduled_deposit_tick is not None
        }
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.scheduled_deposit_tick
                    if row.scheduled_deposit_tick is not None
                    else 1 << 60,
                    row.worker_id.bytes,
                ),
            )
        )

    def _choose_patient_gateway(
        self,
        world: WorldModel,
        projection: TacticalMap,
        patient: EntitySnapshot | None,
        entrance: Position | None,
        exit_cell: Position | None,
        core_position: Position,
    ) -> Position | None:
        if patient is None or patient.position == core_position:
            return None
        preferred = tuple(
            cell for cell in (exit_cell, entrance) if cell is not None
        )
        occupied = dict(world.occupied_cells)
        candidates = []
        for index, (_, cell) in enumerate(cardinal_neighbors(core_position)):
            if (
                cell in world.known_obstacles
                or cell in projection.hostile_occupied
                or occupied.get(cell, 0) >= 2
                or projection.immediate_attackers(cell) >= patient.hp
            ):
                continue
            route = route_to(
                world,
                patient.position,
                cell,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(projection.hostile_occupied - {patient.position}),
            )
            if route is None:
                continue
            candidates.append(
                (
                    (
                        route.distance,
                        projection.future_attackers(cell),
                        preferred.index(cell) if cell in preferred else len(preferred),
                        index,
                    ),
                    cell,
                )
            )
        return min(candidates, key=lambda row: row[0], default=((), None))[1]

    def _patient_progress(
        self,
        world: WorldModel,
        projection: TacticalMap,
        patient: EntitySnapshot | None,
        gateway: Position | None,
    ) -> PatientAdmissionProgress | None:
        if patient is None:
            self.memory.patient_admission_progress = None
            return None
        previous = self.memory.patient_admission_progress
        stalled = (
            previous.stalled_ticks + 1
            if previous is not None
            and previous.patient_id == patient.id
            and previous.last_position == patient.position
            else 0
        )
        entry_distance: int | None
        if world.core is not None and patient.position == world.core.position:
            entry_distance = 0
        elif gateway is None:
            entry_distance = None
        else:
            route = route_to(
                world,
                patient.position,
                gateway,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(
                    projection.hostile_occupied - {patient.position, gateway}
                ),
            )
            entry_distance = None if route is None else route.distance + 1
        progress = PatientAdmissionProgress(
            patient_id=patient.id,
            gateway=gateway,
            started_tick=(
                previous.started_tick
                if previous is not None and previous.patient_id == patient.id
                else world.tick
            ),
            last_position=patient.position,
            stalled_ticks=stalled,
            entry_distance=entry_distance,
        )
        self.memory.patient_admission_progress = progress
        return progress

    def _patient_queue_entries(
        self,
        world: WorldModel,
        projection: TacticalMap,
        wounded: tuple[EntitySnapshot, ...],
        entrance: Position | None,
        exit_cell: Position | None,
        core_position: Position,
        selected_progress: PatientAdmissionProgress | None,
    ) -> tuple[PatientQueueEntry, ...]:
        rows: list[PatientQueueEntry] = []
        for patient in wounded:
            gateway = self.memory.service_patient_gateways.get(patient.id)
            if gateway is not None:
                persisted_route = route_to(
                    world,
                    patient.position,
                    gateway,
                    node_limit=self.config.path_node_limit,
                    blocked=frozenset(
                        projection.hostile_occupied - {patient.position, gateway}
                    ),
                )
                if (
                    manhattan(gateway, core_position) != 1
                    or gateway in world.known_obstacles
                    or gateway in projection.hostile_occupied
                    or projection.immediate_attackers(gateway) >= patient.hp
                    or persisted_route is None
                ):
                    gateway = None
            if gateway is None:
                gateway = self._choose_patient_gateway(
                    world,
                    projection,
                    patient,
                    entrance,
                    exit_cell,
                    core_position,
                )
            if gateway is None:
                self.memory.service_patient_gateways.pop(patient.id, None)
            else:
                self.memory.service_patient_gateways[patient.id] = gateway
            if patient.position == core_position:
                eta = 0
            elif gateway is None:
                eta = None
            else:
                route = route_to(
                    world,
                    patient.position,
                    gateway,
                    node_limit=self.config.path_node_limit,
                    blocked=frozenset(
                        projection.hostile_occupied - {patient.position, gateway}
                    ),
                )
                eta = None if route is None else route.distance + 1
            progress = (
                selected_progress
                if selected_progress is not None
                and selected_progress.patient_id == patient.id
                else None
            )
            urgent = patient.hp * 2 <= UNIT_MAX_HP[patient.unit_type]
            rows.append(
                PatientQueueEntry(
                    patient_id=patient.id,
                    urgent=urgent,
                    hp_percent=patient.hp * 100 // UNIT_MAX_HP[patient.unit_type],
                    eta=eta,
                    gateway=gateway,
                    stalled_ticks=0 if progress is None else progress.stalled_ticks,
                    resource_cost=UNIT_MAX_HP[patient.unit_type] - patient.hp,
                    status=(
                        "ON_CORE"
                        if eta == 0
                        else "ENTRY_READY"
                        if eta == 1
                        else "APPROACHING"
                        if eta is not None
                        else "NO_SAFE_ENTRY"
                    ),
                )
            )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.eta != 0,
                    not row.urgent,
                    row.hp_percent,
                    1 << 30 if row.eta is None else row.eta,
                    row.patient_id.bytes,
                ),
            )
        )

    @staticmethod
    def _service_windows(
        timeline: CoreOperationTimeline,
    ) -> tuple[CoreServiceWindow, ...]:
        rows = []
        for request in timeline.requests:
            if request.actor_id is None:
                continue
            service_tick = (
                request.release_tick - 1
                if request.operation == "DEPOSIT"
                else request.occupy_tick
            )
            enter_tick = (
                request.occupy_tick
                if request.operation == "DEPOSIT"
                else max(timeline.tick, request.occupy_tick - 1)
            )
            rows.append(
                CoreServiceWindow(
                    actor_id=request.actor_id,
                    operation=request.operation,
                    enter_tick=enter_tick,
                    service_tick=service_tick,
                    exit_tick=request.release_tick,
                    gateway=request.gateway,
                    status=(
                        "CURRENT"
                        if service_tick <= timeline.tick
                        else "ENTRY_DUE"
                        if enter_tick <= timeline.tick + 1
                        else "FUTURE"
                    ),
                )
            )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.enter_tick,
                    row.service_tick,
                    row.actor_id.bytes,
                ),
            )
        )

    @staticmethod
    def _service_cell_leases(
        world: WorldModel,
        windows: tuple[CoreServiceWindow, ...],
        entrance: Position | None,
        queue_cells: tuple[Position, ...],
        exit_cell: Position | None,
    ) -> tuple[ServiceCellLease, ...]:
        if world.core is None:
            return ()
        positions = {unit.id: unit.position for unit in world.friendlies}
        leases: dict[Position, ServiceCellLease] = {}
        for window in windows:
            if window.enter_tick > world.tick + 2:
                continue
            cells = [(world.core.position, "CORE_SLOT")]
            if window.operation == "HEAL" and window.gateway is not None:
                cells.append((window.gateway, "PATIENT_GATEWAY"))
            elif (
                window.operation == "DEPOSIT"
                and positions.get(window.actor_id) != world.core.position
            ):
                actor_position = positions.get(window.actor_id)
                if actor_position in queue_cells:
                    cells.append(
                        (
                            actor_position,
                            "CARGO_ENTRANCE"
                            if actor_position == entrance
                            else "QUEUE_NEXT",
                        )
                    )
                elif entrance is not None:
                    cells.append((entrance, "CARGO_ENTRANCE"))
            if exit_cell is not None and window.service_tick <= world.tick:
                cells.append((exit_cell, "SERVICE_EXIT"))
            for cell, purpose in cells:
                start_tick = max(world.tick, window.enter_tick - 1)
                lease = ServiceCellLease(
                    cell=cell,
                    purpose=purpose,
                    owner_id=window.actor_id,
                    start_tick=start_tick,
                    end_tick=window.exit_tick,
                    active=start_tick <= world.tick,
                )
                previous = leases.get(cell)
                purpose_priority = {
                    "PATIENT_GATEWAY": 0,
                    "SERVICE_EXIT": 1,
                    "CARGO_ENTRANCE": 2,
                    "QUEUE_NEXT": 3,
                    "CORE_SLOT": 4,
                }
                if previous is None or (
                    lease.start_tick,
                    purpose_priority[lease.purpose],
                    lease.owner_id.bytes,
                ) < (
                    previous.start_tick,
                    purpose_priority[previous.purpose],
                    b"" if previous.owner_id is None else previous.owner_id.bytes,
                ):
                    leases[cell] = lease
        return tuple(sorted(leases.values(), key=lambda row: (row.cell, row.purpose)))

    @staticmethod
    def _service_blockers(
        world: WorldModel,
        leases: tuple[ServiceCellLease, ...],
    ) -> tuple[tuple[UUID, Position, str], ...]:
        lease_by_cell = {lease.cell: lease for lease in leases if lease.active}
        return tuple(
            sorted(
                (
                    (unit.id, unit.position, f"BLOCKS_{lease_by_cell[unit.position].purpose}")
                    for unit in world.friendlies
                    if unit.position in lease_by_cell
                    and unit.id != lease_by_cell[unit.position].owner_id
                    and (world.core is None or unit.position != world.core.position)
                ),
                key=lambda row: (row[1], row[0].bytes),
            )
        )

    def _service_actors(
        self,
        world: WorldModel,
    ) -> tuple[
        tuple[EntitySnapshot, ...],
        tuple[EntitySnapshot, ...],
        tuple[EntitySnapshot, ...],
    ]:
        assert world.core is not None
        all_carriers = tuple(
            sorted(
                (
                    unit
                    for unit in world.friendlies
                    if unit.unit_type is UnitType.WORKER and unit.cargo > 0
                ),
                key=lambda unit: unit.id.bytes,
            )
        )
        carriers = all_carriers
        if self.memory.storage_saturated:
            headroom = max(0, world.resource_capacity - world.resources)
            if headroom <= 0:
                carriers = ()
            elif headroom < self.config.worker_full_storage_release_space:
                # A small spend should be topped up by only the nearest cargo
                # Workers.  Letting every carrier approach the same queue is
                # exactly what produced the live near-Core pile-up.
                carriers = tuple(
                    sorted(
                        all_carriers,
                        key=lambda unit: (
                            manhattan(unit.position, world.core.position),
                            unit.id.bytes,
                        ),
                    )[: self.config.worker_full_storage_replenishers]
                )
        living_carriers = {unit.id for unit in carriers}
        for worker_id in tuple(self.memory.cargo_arrival_ticks):
            if worker_id not in living_carriers:
                self.memory.cargo_arrival_ticks.pop(worker_id, None)
        wounded = tuple(
            sorted(
                (
                    unit
                    for unit in world.friendlies
                    if unit.hp < UNIT_MAX_HP[unit.unit_type]
                ),
                key=lambda unit: (
                    unit.hp * 100 // UNIT_MAX_HP[unit.unit_type],
                    manhattan(unit.position, world.core.position),
                    unit.id.bytes,
                ),
            )
        )
        urgent = tuple(
            unit
            for unit in wounded
            if unit.hp * 100
            <= UNIT_MAX_HP[unit.unit_type] * self.config.recovery_urgent_percent
        )
        return carriers, wounded, urgent

    @staticmethod
    def _emergency_funding_carrier(
        world: WorldModel,
        carriers: tuple[EntitySnapshot, ...],
        urgent: tuple[EntitySnapshot, ...],
    ) -> UUID | None:
        """Permit one side-adjacent carrier to fund an urgent heal.

        Ordinary delivery is deliberately restricted to the selected ingress
        lane.  The narrow exception retains treatment liveness without
        turning every Core neighbor into a competing queue head.
        """

        if world.core is None or not urgent:
            return None
        patient = urgent[0]
        missing = UNIT_MAX_HP[patient.unit_type] - patient.hp
        if world.resources >= missing:
            return None
        adjacent = tuple(
            carrier
            for carrier in carriers
            if manhattan(carrier.position, world.core.position) == 1
        )
        return (
            min(adjacent, key=lambda unit: unit.id.bytes).id
            if adjacent
            else None
        )

    def _sync_storage_saturation(self, world: WorldModel) -> None:
        """Maintain a small hysteresis around a completely full Core."""

        if world.resources >= world.resource_capacity:
            self.memory.storage_saturated = True
        elif (
            self.memory.storage_saturated
            and world.resource_capacity - world.resources
            >= self.config.worker_full_storage_release_space
        ):
            self.memory.storage_saturated = False
            self.memory.worker_home_guard_targets.clear()

    def _sync_ready_line(
        self,
        world: WorldModel,
        carriers: tuple[EntitySnapshot, ...],
        queue_cells: tuple[Position, ...],
        *,
        emergency_ready_id: UUID | None = None,
    ) -> tuple[
        tuple[EntitySnapshot, ...],
        tuple[EntitySnapshot, ...],
        dict[Position, int],
        EntitySnapshot | None,
    ]:
        assert world.core is not None
        # Only the directed ingress lane owns admission.  Treating all four
        # Core neighbors as ready positions let a large cargo wave occupy the
        # dedicated exit and every alternate egress cell at once.
        ready_positions = frozenset((world.core.position, *queue_cells))
        for carrier in carriers:
            if (
                carrier.position in ready_positions
                or carrier.id == emergency_ready_id
            ):
                self.memory.cargo_arrival_ticks.setdefault(carrier.id, world.tick)
            else:
                self.memory.cargo_arrival_ticks.pop(carrier.id, None)
        ready = tuple(
            sorted(
                (
                    carrier
                    for carrier in carriers
                    if carrier.id in self.memory.cargo_arrival_ticks
                ),
                key=lambda unit: (
                    self.memory.cargo_arrival_ticks[unit.id],
                    manhattan(unit.position, world.core.position),
                    unit.id.bytes,
                ),
            )
        )
        outer = queue_cells[-1] if queue_cells else world.core.position
        approaching = tuple(
            sorted(
                (
                    carrier
                    for carrier in carriers
                    if carrier.id not in self.memory.cargo_arrival_ticks
                ),
                key=lambda unit: (
                    self.memory.service_deposit_ticks.get(unit.id, world.tick),
                    self.memory.service_cargo_first_seen_ticks.get(unit.id, world.tick),
                    manhattan(unit.position, outer),
                    unit.id.bytes,
                ),
            )
        )
        lane_index = {cell: index for index, cell in enumerate(queue_cells)}

        def service_depth(unit: EntitySnapshot) -> int:
            if unit.position == world.core.position:
                return -2
            if manhattan(unit.position, world.core.position) == 1:
                return -1
            return lane_index.get(unit.position, len(queue_cells))

        head = min(
            ready,
            key=lambda unit: (
                service_depth(unit),
                self.memory.cargo_arrival_ticks[unit.id],
                unit.id.bytes,
            ),
            default=None,
        )
        return ready, approaching, lane_index, head

    @staticmethod
    def _existing_is_front(
        worker_id: UUID | None,
        ready: tuple[EntitySnapshot, ...],
        ready_ids: set[UUID],
        lane_index: dict[Position, int],
    ) -> bool:
        if worker_id not in ready_ids:
            return False
        actor = next(unit for unit in ready if unit.id == worker_id)
        actor_index = lane_index.get(actor.position, -1)
        return not any(
            lane_index.get(unit.position, -1) < actor_index for unit in ready
        )

    def _select_admission(
        self,
        world: WorldModel,
        service_core_position: Position,
        previous_admission: UUID | None,
        carriers: tuple[EntitySnapshot, ...],
        wounded: tuple[EntitySnapshot, ...],
        urgent: tuple[EntitySnapshot, ...],
        ready: tuple[EntitySnapshot, ...],
        ready_ids: set[UUID],
        lane_index: dict[Position, int],
        head: EntitySnapshot | None,
        units_by_id: dict[UUID, EntitySnapshot],
        patient_gateway: Position | None,
        projection: TacticalMap,
    ) -> tuple[UUID | None, str, int]:
        admission: UUID | None = None
        service = "IDLE"
        reserved = 0
        patient = urgent[0] if urgent else None
        core_occupant = next(
            (
                unit
                for unit in units_by_id.values()
                if unit.position == service_core_position
            ),
            None,
        )
        core_carrier = next(
            (
                carrier
                for carrier in ready
                if carrier.position == service_core_position
                and world.resources < world.resource_capacity
            ),
            None,
        )
        executable_head = (
            head
            if head is not None
            and manhattan(head.position, service_core_position) <= 1
            else None
        )
        if core_carrier is not None:
            admission = core_carrier.id
            service = "DEPOSIT"
        elif (
            core_occupant is not None
            and core_occupant.hp < UNIT_MAX_HP[core_occupant.unit_type]
            and world.resources
            >= UNIT_MAX_HP[core_occupant.unit_type] - core_occupant.hp
        ):
            # The physical occupant is authoritative.  A future patient or
            # adjacent carrier cannot displace a heal that is executable now.
            admission = core_occupant.id
            missing = UNIT_MAX_HP[core_occupant.unit_type] - core_occupant.hp
            reserved = missing
            service = (
                "EMERGENCY_HEAL"
                if core_occupant.hp * 2 <= UNIT_MAX_HP[core_occupant.unit_type]
                else "MAINTENANCE_HEAL"
            )
        elif patient is not None:
            missing = UNIT_MAX_HP[patient.unit_type] - patient.hp
            reserved = missing
            if (
                patient.unit_type is UnitType.WORKER
                and patient.cargo > 0
                and world.resources < world.resource_capacity
            ):
                if (
                    patient.id in ready_ids
                    and manhattan(patient.position, service_core_position) <= 1
                ):
                    admission = (
                        executable_head.id
                        if executable_head is not None
                        and executable_head.id != patient.id
                        else patient.id
                    )
                    service = "WOUNDED_CARGO_DEPOSIT"
                elif executable_head is not None:
                    admission = (
                        previous_admission
                        if self._existing_is_front(
                            previous_admission,
                            ready,
                            ready_ids,
                            lane_index,
                        )
                        else executable_head.id
                    )
                    service = "HEAL_FUNDING"
                else:
                    service = "EMERGENCY_APPROACH"
            elif world.resources >= missing:
                # Funding is reserved immediately, but the physical Core slot
                # is admitted only when the patient can actually enter this
                # Tick.  A distant patient keeps approaching through its
                # dedicated gateway while ready cargo continues to flow.
                patient_can_enter = (
                    patient.position == service_core_position
                    or manhattan(patient.position, service_core_position) == 1
                    and projection.immediate_attackers(service_core_position) < patient.hp
                )
                if patient_can_enter:
                    admission = patient.id
                    service = "EMERGENCY_HEAL"
                elif executable_head is not None:
                    admission = (
                        previous_admission
                        if self._existing_is_front(
                            previous_admission,
                            ready,
                            ready_ids,
                            lane_index,
                        )
                        else executable_head.id
                    )
                    service = "DEPOSIT_BEFORE_PATIENT"
                else:
                    service = "EMERGENCY_APPROACH"
            elif executable_head is not None:
                admission = (
                    previous_admission
                    if self._existing_is_front(
                        previous_admission,
                        ready,
                        ready_ids,
                        lane_index,
                    )
                    else executable_head.id
                )
                service = "HEAL_FUNDING"
            else:
                service = "RECOVERY_WAITING_FOR_FUNDS"

        if admission is None and previous_admission is not None:
            previous_unit = units_by_id.get(previous_admission)
            if (
                previous_unit is not None
                and manhattan(previous_unit.position, service_core_position) <= 1
                and self._existing_is_front(
                    previous_admission,
                    ready,
                    ready_ids,
                    lane_index,
                )
            ):
                admission = previous_admission
                service = "DEPOSIT"
        if (
            admission is None
            and head is not None
            and manhattan(head.position, service_core_position) <= 1
        ):
            admission = head.id
            service = "DEPOSIT"
        elif admission is None and wounded and patient is None:
            maintenance = wounded[0]
            missing = UNIT_MAX_HP[maintenance.unit_type] - maintenance.hp
            reserved = max(reserved, missing)
            if (
                world.resources >= missing
                and manhattan(maintenance.position, service_core_position) <= 1
            ):
                admission = maintenance.id
                service = "MAINTENANCE_HEAL"
            else:
                service = "MAINTENANCE_APPROACH"
        if admission is None and carriers and service == "IDLE":
            service = "DEPOSIT_APPROACH"
        return admission, service, reserved

    def _operation_timeline(
        self,
        world: WorldModel,
        projection: TacticalMap,
        admission_id: UUID | None,
        service: str,
        patient_progress: PatientAdmissionProgress | None,
        ready: tuple[EntitySnapshot, ...],
        head: EntitySnapshot | None,
        reserved_resources: int,
        exit_cell: Position | None,
        patient_gateway: Position | None,
        carriers: tuple[EntitySnapshot, ...],
        return_reservations: tuple[CargoReturnReservation, ...],
    ) -> CoreOperationTimeline:
        assert world.core is not None
        core = world.core
        units = {unit.id: unit for unit in world.friendlies}
        requests: list[CoreOperationRequest] = []
        patient = (
            None
            if patient_progress is None
            else units.get(patient_progress.patient_id)
        )
        if patient is not None and patient_progress.entry_distance is not None:
            missing = UNIT_MAX_HP[patient.unit_type] - patient.hp
            eta = patient_progress.entry_distance
            if (
                eta == 0
                and service in {
                    "DEPOSIT",
                    "DEPOSIT_BEFORE_PATIENT",
                    "WOUNDED_CARGO_DEPOSIT",
                }
            ):
                eta = 1
            requests.append(
                CoreOperationRequest(
                    actor_id=patient.id,
                    operation="HEAL",
                    eta=eta,
                    occupy_tick=world.tick + eta,
                    release_tick=world.tick + eta + 1,
                    priority=10,
                    resource_cost=missing,
                    gateway=patient_gateway,
                )
            )
        # Deposits and treatment share the same physical Core slot.  Materialize
        # every cargo appointment on the same future timeline so Core production
        # can use free Ticks without stealing the Tick on which a Worker is due.
        # A treatment appointment shifts colliding deposits but does not discard
        # their stable order.
        patient_ticks = {
            request.occupy_tick
            for request in requests
            if request.operation == "HEAL"
        }
        carriers_by_id = {carrier.id: carrier for carrier in carriers}
        for reservation in return_reservations:
            carrier = carriers_by_id.get(reservation.worker_id)
            assigned = reservation.scheduled_deposit_tick
            if carrier is None or assigned is None:
                continue
            # Treatment owns a colliding physical window.  Delay the calendar
            # entry rather than normalising an unreachable appointment to the
            # current Tick.  The next service plan will persist this shift.
            while assigned in patient_ticks or assigned - 1 in patient_ticks:
                assigned += 1
            self.memory.service_deposit_ticks[carrier.id] = assigned
            occupy_tick = (
                world.tick
                if carrier.position == core.position
                else assigned - 1
            )
            requests.append(
                CoreOperationRequest(
                    actor_id=carrier.id,
                    operation="DEPOSIT",
                    eta=occupy_tick - world.tick,
                    occupy_tick=occupy_tick,
                    release_tick=assigned + 1,
                    priority=20,
                    resource_gain=carrier.cargo,
                    gateway=self.memory.service_entrance,
                )
            )
        requests.sort(
            key=lambda item: (
                item.eta,
                item.priority,
                b"" if item.actor_id is None else item.actor_id.bytes,
            )
        )
        current_occupant = next(
            (unit.id for unit in world.friendlies if unit.position == core.position),
            None,
        )
        occupant_unit = (
            units.get(current_occupant)
            if current_occupant is not None
            else None
        )
        admission = units.get(admission_id) if admission_id is not None else None
        incoming_cargo = any(
            row.status == "RETURNING"
            and row.first_position == core.position
            and row.departure_tick is not None
            and row.departure_tick <= world.tick
            for row in return_reservations
        )
        incoming_patient = bool(
            patient_progress is not None
            and patient_progress.entry_distance is not None
            and patient_progress.entry_distance <= 1
        )
        current_slot_reserved = bool(
            current_occupant is not None
            or incoming_cargo
            or incoming_patient
        )
        next_request = requests[0] if requests else None
        protected_next = {
            cell
            for cell in (self.memory.service_entrance, *self.memory.service_queue_cells)
            if cell is not None
        }
        egress_rows = tuple(
            cell
            for _, cell in cardinal_neighbors(core.position)
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in projection.hostile_occupied
            and cell not in protected_next
            and projection.immediate_attackers(cell) == 0
        )
        occupant_can_egress = bool(
            occupant_unit is not None
            and occupant_unit.id != admission_id
            and (
                occupant_unit.unit_type is not UnitType.WORKER
                or occupant_unit.cargo == 0
            )
            and egress_rows
        )
        production_allowed = (
            (not current_slot_reserved or occupant_can_egress)
            and bool(egress_rows)
        )
        reason = "SAFE_BEFORE_FUTURE_SERVICE"
        if not egress_rows:
            production_allowed = False
            reason = "NO_SAFE_SPAWN_EGRESS"
        elif next_request is not None and next_request.eta <= 1:
            # A service actor adjacent to the Core resolves movement before
            # heal/deposit in this Tick, so a spawn would contend for the same
            # two-entity cell even if an exit exists for the following Tick.
            production_allowed = False
            reason = "SERVICE_DUE_THIS_TICK"
        elif current_slot_reserved and occupant_can_egress:
            reason = "CURRENT_OCCUPANT_CAN_EGRESS"
        elif current_slot_reserved:
            reason = "CURRENT_TICK_SERVICE_SLOT"
        return CoreOperationTimeline(
            tick=world.tick,
            requests=tuple(requests),
            current_slot_owner=current_occupant,
            current_slot_reserved=current_slot_reserved,
            next_service_eta=None if next_request is None else next_request.eta,
            next_service_tick=None if next_request is None else next_request.occupy_tick,
            next_release_tick=None if next_request is None else next_request.release_tick,
            production_allowed=production_allowed,
            spawn_egress_cell=min(egress_rows, default=exit_cell),
            reason=reason,
        )

    @staticmethod
    def _queue_slots(
        core_position: Position,
        ready: tuple[EntitySnapshot, ...],
        approaching: tuple[EntitySnapshot, ...],
        lane_index: dict[Position, int],
        queue_cells: tuple[Position, ...],
        exit_cell: Position | None,
        admission_id: UUID | None,
        allow_advance: bool,
    ) -> tuple[tuple[UUID, Position], ...]:
        rows: list[tuple[UUID, Position]] = []
        for carrier in ready:
            target = carrier.position
            index = lane_index.get(carrier.position)
            if allow_advance and carrier.id == admission_id:
                if carrier.position == core_position:
                    target = core_position
                elif index is not None and index > 0:
                    target = queue_cells[index - 1]
                else:
                    target = core_position
            elif carrier.position == core_position and carrier.id != admission_id:
                target = queue_cells[0] if queue_cells else (exit_cell or core_position)
            elif allow_advance and index is not None and index > 0:
                target = queue_cells[index - 1]
            rows.append((carrier.id, target))
        outer = queue_cells[-1] if queue_cells else core_position
        rows.extend((carrier.id, outer) for carrier in approaching)
        return tuple(rows)

    def combat_egress_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        queue: CoreServiceQueue,
    ) -> list[ActionIntent]:
        """Release combat units born or waiting on the Core service slot."""

        if world.core is None:
            return []
        intents: list[ActionIntent] = []
        occupied = dict(world.occupied_cells)
        protected = {cell for cell in (queue.entrance, *queue.queue_cells) if cell is not None}
        for unit in world.friendlies:
            if (
                unit.unit_type not in {UnitType.VANGUARD, UnitType.RANGER}
                or unit.position != world.core.position
                or queue.admission_id == unit.id
            ):
                continue
            preferred = queue.exit_cell
            rows = []
            for index, (direction, destination) in enumerate(cardinal_neighbors(unit.position)):
                if destination in world.known_obstacles or destination in projection.hostile_occupied:
                    continue
                if destination in protected and destination != preferred:
                    continue
                score = (
                    int(destination != preferred),
                    projection.immediate_attackers(destination),
                    projection.future_attackers(destination),
                    occupied.get(destination, 0),
                    index,
                )
                rows.append((score, direction, destination))
            intents.extend(
                ActionIntent.move(
                    unit.id,
                    UnitMission.CLEAR_CORE,
                    60,
                    direction,
                    destination,
                    risk=score[1] * 100 + score[2] * 10,
                    exclusive_destination=True,
                    tie_break=score,
                    reason=(
                        "CORE_SERVICE_EXIT"
                        if destination == preferred
                        else "CORE_EXIT_ALTERNATE"
                    ),
                    metadata=(
                        ("allow_protected", destination == preferred),
                        ("allow_head_on_swap", True),
                    ),
                )
                for score, direction, destination in sorted(rows)
            )
            intents.append(
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    UnitMission.CLEAR_CORE,
                    61,
                    reason="CORE_EXIT_BLOCKED_THIS_TICK" if rows else "CORE_EXIT_BLOCKED",
                )
            )
        return intents

    def _choose_lane(
        self,
        world: WorldModel,
        projection: TacticalMap,
        carriers: tuple,
        *,
        core_position: Position | None = None,
        force_replan: bool = False,
    ) -> tuple[Position | None, tuple[Position, ...], Position | None]:
        assert world.core is not None
        core = core_position or world.core.destination or world.core.position
        enemy_positions = projection.hostile_occupied
        occupied = dict(world.occupied_cells)
        existing = self.memory.service_queue_cells
        existing_reachable = tuple(
            cargo_return_route(
                world,
                projection,
                carrier,
                core,
                existing,
                node_limit=self.config.path_node_limit,
                exit_cell=self.memory.service_exit_cell,
            )
            for carrier in carriers
        )
        if (
            not force_replan
            and self._lane_valid(core, existing, world, projection)
            and (
                not carriers
                or any(row.route_distance is not None for row in existing_reachable)
            )
        ):
            entrance = existing[0]
            exit_cell = self.memory.service_exit_cell
            if exit_cell is not None and (
                exit_cell in world.known_obstacles
                or exit_cell in enemy_positions
                or exit_cell == entrance
            ):
                exit_cell = None
            if exit_cell is None:
                exit_cell = self._choose_exit(core, entrance, world, projection)
            return entrance, existing, exit_cell

        paths: list[tuple[Position, ...]] = []
        for _, entrance in cardinal_neighbors(core):
            if (
                entrance not in world.known_passable
                or entrance in world.known_obstacles
                or entrance in enemy_positions
            ):
                continue
            self._extend_lane(
                world,
                projection,
                core,
                (entrance,),
                paths,
            )
        if not paths:
            # A migrated Core can sit at the end of a narrow pocket where a
            # two-cell queue has no outside entrance.  A one-cell lane is
            # still useful when it can be approached from a different side;
            # retaining an inward-facing cul-de-sac would strand every remote
            # carrier behind the protected inner cell.
            for _, entrance in cardinal_neighbors(core):
                lane = (entrance,)
                if self._lane_valid(core, lane, world, projection):
                    paths.append(lane)
        if not paths:
            self.memory.service_entrance = None
            self.memory.service_queue_cells = ()
            self.memory.service_exit_cell = None
            return None, (), None

        def path_score(path: tuple[Position, ...]) -> tuple[int, ...]:
            candidate_exit = self._choose_exit(core, path[0], world, projection)
            carrier_routes = tuple(
                cargo_return_route(
                    world,
                    projection,
                    carrier,
                    core,
                    path,
                    node_limit=self.config.path_node_limit,
                    exit_cell=candidate_exit,
                )
                for carrier in carriers
            )
            reachable = tuple(
                row.route_distance
                for row in carrier_routes
                if row.route_distance is not None
            )
            bends = sum(
                (path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1])
                != (path[index - 1][0] - (core if index == 1 else path[index - 2])[0],
                    path[index - 1][1] - (core if index == 1 else path[index - 2])[1])
                for index in range(1, len(path))
            )
            return (
                -len(reachable),
                sum(projection.immediate_attackers(cell) for cell in path),
                sum(projection.future_attackers(cell) for cell in path),
                sum(projection.remembered_danger.get(cell, 0) for cell in path),
                sum(max(0, occupied.get(cell, 0) - 1) for cell in path),
                sum(reachable),
                bends,
                path,
            )

        chosen = min(paths, key=path_score)
        entrance = chosen[0]
        exit_cell = self._choose_exit(core, entrance, world, projection)
        self.memory.service_entrance = entrance
        self.memory.service_queue_cells = chosen
        self.memory.service_exit_cell = exit_cell
        return entrance, chosen, exit_cell

    def _extend_lane(
        self,
        world: WorldModel,
        projection: TacticalMap,
        core: Position,
        path: tuple[Position, ...],
        output: list[tuple[Position, ...]],
    ) -> None:
        if len(path) >= self.config.service_lane_depth:
            if self._lane_has_external_approach(core, path, world, projection):
                output.append(path)
            return
        for direction in DIRECTION_ORDER:
            candidate = add_direction(path[-1], direction)
            if (
                candidate == core
                or candidate in path
                or candidate not in world.known_passable
                or candidate in world.known_obstacles
                or candidate in projection.hostile_occupied
            ):
                continue
            self._extend_lane(world, projection, core, (*path, candidate), output)

    @staticmethod
    def _lane_valid(
        core: Position,
        lane: tuple[Position, ...],
        world: WorldModel,
        projection: TacticalMap,
    ) -> bool:
        if not lane or manhattan(core, lane[0]) != 1:
            return False
        previous = core
        for cell in lane:
            if (
                manhattan(previous, cell) != 1
                or cell not in world.known_passable
                or cell in world.known_obstacles
                or cell in projection.hostile_occupied
                or projection.immediate_attackers(cell)
            ):
                return False
            previous = cell
        return CoreServicePlanner._lane_has_external_approach(
            core,
            lane,
            world,
            projection,
        )

    @staticmethod
    def _lane_has_external_approach(
        core: Position,
        lane: tuple[Position, ...],
        world: WorldModel,
        projection: TacticalMap,
    ) -> bool:
        """Require a queue tail that cargo can enter without crossing the line.

        Merely checking that every lane cell is passable accepts a cul-de-sac
        whose only neighbor is the preceding protected queue cell.  Remote
        Workers are intentionally forbidden from cutting through that cell,
        so such a lane is topologically unusable even though each cell is
        individually legal.
        """

        if not lane:
            return False
        infrastructure = frozenset((core, *lane))
        return any(
            candidate not in infrastructure
            and candidate in world.known_passable
            and candidate not in world.known_obstacles
            and candidate not in projection.hostile_occupied
            and projection.immediate_attackers(candidate) == 0
            for _, candidate in cardinal_neighbors(lane[-1])
        )

    @staticmethod
    def _choose_exit(
        core: Position,
        entrance: Position,
        world: WorldModel,
        projection: TacticalMap,
    ) -> Position:
        resources = set(world.visible_resources)
        resources.update(position for position, _ in world.remembered_resources)
        options = [
            (
                projection.immediate_attackers(cell),
                projection.future_attackers(cell),
                int(cell in resources),
                dict(world.occupied_cells).get(cell, 0),
                index,
                cell,
            )
            for index, (_, cell) in enumerate(cardinal_neighbors(core))
            if cell != entrance
            and cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in projection.hostile_occupied
        ]
        return min(options)[-1] if options else entrance

    def _clear_admission(self) -> None:
        self.memory.service_admission_id = None
        self.memory.service_kind = None
        self.memory.cargo_arrival_ticks.clear()
        self.memory.patient_admission_progress = None
        self.memory.service_worker_progress.clear()

    @staticmethod
    def _release_reason(previous, admission, units, ready_ids, head) -> str | None:
        if previous is None or previous == admission:
            return None
        previous_unit = units.get(previous)
        if previous_unit is None:
            return "ACTOR_GONE"
        if previous_unit.unit_type is UnitType.WORKER and previous_unit.cargo == 0:
            return "CARGO_RELEASED"
        if previous_unit.unit_type is UnitType.WORKER and previous not in ready_ids:
            return "LEFT_READY_LINE"
        if head is not None and admission == head.id:
            return "FRONT_WORKER_OVERRIDE"
        return "SERVICE_RESELECTED"

    @staticmethod
    def _admission_reason(previous, admission, service, ready, head) -> str | None:
        if admission is None:
            return None
        if admission == previous:
            return "PERSISTED_READY"
        if service == "EMERGENCY_HEAL":
            return "EMERGENCY_HEAL_PRIORITY"
        if service == "MAINTENANCE_HEAL":
            return "MAINTENANCE_HEAL"
        if head is not None and admission == head.id:
            fifo = ready[0] if ready else None
            return "FRONT_WORKER_OVERRIDE" if fifo is not None and fifo.id != head.id else "READY_FIFO"
        return "SERVICE_SELECTED"
