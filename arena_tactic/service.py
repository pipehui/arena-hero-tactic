from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from arena_hero import CoreState, Position, UnitType

from .config import TacticConfig
from .geometry import DIRECTION_ORDER, add_direction, cardinal_neighbors, manhattan
from .models import (
    ActionIntent,
    CoreServiceQueue,
    EntitySnapshot,
    IntentAction,
    UnitMission,
    WorldModel,
)
from .projection import TacticalMap
from .rules import UNIT_MAX_HP
from .state import TacticMemory


_SERVICE_MISSIONS = frozenset(
    {
        UnitMission.DEPOSIT,
        UnitMission.RETURN_CARGO,
        UnitMission.CLEAR_CORE,
        UnitMission.RECOVER,
    }
)


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
        )
        if previous_lane != (entrance, queue_cells):
            self.memory.cargo_arrival_ticks.clear()

        ready, approaching, lane_index, head = self._sync_ready_line(
            world,
            carriers,
            queue_cells,
        )
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
                approaching_depositors=tuple(unit.id for unit in (*ready, *approaching)),
                queue_slots=tuple((unit.id, target) for unit in (*ready, *approaching)),
                wounded=tuple(unit.id for unit in wounded),
                entrance=entrance,
                queue_cells=queue_cells,
                exit_cell=exit_cell,
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
        threatened = any(projection.immediate_attackers(cell) for cell in lane_cells)
        units_by_id = {unit.id: unit for unit in world.friendlies}
        admission, service, reserved = self._select_admission(
            world,
            service_core_position,
            previous_admission,
            carriers,
            wounded,
            urgent,
            ready,
            ready_ids,
            lane_index,
            head,
            units_by_id,
        )
        if (
            self.memory.storage_saturated
            and admission is None
            and service in {"IDLE", "DEPOSIT_APPROACH"}
        ):
            service = "STORAGE_SATURATED_HOME_GUARD"
        reserved = max(
            reserved,
            sum(
                UNIT_MAX_HP[UnitType.WORKER] - unit.hp
                for unit in wounded
                if unit.unit_type is UnitType.WORKER
            ),
        )

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
        allow_advance = not threatened and service in {
            "DEPOSIT",
            "HEAL_FUNDING",
            "WOUNDED_CARGO_DEPOSIT",
        }
        return CoreServiceQueue(
            service="PAUSED" if threatened else service,
            admission_id=admission,
            service_core_position=service_core_position,
            depositors=tuple(unit.id for unit in (*ready, *approaching)),
            ready_depositors=tuple(unit.id for unit in ready),
            approaching_depositors=tuple(unit.id for unit in approaching),
            ready_ticks=tuple(
                (unit.id, self.memory.cargo_arrival_ticks[unit.id]) for unit in ready
            ),
            queue_slots=self._queue_slots(
                core.position,
                ready,
                approaching,
                lane_index,
                queue_cells,
                exit_cell,
                admission,
                allow_advance,
            ),
            wounded=tuple(unit.id for unit in wounded),
            entrance=entrance,
            queue_cells=queue_cells,
            exit_cell=exit_cell,
            reserved_resources=reserved,
            paused_reason="LANE_THREATENED" if threatened else None,
            previous_admission_id=previous_admission,
            admission_reason=admission_reason,
            release_reason=release_reason,
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
    ) -> tuple[
        tuple[EntitySnapshot, ...],
        tuple[EntitySnapshot, ...],
        dict[Position, int],
        EntitySnapshot | None,
    ]:
        assert world.core is not None
        # A loaded Worker already beside the Core is physically one move from
        # service even when it approached from a side other than the selected
        # queue entrance.  Requiring it to walk around and join the nominal
        # line can deadlock urgent healing: recovery waits for deposited
        # funds while every adjacent carrier waits for an admission that only
        # queue-line members could previously receive.
        ready_positions = frozenset(
            (
                world.core.position,
                *queue_cells,
                *(cell for _, cell in cardinal_neighbors(world.core.position)),
            )
        )
        for carrier in carriers:
            if carrier.position in ready_positions:
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
    ) -> tuple[UUID | None, str, int]:
        admission: UUID | None = None
        service = "IDLE"
        reserved = 0
        patient = urgent[0] if urgent else None
        core_carrier = next(
            (
                carrier
                for carrier in ready
                if carrier.position == service_core_position
                and world.resources < world.resource_capacity
            ),
            None,
        )
        if core_carrier is not None:
            admission = core_carrier.id
            service = "DEPOSIT"
        elif patient is not None:
            missing = UNIT_MAX_HP[patient.unit_type] - patient.hp
            reserved = missing
            if (
                patient.unit_type is UnitType.WORKER
                and patient.cargo > 0
                and world.resources < world.resource_capacity
            ):
                if patient.id in ready_ids:
                    admission = (
                        head.id
                        if head is not None and head.id != patient.id
                        else patient.id
                    )
                    service = "WOUNDED_CARGO_DEPOSIT"
                elif head is not None:
                    admission = (
                        previous_admission
                        if self._existing_is_front(
                            previous_admission,
                            ready,
                            ready_ids,
                            lane_index,
                        )
                        else head.id
                    )
                    service = "HEAL_FUNDING"
                else:
                    service = "EMERGENCY_APPROACH"
            elif world.resources >= missing:
                if manhattan(
                    patient.position,
                    service_core_position,
                ) <= self.config.service_patient_ready_radius:
                    admission = patient.id
                    service = "EMERGENCY_HEAL"
                else:
                    service = "EMERGENCY_APPROACH"
            elif head is not None:
                admission = (
                    previous_admission
                    if self._existing_is_front(
                        previous_admission,
                        ready,
                        ready_ids,
                        lane_index,
                    )
                    else head.id
                )
                service = "HEAL_FUNDING"
            else:
                service = "RECOVERY_WAITING_FOR_FUNDS"

        if admission is None and previous_admission is not None:
            previous_unit = units_by_id.get(previous_admission)
            if previous_unit is not None and self._existing_is_front(
                previous_admission,
                ready,
                ready_ids,
                lane_index,
            ):
                admission = previous_admission
                service = "DEPOSIT"
        if admission is None and head is not None:
            admission = head.id
            service = "DEPOSIT"
        elif admission is None and wounded and patient is None:
            maintenance = wounded[0]
            missing = UNIT_MAX_HP[maintenance.unit_type] - maintenance.hp
            reserved = max(reserved, missing)
            if world.resources >= missing:
                admission = maintenance.id
                service = "MAINTENANCE_HEAL"
        if admission is None and carriers and service == "IDLE":
            service = "DEPOSIT_APPROACH"
        return admission, service, reserved

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
    ) -> tuple[Position | None, tuple[Position, ...], Position | None]:
        assert world.core is not None
        core = core_position or world.core.destination or world.core.position
        enemy_positions = projection.hostile_occupied
        occupied = dict(world.occupied_cells)
        existing = self.memory.service_queue_cells
        if self._lane_valid(core, existing, world, projection):
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
            bends = sum(
                (path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1])
                != (path[index - 1][0] - (core if index == 1 else path[index - 2])[0],
                    path[index - 1][1] - (core if index == 1 else path[index - 2])[1])
                for index in range(1, len(path))
            )
            return (
                sum(projection.immediate_attackers(cell) for cell in path),
                sum(projection.future_attackers(cell) for cell in path),
                sum(projection.remembered_danger.get(cell, 0) for cell in path),
                sum(max(0, occupied.get(cell, 0) - 1) for cell in path),
                min(
                    (manhattan(carrier.position, path[-1]) for carrier in carriers),
                    default=0,
                ),
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
        options = [
            (
                projection.immediate_attackers(cell),
                projection.future_attackers(cell),
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
