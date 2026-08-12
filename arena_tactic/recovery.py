from __future__ import annotations

from arena_hero import CoreState, Position, UnitType

from .config import TacticConfig
from .geometry import cardinal_neighbors, count_open_neighbors, manhattan, manhattan_ring
from .models import (
    ActionIntent,
    CoreServiceQueue,
    EntitySnapshot,
    IntentAction,
    MissionState,
    UnitMission,
    WorldModel,
)
from .planning import move_viability, weighted_route_to
from .projection import TacticalMap
from .rules import UNIT_MAX_HP
from .service import service_protected_positions
from .state import TacticMemory
from .worker_safety import WorkerSafetyEvaluator


class RecoveryPlanner:
    """Keep every wounded combat unit in a stable recovery workflow."""

    def __init__(
        self,
        config: TacticConfig,
        memory: TacticMemory | None = None,
    ) -> None:
        self.config = config
        self.memory = memory or TacticMemory()
        self.worker_safety = WorkerSafetyEvaluator()

    def intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        service: CoreServiceQueue,
    ) -> list[ActionIntent]:
        if world.core is None:
            return []
        units = {unit.id: unit for unit in world.friendlies}
        wounded = tuple(
            unit
            for wounded_id in service.wounded
            if (unit := units.get(wounded_id)) is not None
            and not (unit.unit_type is UnitType.WORKER and unit.cargo > 0)
        )
        protected = service_protected_positions(world, service)
        jobs = {job.actor_id: job for job in service.jobs}
        intents: list[ActionIntent] = []
        reserved_staging: set[Position] = set()
        for unit in wounded:
            if unit.unit_type is UnitType.WORKER:
                existing = self.memory.unit_missions.get(unit.id)
                self.memory.unit_missions[unit.id] = MissionState(
                    UnitMission.RECOVER,
                    world.core.position,
                    (
                        existing.assigned_tick
                        if existing is not None
                        and existing.mission is UnitMission.RECOVER
                        else world.tick
                    ),
                )
            urgent = unit.hp * 2 <= UNIT_MAX_HP[unit.unit_type]
            priority = 40 if urgent else 45
            service_job = jobs.get(unit.id)
            scheduled = bool(
                service_job is not None and "HEAL" in service_job.operations
            )
            admitted = service.admission_id == unit.id
            missing = UNIT_MAX_HP[unit.unit_type] - unit.hp
            if (
                unit.position == world.core.position
                and service_job is not None
                and not admitted
                and service_job.service_tick is not None
                and service_job.service_tick > world.tick
            ):
                # An underfunded patient cannot occupy the only Core Unit slot
                # while the single funding Worker waits outside.  Yield to a
                # safe recovery cell, keep the same HEAL job, then re-enter as
                # soon as the deposit has funded treatment.
                egress_rows = []
                for index, (direction, destination) in enumerate(
                    cardinal_neighbors(unit.position)
                ):
                    if (
                        destination in world.known_obstacles
                        or destination in projection.hostile_occupied
                        or destination == service.entrance
                        or destination in service.queue_cells
                        or projection.immediate_attackers(destination) >= unit.hp
                    ):
                        continue
                    viability = move_viability(
                        world,
                        unit.position,
                        destination,
                        target=service.exit_cell,
                        blocked=frozenset(protected - {unit.position, destination}),
                        node_limit=min(self.config.path_node_limit, 256),
                        require_open_area=True,
                    )
                    if not viability.viable:
                        continue
                    egress_rows.append(
                        (
                            (
                                int(destination != service.exit_cell),
                                projection.future_attackers(destination),
                                projection.threat_heat.get(destination, 0),
                                index,
                            ),
                            direction,
                            destination,
                            viability,
                        )
                    )
                for score, direction, destination, viability in sorted(
                    egress_rows, key=lambda row: row[0]
                ):
                    intents.append(
                        ActionIntent.move(
                            unit.id,
                            UnitMission.RECOVER,
                            priority - 1,
                            direction,
                            destination,
                            risk=self._risk(projection, destination),
                            exclusive_destination=True,
                            tie_break=score,
                            reason="PATIENT_YIELD_FOR_FUNDING",
                            metadata=(
                                ("allow_protected", True),
                                ("allow_head_on_swap", True),
                                ("service_tick", service_job.service_tick),
                            ) + viability.metadata,
                        )
                    )
                if egress_rows:
                    continue
            if (
                admitted
                and service.paused_reason is None
                and unit.position == world.core.position
                and world.core.state is CoreState.NORMAL
            ):
                intents.append(
                    ActionIntent.simple(
                        unit.id,
                        IntentAction.HEAL,
                        UnitMission.RECOVER,
                        priority,
                        resource_cost=missing,
                        reason=service.service,
                    )
                )
                continue

            target = None
            if scheduled and service.paused_reason is None:
                target = (
                    service_job.gateway
                    if service_job is not None
                    and service_job.gateway is not None
                    and unit.position != service_job.gateway
                    else world.core.position
                )
            if target is None:
                target = self._staging_cell(
                    world,
                    projection,
                    unit,
                    protected,
                    reserved_staging,
                    service.service_core_position,
                )
                if target is not None:
                    reserved_staging.add(target)
            route_blocked = False
            if target is not None and unit.position != target:
                blocked = set(projection.hostile_occupied)
                blocked.update(
                    cell
                    for cell, count in projection.occupied_cells.items()
                    if count >= 2
                    and cell not in {unit.position, target, world.core.position}
                )
                allowed_service_cells = {unit.position, target, world.core.position}
                if admitted or scheduled:
                    # An admitted patient must be able to use the same narrow
                    # corridor that cargo normally occupies.  The resolver
                    # still arbitrates capacity and same-Tick handoffs.
                    allowed_service_cells.update(service.queue_cells)
                    if service.entrance is not None:
                        allowed_service_cells.add(service.entrance)
                    if service.exit_cell is not None:
                        allowed_service_cells.add(service.exit_cell)
                    if service.patient_gateway is not None:
                        allowed_service_cells.add(service.patient_gateway)
                blocked.update(
                    cell
                    for cell in protected
                    if cell not in allowed_service_cells
                )
                blocked.update(
                    cell
                    for cell in projection.immediate_damage
                    if projection.immediate_attackers(cell) >= unit.hp
                )
                route_block = frozenset(blocked - {unit.position, target})
                if unit.unit_type is UnitType.WORKER and unit.hp < UNIT_MAX_HP[UnitType.WORKER]:
                    steps = self.worker_safety.recovery_steps(
                        world,
                        projection,
                        unit,
                        target,
                        blocked=route_block,
                        node_limit=self.config.path_node_limit,
                        lookahead_node_limit=self.config.worker_escape_lookahead_nodes,
                    )
                    job_step = next(
                        (
                            step
                            for step in steps
                            if service_job is not None
                            and service_job.first_position == step.destination
                            and step.route_reachable
                            and step.survival_terminals > 0
                            and step.forward_exits >= 2
                            and step.immediate_attackers < unit.hp
                            and step.future_attackers < unit.hp
                            and not any(
                                enemy.visible_now
                                and enemy.unit_type
                                in {UnitType.VANGUARD, UnitType.RANGER}
                                for enemy in projection.enemies
                            )
                        ),
                        None,
                    )
                    if job_step is not None:
                        # The unified calendar and the actor now consume the
                        # same safe first step.  Alternative high-exit moves
                        # remain available if the resolver rejects it, but may
                        # no longer make a calm wounded Worker drift away from
                        # treatment or oscillate between staging cells.
                        intents.append(
                            ActionIntent.move(
                                unit.id,
                                UnitMission.RECOVER,
                                priority - 1,
                                job_step.direction,
                                job_step.destination,
                                risk=self.worker_safety.risk(
                                    projection, job_step.destination
                                ),
                                exclusive_destination=True,
                                tie_break=job_step.score,
                                reason="CORE_SERVICE_ROUTE_ADVANCE",
                                metadata=(
                                    ("allow_protected", True),
                                    ("allow_head_on_swap", True),
                                    ("forward_exits", job_step.forward_exits),
                                    ("survival_terminals", job_step.survival_terminals),
                                    ("service_tick", service_job.service_tick),
                                ),
                            )
                        )
                    for step in steps:
                        intents.append(
                            ActionIntent.move(
                                unit.id,
                                UnitMission.RECOVER,
                                priority,
                                step.direction,
                                step.destination,
                                risk=self.worker_safety.risk(
                                    projection, step.destination
                                ),
                                exclusive_destination=True,
                                tie_break=step.score,
                                reason=(
                                    service.service
                                    if admitted or scheduled
                                    else "WORKER_RECOVERY_SAFE_APPROACH"
                                ),
                                metadata=(
                                    ("allow_protected", admitted or scheduled),
                                    ("allow_head_on_swap", admitted or scheduled),
                                    ("forward_exits", step.forward_exits),
                                    ("survival_terminals", step.survival_terminals),
                                    ("first_step_heat", step.heat),
                                    ("future_attackers", step.future_attackers),
                                    ("route_reachable", step.route_reachable),
                                ),
                            )
                        )
                    route_blocked = not steps
                else:
                    route = weighted_route_to(
                        world,
                        unit.position,
                        target,
                        node_limit=self.config.path_node_limit,
                        blocked=route_block,
                        cell_costs=self.worker_safety.route_costs(projection),
                    )
                    if route is not None and route.first_direction is not None:
                        intents.append(
                            ActionIntent.move(
                                unit.id,
                                UnitMission.RECOVER,
                                priority,
                                route.first_direction,
                                route.first_position,
                                risk=self._risk(projection, route.first_position),
                                exclusive_destination=True,
                                tie_break=(route.distance,),
                                reason=(
                                    service.service
                                    if admitted or scheduled
                                    else "RECOVERY_STAGING_APPROACH"
                                ),
                                metadata=(
                                    ("allow_protected", admitted or scheduled),
                                    ("allow_head_on_swap", admitted or scheduled),
                                ),
                            )
                        )
                    else:
                        route_blocked = True
            intents.append(
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    UnitMission.RECOVER,
                    priority + 1,
                    reason=(
                        f"RECOVERY_PAUSED_{service.paused_reason}"
                        if service.paused_reason is not None
                        else (
                            (
                                "NO_SURVIVABLE_RECOVERY_STEP"
                                if unit.unit_type is UnitType.WORKER
                                else "RECOVERY_ROUTE_BLOCKED_THIS_TICK"
                            )
                            if target is None or route_blocked
                            else "WAITING_FOR_RECOVERY_ENTRY"
                        )
                    ),
                    metadata=(
                        ("staging", target),
                        ("admitted", admitted),
                    ),
                )
            )
        return intents

    def survival_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> list[ActionIntent]:
        if world.core is None:
            return []
        intents: list[ActionIntent] = []
        for unit in world.friendlies:
            if unit.unit_type not in {UnitType.VANGUARD, UnitType.RANGER}:
                continue
            if unit.hp * 2 > UNIT_MAX_HP[unit.unit_type]:
                continue
            current = projection.immediate_attackers(unit.position)
            if current == 0:
                continue
            rows = []
            for index, (direction, destination) in enumerate(cardinal_neighbors(unit.position)):
                if destination in world.known_obstacles or destination in projection.hostile_occupied:
                    continue
                immediate = projection.immediate_attackers(destination)
                if immediate >= unit.hp or immediate >= current:
                    continue
                viability = move_viability(
                    world,
                    unit.position,
                    destination,
                    target=world.core.position,
                    blocked=projection.hostile_occupied,
                    node_limit=min(self.config.path_node_limit, 256),
                    require_open_area=True,
                    terminal_exception=(
                        "CORE_SERVICE"
                        if destination == world.core.position
                        else None
                    ),
                )
                if not viability.viable:
                    continue
                score = (
                    immediate,
                    projection.future_attackers(destination),
                    manhattan(destination, world.core.position),
                    index,
                )
                rows.append((score, direction, destination, viability))
            for score, direction, destination, viability in sorted(
                rows, key=lambda row: row[0]
            ):
                intents.append(
                    ActionIntent.move(
                        unit.id,
                        UnitMission.RECOVER,
                        20,
                        direction,
                        destination,
                        risk=score[0] * 100 + score[1] * 10,
                        exclusive_destination=True,
                        tie_break=score,
                        reason="LETHAL_EXPOSURE_WITHDRAWAL",
                        metadata=(("allow_protected", True),) + viability.metadata,
                    )
                )
        return intents

    @staticmethod
    def _staging_cell(
        world: WorldModel,
        projection: TacticalMap,
        unit: EntitySnapshot,
        protected: frozenset[Position],
        reserved: set[Position],
        service_core_position: Position | None,
    ) -> Position | None:
        assert world.core is not None
        center = service_core_position or world.core.position
        occupied = dict(world.occupied_cells)
        candidates: list[tuple[tuple[int, ...], Position]] = []
        for radius in (2, 3):
            for cell in manhattan_ring(center, radius):
                if (
                    cell not in world.known_passable
                    or cell in world.known_obstacles
                    or cell in projection.hostile_occupied
                    or cell in protected
                    or cell in reserved
                    or projection.immediate_attackers(cell) >= unit.hp
                    or projection.future_attackers(cell) >= unit.hp
                    or count_open_neighbors(cell, world.known_obstacles) < 2
                ):
                    continue
                score = (
                    projection.immediate_attackers(cell),
                    projection.future_attackers(cell),
                    projection.threat_heat.get(cell, 0),
                    occupied.get(cell, 0),
                    manhattan(unit.position, cell),
                    radius,
                    cell[0],
                    cell[1],
                )
                candidates.append((score, cell))
            if candidates:
                break
        return min(candidates)[1] if candidates else None

    @staticmethod
    def _risk(projection: TacticalMap, cell: Position | None) -> int:
        if cell is None:
            return 0
        immediate, future, remembered = projection.exposure(cell)
        return immediate * 100 + future * 10 + remembered
