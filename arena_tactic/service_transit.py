from __future__ import annotations

from collections.abc import Mapping

from arena_hero import Position, UnitType

from .config import TacticConfig
from .models import (
    ActionIntent,
    CoreServiceJob,
    DestinationExclusivity,
    EntitySnapshot,
    ServiceTransitKind,
    ServiceTransitRoute,
    UnitMission,
    WorldModel,
)
from .planning import Route, move_viability, weighted_route_to
from .projection import TacticalMap
from .rules import UNIT_MAX_HP


class CoreServiceTransitPlanner:
    """One movement authority for cargo and patient Core service trips.

    The service calendar decides *when* an actor owns the Core slot.  This
    planner decides *how* it reaches the gateway and emits the same ranked
    first-step shape for Workers and wounded combat units.  The resolver owns
    final occupancy arbitration, including the narrow patient-through-guard
    exception.
    """

    def __init__(self, config: TacticConfig) -> None:
        self.config = config

    @staticmethod
    def kind_for(
        job: CoreServiceJob | None,
        unit: EntitySnapshot,
    ) -> ServiceTransitKind:
        operations = () if job is None else job.operations
        if "DEPOSIT" in operations and "HEAL" in operations:
            return ServiceTransitKind.DEPOSIT_THEN_HEAL
        if "DEPOSIT" in operations or (
            unit.unit_type is UnitType.WORKER and unit.cargo > 0
        ):
            return ServiceTransitKind.DEPOSIT
        return ServiceTransitKind.HEAL

    def routes(
        self,
        world: WorldModel,
        unit: EntitySnapshot,
        target: Position,
        *,
        blocked: frozenset[Position],
        cell_costs: Mapping[Position, int] | None = None,
        preferred: Route | None = None,
        max_options: int = 3,
    ) -> tuple[Route, ...]:
        options: list[Route] = []
        first_steps: set[Position] = set()
        if (
            preferred is not None
            and preferred.first_direction is not None
            and preferred.first_position is not None
        ):
            options.append(preferred)
            first_steps.add(preferred.first_position)
        while len(options) < max_options:
            route = weighted_route_to(
                world,
                unit.position,
                target,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(set(blocked) | first_steps),
                cell_costs=cell_costs,
            )
            if (
                route is None
                or route.first_direction is None
                or route.first_position is None
                or route.first_position in first_steps
            ):
                break
            options.append(route)
            first_steps.add(route.first_position)
        return tuple(options)

    def intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        unit: EntitySnapshot,
        target: Position,
        routes: tuple[Route, ...],
        *,
        mission: UnitMission,
        priority: int,
        reason: str,
        kind: ServiceTransitKind,
        job: CoreServiceJob | None = None,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> tuple[ServiceTransitRoute, list[ActionIntent], int]:
        assert world.core is not None
        intents: list[ActionIntent] = []
        rejected = 0
        recorded: list[tuple] = []
        for rank, route in enumerate(routes[:3]):
            destination = route.first_position
            direction = route.first_direction
            if destination is None or direction is None:
                continue
            if (
                destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or projection.immediate_attackers(destination) >= unit.hp
                or projection.future_attackers(destination) >= unit.hp
            ):
                rejected += 1
                continue
            terminal = "CORE_SERVICE" if destination == world.core.position else None
            viability = move_viability(
                world,
                unit.position,
                destination,
                target=None,
                node_limit=min(self.config.path_node_limit, 64),
                require_continuation=False,
                require_open_area=False,
                terminal_exception=terminal,
            )
            if not viability.viable:
                rejected += 1
                continue
            exclusivity = DestinationExclusivity.NONE
            if destination == world.core.position:
                exclusivity = DestinationExclusivity.PHYSICAL
            elif (
                kind in {ServiceTransitKind.HEAL, ServiceTransitKind.DEPOSIT_THEN_HEAL}
                and unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                and unit.hp < UNIT_MAX_HP[unit.unit_type]
            ):
                exclusivity = DestinationExclusivity.SERVICE_TRANSIT
            co_occupant = next(
                (
                    friendly
                    for friendly in world.friendlies
                    if friendly.id != unit.id and friendly.position == destination
                ),
                None,
            )
            immediate, future, remembered = projection.exposure(destination)
            recorded.append((direction, destination, route.distance))
            intents.append(
                ActionIntent.move(
                    unit.id,
                    mission,
                    priority + rank,
                    direction,
                    destination,
                    risk=immediate * 100 + future * 10 + remembered,
                    destination_exclusivity=exclusivity,
                    tie_break=(rank, route.distance),
                    reason=reason if rank == 0 else "SERVICE_ROUTE_ALTERNATE",
                    metadata=(
                        ("service_transit_kind", kind.value),
                        ("route_rank", rank),
                        ("remaining_distance", route.distance),
                        ("service_tick", None if job is None else job.service_tick),
                        ("exit_tick", None if job is None else job.exit_tick),
                        ("allow_protected", True),
                        (
                            "allow_head_on_swap",
                            destination == world.core.position
                            or unit.position == world.core.position,
                        ),
                        (
                            "service_transit_shared_with",
                            None if co_occupant is None else str(co_occupant.id),
                        ),
                    )
                    + metadata
                    + viability.metadata,
                )
            )
        descriptor = ServiceTransitRoute(
            actor_id=unit.id,
            kind=kind,
            target=target,
            route_distance=None if not routes else routes[0].distance,
            options=tuple(recorded),
            service_tick=None if job is None else job.service_tick,
            exit_tick=None if job is None else job.exit_tick,
        )
        return descriptor, intents, rejected
