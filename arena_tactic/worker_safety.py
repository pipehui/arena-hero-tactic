from __future__ import annotations

from dataclasses import dataclass

from arena_hero import Direction, Position, UnitType

from .geometry import cardinal_neighbors, manhattan
from .models import EntitySnapshot, WorldModel
from .planning import Route, weighted_route_to
from .projection import TacticalMap


@dataclass(frozen=True, slots=True)
class WorkerStepSafety:
    direction: Direction
    destination: Position
    forward_exits: int
    survival_terminals: int
    immediate_attackers: int
    future_attackers: int
    heat: int
    route: Route
    route_reachable: bool
    direction_index: int

    @property
    def score(self) -> tuple[int, ...]:
        return (
            int(self.survival_terminals == 0),
            0 if self.forward_exits >= 2 else (1 if self.forward_exits == 1 else 2),
            self.immediate_attackers,
            self.future_attackers,
            self.heat,
            int(not self.route_reachable),
            self.route.distance,
            self.direction_index,
        )


class WorkerSafetyEvaluator:
    """One source of truth for Worker risk and two-step survivability."""

    @staticmethod
    def route_costs(projection: TacticalMap) -> dict[Position, int]:
        return dict(projection.route_costs_for(UnitType.WORKER))

    @staticmethod
    def forward_safe_exits(
        world: WorldModel,
        projection: TacticalMap,
        position: Position,
        hp: int,
        *,
        origin: Position | None = None,
        target: Position | None = None,
    ) -> int:
        return sum(
            neighbor != origin
            and neighbor not in world.known_obstacles
            and neighbor not in projection.hostile_occupied
            and (neighbor in world.known_passable or neighbor == target)
            and projection.immediate_attackers(neighbor) < hp
            and projection.future_attackers(neighbor) < hp
            for _, neighbor in cardinal_neighbors(position)
        )

    @staticmethod
    def survival_terminals(
        world: WorldModel,
        projection: TacticalMap,
        start: Position,
        hp: int,
        *,
        origin: Position | None = None,
        depth_limit: int = 2,
        node_limit: int = 32,
        target: Position | None = None,
    ) -> int:
        frontier = [(start, 0)]
        visited = {start}
        if origin is not None:
            visited.add(origin)
        terminals = 0
        while frontier and len(visited) <= node_limit:
            cell, depth = frontier.pop(0)
            if depth >= depth_limit:
                terminals += 1
                continue
            for _, neighbor in cardinal_neighbors(cell):
                if (
                    neighbor in visited
                    or neighbor in world.known_obstacles
                    or neighbor in projection.hostile_occupied
                    or (neighbor not in world.known_passable and neighbor != target)
                    or projection.immediate_attackers(neighbor) >= hp
                    or projection.future_attackers(neighbor) >= hp
                ):
                    continue
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
        return terminals

    def recovery_steps(
        self,
        world: WorldModel,
        projection: TacticalMap,
        worker: EntitySnapshot,
        target: Position,
        *,
        blocked: frozenset[Position],
        node_limit: int,
        lookahead_node_limit: int,
    ) -> tuple[WorkerStepSafety, ...]:
        costs = self.route_costs(projection)
        rows: list[WorkerStepSafety] = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(worker.position)
        ):
            continuation = None
            if (
                destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or destination in blocked
                or (
                    destination not in world.known_passable
                    and destination != target
                )
                or projection.immediate_attackers(destination) >= worker.hp
            ):
                continue
            if destination == target:
                route = Route(1, direction, destination)
            else:
                onward_blocked = set(blocked)
                onward_blocked.add(worker.position)
                onward_blocked.discard(destination)
                onward_blocked.discard(target)
                continuation = weighted_route_to(
                    world,
                    destination,
                    target,
                    node_limit=node_limit,
                    blocked=frozenset(onward_blocked),
                    cell_costs=costs,
                )
                route = Route(
                    (
                        continuation.distance + 1
                        if continuation is not None
                        else manhattan(destination, target) + 1
                    ),
                    direction,
                    destination,
                )
            rows.append(
                WorkerStepSafety(
                    direction=direction,
                    destination=destination,
                    forward_exits=self.forward_safe_exits(
                        world,
                        projection,
                        destination,
                        worker.hp,
                        origin=worker.position,
                        target=target,
                    ),
                    survival_terminals=self.survival_terminals(
                        world,
                        projection,
                        destination,
                        worker.hp,
                        origin=worker.position,
                        node_limit=lookahead_node_limit,
                        target=target,
                    ),
                    immediate_attackers=projection.immediate_attackers(destination),
                    future_attackers=projection.future_attackers(destination),
                    heat=projection.worker_exposure(destination)[2],
                    route=route,
                    route_reachable=(
                        destination == target or continuation is not None
                    ),
                    direction_index=index,
                )
            )
        return tuple(sorted(rows, key=lambda item: item.score))

    @staticmethod
    def risk(projection: TacticalMap, cell: Position) -> int:
        immediate, future, remembered = projection.worker_exposure(cell)
        return immediate * 100 + future * 10 + remembered
