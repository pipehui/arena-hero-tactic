from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from arena_hero import Direction, Position, UnitType

from .geometry import cardinal_neighbors, diamond, manhattan, unit_attack_cells
from .models import EnemyCoreControlZone, EntitySnapshot, WorldModel
from .planning import move_viability, Route, weighted_route_to
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

    def __init__(self) -> None:
        self._threat_layer_cache: dict[
            tuple[int, tuple[UUID, ...], int],
            tuple[dict[Position, int], ...],
        ] = {}

    def threat_layers(
        self,
        world: WorldModel,
        projection: TacticalMap,
        threat_ids: tuple[UUID, ...],
        depth_limit: int,
    ) -> tuple[dict[Position, int], ...]:
        """Conservative attack envelopes after each future movement phase."""

        key = (
            projection.tick,
            tuple(sorted(threat_ids, key=lambda item: item.bytes)),
            depth_limit,
        )
        cached = self._threat_layer_cache.get(key)
        if cached is not None:
            return cached
        if len(self._threat_layer_cache) > 32:
            self._threat_layer_cache.clear()
        positions = {
            enemy.enemy_id: (
                {enemy.observed_position}
                if enemy.visible_now
                else set(enemy.possible_positions)
            )
            for enemy in projection.enemies
            if enemy.enemy_id in threat_ids
        }
        unit_types = {
            enemy.enemy_id: enemy.unit_type
            for enemy in projection.enemies
            if enemy.enemy_id in positions
        }
        layers: list[dict[Position, int]] = []
        for depth in range(depth_limit + 1):
            attackers: Counter[Position] = Counter()
            for enemy_id, enemy_positions in positions.items():
                attacked: set[Position] = set()
                for position in enemy_positions:
                    attacked.update(
                        unit_attack_cells(
                            position,
                            unit_types[enemy_id],
                            world.known_obstacles,
                        )
                    )
                attackers.update(attacked)
            layers.append(dict(attackers))
            if depth == depth_limit:
                break
            positions = {
                enemy_id: enemy_positions
                | {
                    neighbor
                    for position in enemy_positions
                    for _, neighbor in cardinal_neighbors(position)
                    if neighbor not in world.known_obstacles
                }
                for enemy_id, enemy_positions in positions.items()
            }
        result = tuple(layers)
        self._threat_layer_cache[key] = result
        return result

    def projected_attackers(
        self,
        world: WorldModel,
        projection: TacticalMap,
        cell: Position,
        *,
        depth: int,
        threat_ids: tuple[UUID, ...],
        depth_limit: int,
    ) -> int:
        if not threat_ids:
            return 0
        layers = self.threat_layers(world, projection, threat_ids, depth_limit)
        return layers[min(depth, len(layers) - 1)].get(cell, 0)

    @staticmethod
    def route_costs(projection: TacticalMap) -> dict[Position, int]:
        return dict(projection.route_costs_for(UnitType.WORKER))

    @staticmethod
    def navigation_layers(
        projection: TacticalMap,
        zones: tuple[EnemyCoreControlZone, ...],
    ) -> tuple[frozenset[Position], dict[Position, int]]:
        """Return the shared hard/soft Worker map around remembered enemy Cores."""

        blocked: set[Position] = set()
        costs = WorkerSafetyEvaluator.route_costs(projection)
        for zone in zones:
            blocked.update(diamond(zone.center, zone.exclusion_radius))
            for cell in diamond(zone.center, zone.clear_radius):
                distance = manhattan(cell, zone.center)
                if distance <= zone.exclusion_radius:
                    continue
                costs[cell] = costs.get(cell, 0) + (
                    zone.clear_radius - distance + 1
                ) * 64
        return frozenset(blocked), costs

    def forward_safe_exits(
        self,
        world: WorldModel,
        projection: TacticalMap,
        position: Position,
        hp: int,
        *,
        origin: Position | None = None,
        target: Position | None = None,
        threat_ids: tuple[UUID, ...] = (),
        threat_depth: int = 1,
    ) -> int:
        layers = (
            self.threat_layers(
                world,
                projection,
                threat_ids,
                max(1, threat_depth),
            )
            if threat_ids
            else ()
        )
        return sum(
            neighbor != origin
            and neighbor not in world.known_obstacles
            and neighbor not in projection.hostile_occupied
            and (neighbor in world.known_passable or neighbor == target)
            and projection.immediate_attackers(neighbor) < hp
            and projection.future_attackers(neighbor) < hp
            and (
                not layers
                or layers[min(threat_depth, len(layers) - 1)].get(neighbor, 0) < hp
            )
            for _, neighbor in cardinal_neighbors(position)
        )

    def survival_terminals(
        self,
        world: WorldModel,
        projection: TacticalMap,
        start: Position,
        hp: int,
        *,
        origin: Position | None = None,
        depth_limit: int = 2,
        node_limit: int = 32,
        target: Position | None = None,
        threat_ids: tuple[UUID, ...] = (),
    ) -> int:
        threat_layers = (
            self.threat_layers(
                world,
                projection,
                threat_ids,
                depth_limit,
            )
            if threat_ids
            else ()
        )
        if threat_layers and threat_layers[0].get(start, 0) >= hp:
            return 0
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
                unknown_frontier = (
                    neighbor not in world.known_passable
                    and neighbor != target
                    and neighbor not in world.known_obstacles
                    and neighbor not in projection.hostile_occupied
                )
                if unknown_frontier:
                    # Unknown terrain is never traversed as a proved route,
                    # but a safely reached frontier is a viable continuation:
                    # visibility on the next Tick will reveal whether it is
                    # open.  Without this terminal case, a Worker near the
                    # edge of its current vision is falsely classified as
                    # trapped even while moving directly away from threats.
                    if (
                        projection.immediate_attackers(neighbor) < hp
                        and projection.future_attackers(neighbor) < hp
                        and (
                            not threat_layers
                            or threat_layers[
                                min(depth + 1, depth_limit)
                            ].get(neighbor, 0)
                            < hp
                        )
                    ):
                        terminals += 1
                    continue
                if (
                    neighbor in visited
                    or neighbor in world.known_obstacles
                    or neighbor in projection.hostile_occupied
                    or projection.immediate_attackers(neighbor) >= hp
                    or projection.future_attackers(neighbor) >= hp
                    or (
                        threat_layers
                        and threat_layers[min(depth + 1, depth_limit)].get(neighbor, 0)
                        >= hp
                    )
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
            terminal_exception = (
                "CORE_SERVICE"
                if world.core is not None
                and destination == world.core.position
                else None
            )
            viability = move_viability(
                world,
                worker.position,
                destination,
                target=target,
                blocked=blocked,
                node_limit=min(node_limit, 512),
                require_continuation=(
                    terminal_exception is None and destination != target
                ),
                terminal_exception=terminal_exception,
            )
            if not viability.viable:
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
                    forward_exits=viability.forward_exits,
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
