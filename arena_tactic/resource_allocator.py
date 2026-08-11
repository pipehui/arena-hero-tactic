from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from uuid import UUID

from arena_hero import Position

from .config import TacticConfig
from .geometry import manhattan
from .models import EntitySnapshot, UnitMission, WorldModel
from .planning import bfs_distances
from .projection import TacticalMap
from .state import TacticMemory


@dataclass(frozen=True, slots=True)
class ResourceAssignment:
    worker_id: UUID
    resource: Position
    cost: int


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    assignments: tuple[ResourceAssignment, ...]
    unreachable_pairs: int = 0

    def as_dict(self) -> dict[UUID, Position]:
        return {item.worker_id: item.resource for item in self.assignments}


@dataclass(slots=True)
class _Edge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _add_edge(graph: list[list[_Edge]], source: int, target: int, cost: int) -> None:
    forward = _Edge(target, len(graph[target]), 1, cost)
    backward = _Edge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(backward)


def minimum_cost_matching(
    workers: tuple[UUID, ...],
    resources: tuple[Position, ...],
    costs: dict[tuple[UUID, Position], int],
) -> tuple[tuple[UUID, Position, int], ...]:
    """Return a deterministic maximum-cardinality, minimum-cost matching.

    This is a small successive-shortest-augmenting-path solver.  It is kept
    independent from Arena Hero state so adversarial assignment matrices can
    be tested directly.
    """

    ordered_workers = tuple(sorted(workers, key=lambda item: item.bytes))
    ordered_resources = tuple(sorted(resources))
    worker_count = len(ordered_workers)
    resource_count = len(ordered_resources)
    if not worker_count or not resource_count:
        return ()

    source = 0
    worker_offset = 1
    resource_offset = worker_offset + worker_count
    sink = resource_offset + resource_count
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    for index in range(worker_count):
        _add_edge(graph, source, worker_offset + index, 0)
    for resource_index in range(resource_count):
        _add_edge(graph, resource_offset + resource_index, sink, 0)
    for worker_index, worker_id in enumerate(ordered_workers):
        for resource_index, resource in enumerate(ordered_resources):
            cost = costs.get((worker_id, resource))
            if cost is None:
                continue
            _add_edge(
                graph,
                worker_offset + worker_index,
                resource_offset + resource_index,
                max(0, cost),
            )

    node_count = len(graph)
    potentials = [0] * node_count
    infinity = 1 << 60
    while True:
        distances = [infinity] * node_count
        parents: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                candidate = distance + edge.cost + potentials[node] - potentials[edge.to]
                if candidate >= distances[edge.to]:
                    continue
                distances[edge.to] = candidate
                parents[edge.to] = node, edge_index
                heappush(queue, (candidate, edge.to))
        if parents[sink] is None:
            break
        for node, distance in enumerate(distances):
            if distance < infinity:
                potentials[node] += distance
        node = sink
        while node != source:
            parent, edge_index = parents[node]  # type: ignore[misc]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent

    result: list[tuple[UUID, Position, int]] = []
    for worker_index, worker_id in enumerate(ordered_workers):
        node = worker_offset + worker_index
        for edge in graph[node]:
            if not resource_offset <= edge.to < sink or edge.capacity != 0:
                continue
            resource = ordered_resources[edge.to - resource_offset]
            if (worker_id, resource) in costs:
                result.append((worker_id, resource, costs[(worker_id, resource)]))
                break
    return tuple(result)


class ResourceAllocator:
    """Global Worker/resource allocator using route-aware matching."""

    def __init__(self, config: TacticConfig, memory: TacticMemory) -> None:
        self.config = config
        self.memory = memory

    def allocate(
        self,
        world: WorldModel,
        projection: TacticalMap,
        workers: tuple[EntitySnapshot, ...],
    ) -> ResourceAllocation:
        if not workers or not projection.resources or world.core is None:
            return ResourceAllocation(())

        resources = tuple(sorted(resource.position for resource in projection.resources))
        seen_ticks = {
            resource.position: resource.last_seen_tick
            for resource in projection.resources
        }
        blocked = self._blocked_positions(world, projection)
        stable, workers, resources = self._retain_work_orders(
            world,
            projection,
            workers,
            resources,
            seen_ticks,
            blocked,
        )
        if not workers or not resources:
            return ResourceAllocation(stable)

        route_distances, core_distances = self._distance_tables(
            world,
            workers,
            resources,
            blocked,
        )
        costs: dict[tuple[UUID, Position], int] = {}
        unreachable = 0
        for worker in sorted(workers, key=lambda item: item.id.bytes):
            for resource in resources:
                outbound = route_distances.get((worker.id, resource))
                if outbound is None:
                    unreachable += 1
                    continue
                # A Worker only needs a proven outbound route to accept a
                # resource job.  Requiring the same bounded BFS to also reach
                # the distant Core made freshly discovered frontier nodes look
                # unreachable even when the Worker was standing beside them.
                # The loaded return planner advances in bounded segments, so
                # an unproven full return route is a cost-estimation concern,
                # not a reason to ignore the resource.
                return_distance = core_distances.get(
                    resource,
                    manhattan(resource, world.core.position),
                )
                costs[(worker.id, resource)] = self._pair_cost(
                    world,
                    projection,
                    seen_ticks,
                    resource,
                    outbound,
                    return_distance,
                )

        rows = minimum_cost_matching(
            tuple(worker.id for worker in workers),
            resources,
            costs,
        )
        return ResourceAllocation(
            assignments=stable
            + tuple(
                ResourceAssignment(worker_id, resource, cost)
                for worker_id, resource, cost in rows
            ),
            unreachable_pairs=unreachable,
        )

    def _blocked_positions(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> frozenset[Position]:
        return (
            projection.hostile_occupied
            | frozenset(projection.immediate_damage)
        )

    def _retain_work_orders(
        self,
        world: WorldModel,
        projection: TacticalMap,
        workers: tuple[EntitySnapshot, ...],
        resources: tuple[Position, ...],
        seen_ticks: dict[Position, int],
        blocked: frozenset[Position],
    ) -> tuple[
        tuple[ResourceAssignment, ...],
        tuple[EntitySnapshot, ...],
        tuple[Position, ...],
    ]:
        # Persistent work orders are the most important optimization borrowed
        # from long-running community tactics: a Worker does not solve the
        # whole assignment problem again while its remembered node remains
        # safely reachable.  New/depleted/blocked targets fall back to the
        # global matcher below.
        stable: list[ResourceAssignment] = []
        claimed: set[Position] = set()
        remaining_workers: list[EntitySnapshot] = []
        resource_set = set(resources)
        for worker in sorted(workers, key=lambda item: item.id.bytes):
            previous = self.memory.unit_missions.get(worker.id)
            target = (
                previous.target
                if previous is not None and previous.mission is UnitMission.HARVEST
                else None
            )
            if (
                target is None
                or target not in resource_set
                or target in claimed
                or target in blocked
            ):
                remaining_workers.append(worker)
                continue
            distances, _ = bfs_distances(
                world,
                target,
                node_limit=self.config.distance_field_node_limit,
                blocked=blocked - {target, worker.position, world.core.position},
                targets=frozenset((worker.position, world.core.position)),
            )
            outbound_distance = distances.get(worker.position)
            if outbound_distance is None:
                remaining_workers.append(worker)
                continue
            return_distance = distances.get(
                world.core.position,
                manhattan(target, world.core.position),
            )
            claimed.add(target)
            stable.append(
                ResourceAssignment(
                    worker.id,
                    target,
                    max(
                        0,
                        self._pair_cost(
                            world,
                            projection,
                            seen_ticks,
                            target,
                            outbound_distance,
                            return_distance,
                        )
                        - self.config.resource_assignment_persistence_bonus,
                    ),
                )
            )
        return (
            tuple(stable),
            tuple(remaining_workers),
            tuple(resource for resource in resources if resource not in claimed),
        )

    def _distance_tables(
        self,
        world: WorldModel,
        workers: tuple[EntitySnapshot, ...],
        resources: tuple[Position, ...],
        blocked: frozenset[Position],
    ) -> tuple[dict[tuple[UUID, Position], int], dict[Position, int]]:
        assert world.core is not None
        route_distances: dict[tuple[UUID, Position], int] = {}
        # Seed return costs with an admissible geometric estimate.  A bounded
        # search may refine it, but failure to cover the whole explored map
        # must not invalidate an otherwise reachable resource assignment.
        core_distances: dict[Position, int] = {
            resource: manhattan(resource, world.core.position)
            for resource in resources
        }
        usable_resources = tuple(
            resource for resource in resources if resource not in blocked
        )
        # The map is undirected.  Build fields from the smaller side of the
        # bipartite assignment and stop each BFS as soon as every relevant
        # endpoint has been reached.  In normal play there are many Workers
        # but only a handful of remembered nodes, reducing per-Tick path scans
        # by an order of magnitude without changing any matching cost.
        if len(usable_resources) <= len(workers) + 1:
            endpoints = frozenset(
                (world.core.position, *(worker.position for worker in workers))
            )
            for resource in usable_resources:
                distances, _ = bfs_distances(
                    world,
                    resource,
                    node_limit=self.config.distance_field_node_limit,
                    blocked=blocked - {resource},
                    targets=endpoints,
                )
                if world.core.position in distances:
                    core_distances[resource] = distances[world.core.position]
                for worker in workers:
                    if worker.position in distances:
                        route_distances[(worker.id, resource)] = distances[
                            worker.position
                        ]
        else:
            proven_core_distances, _ = bfs_distances(
                world,
                world.core.position,
                node_limit=self.config.distance_field_node_limit,
                blocked=blocked - {world.core.position},
                targets=frozenset(usable_resources),
            )
            for resource in usable_resources:
                if resource in proven_core_distances:
                    core_distances[resource] = proven_core_distances[resource]
            for worker in sorted(workers, key=lambda item: item.id.bytes):
                distances, _ = bfs_distances(
                    world,
                    worker.position,
                    node_limit=self.config.distance_field_node_limit,
                    blocked=blocked - {worker.position},
                    targets=frozenset(usable_resources),
                )
                for resource in usable_resources:
                    if resource in distances:
                        route_distances[(worker.id, resource)] = distances[resource]
        return route_distances, core_distances

    def _pair_cost(
        self,
        world: WorldModel,
        projection: TacticalMap,
        seen_ticks: dict[Position, int],
        resource: Position,
        outbound: int,
        return_distance: int,
    ) -> int:
        immediate, future, remembered = projection.worker_exposure(resource)
        age = max(0, world.tick - seen_ticks.get(resource, world.tick))
        uncertainty = (
            0
            if (
                (intel := projection.resource(resource)) is not None
                and intel.visible_now
            )
            else 2 + min(8, age // 32)
        )
        congestion = min(16, self.memory.congestion_counts.get(resource, 0))
        return max(
            0,
            outbound
            + return_distance
            + immediate * 100
            + future * 12
            + remembered * 2
            + uncertainty
            + congestion,
        )
