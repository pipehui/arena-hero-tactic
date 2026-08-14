from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from collections.abc import Mapping

from arena_hero import Direction, Position

from .geometry import DIRECTION_ORDER, add_direction, cardinal_neighbors, diamond, manhattan
from .models import WorldModel


_STEPS: tuple[tuple[Direction, int, int], ...] = (
    (Direction.UP, 0, -1),
    (Direction.RIGHT, 1, 0),
    (Direction.DOWN, 0, 1),
    (Direction.LEFT, -1, 0),
)

@dataclass(frozen=True, slots=True)
class MoveViability:
    """Pure proof that a first step does not enter a known sealed branch."""

    forward_exits: int
    continuation_reachable: bool
    local_open: bool
    unknown_frontier: bool
    viable: bool
    terminal_exception: str | None = None
    rejection_reason: str | None = None

    @property
    def metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("forward_exits", self.forward_exits),
            ("continuation_reachable", self.continuation_reachable),
            ("local_open", self.local_open),
            ("unknown_frontier", self.unknown_frontier),
            ("dead_end_rejected", not self.viable),
            ("terminal_exception", self.terminal_exception),
            ("viability_rejection", self.rejection_reason),
        )


_viability_cache_world: WorldModel | None = None
_viability_cache: dict[tuple[object, ...], MoveViability] = {}


@dataclass(frozen=True, slots=True)
class Route:
    distance: int
    first_direction: Direction | None
    first_position: Position | None
    viability: MoveViability | None = None


def path_to(
    world: WorldModel,
    start: Position,
    target: Position,
    *,
    node_limit: int,
    blocked: frozenset[Position] = frozenset(),
) -> tuple[Position, ...] | None:
    """Return an obstacle-aware deterministic path including both endpoints."""

    if start == target:
        return (start,)
    passable = world.known_passable
    obstacles = world.known_obstacles
    if target in obstacles:
        return None
    distances = {start: 0}
    parents: dict[Position, Position] = {}
    queue: list[tuple[int, int, int, int, Position]] = []
    start_h = manhattan(start, target)
    heappush(queue, (start_h, start_h, start[0], start[1], start))
    while queue and len(distances) < node_limit:
        _, _, _, _, current = heappop(queue)
        if current == target:
            path = [target]
            while path[-1] != start:
                path.append(parents[path[-1]])
            path.reverse()
            return tuple(path)
        for _, neighbor in cardinal_neighbors(current):
            if (
                neighbor in obstacles
                or (neighbor not in passable and neighbor not in {start, target})
                or (neighbor in blocked and neighbor not in {start, target})
            ):
                continue
            distance = distances[current] + 1
            if distance >= distances.get(neighbor, 1 << 60):
                continue
            distances[neighbor] = distance
            parents[neighbor] = current
            heuristic = manhattan(neighbor, target)
            heappush(
                queue,
                (distance + heuristic, heuristic, neighbor[0], neighbor[1], neighbor),
            )
    return None


def bfs_distances(
    world: WorldModel,
    start: Position,
    *,
    node_limit: int,
    blocked: frozenset[Position] = frozenset(),
    targets: frozenset[Position] | None = None,
) -> tuple[dict[Position, int], dict[Position, tuple[Position, Direction]]]:
    passable = world.known_passable
    obstacles = world.known_obstacles
    distances = {start: 0}
    parents: dict[Position, tuple[Position, Direction]] = {}
    queue: deque[Position] = deque((start,))
    remaining = None if targets is None else set(targets) - {start}
    if remaining is not None and not remaining:
        return distances, parents
    while queue and len(distances) < node_limit:
        current = queue.popleft()
        cx, cy = current
        for direction, dx, dy in _STEPS:
            neighbor = cx + dx, cy + dy
            if (
                neighbor in distances
                or neighbor in obstacles
                or (neighbor not in passable and neighbor != start)
                or (neighbor in blocked and neighbor != start)
            ):
                continue
            distances[neighbor] = distances[current] + 1
            parents[neighbor] = current, direction
            queue.append(neighbor)
            if remaining is not None:
                remaining.discard(neighbor)
                if not remaining:
                    return distances, parents
    return distances, parents


def weighted_distance_field(
    world: WorldModel,
    start: Position,
    *,
    node_limit: int,
    blocked: frozenset[Position] = frozenset(),
    cell_costs: Mapping[Position, int] | None = None,
) -> tuple[dict[Position, int], dict[Position, tuple[Position, Direction]]]:
    """Return a deterministic risk-weighted field rooted at ``start``."""

    passable = world.known_passable
    obstacles = world.known_obstacles
    costs = cell_costs or {}
    distances = {start: 0}
    parents: dict[Position, tuple[Position, Direction]] = {}
    queue: list[tuple[int, int, int, Position]] = [(0, start[0], start[1], start)]
    settled: set[Position] = set()
    while queue and len(settled) < node_limit:
        distance, _, _, current = heappop(queue)
        if current in settled or distance != distances[current]:
            continue
        settled.add(current)
        for direction, neighbor in cardinal_neighbors(current):
            if (
                neighbor in settled
                or neighbor in obstacles
                or (neighbor not in passable and neighbor != start)
                or (neighbor in blocked and neighbor != start)
            ):
                continue
            next_distance = distance + 1 + max(0, costs.get(neighbor, 0))
            if next_distance >= distances.get(neighbor, 1 << 60):
                continue
            distances[neighbor] = next_distance
            parents[neighbor] = current, direction
            heappush(
                queue,
                (next_distance, neighbor[0], neighbor[1], neighbor),
            )
    return (
        {cell: distances[cell] for cell in settled},
        {cell: parent for cell, parent in parents.items() if cell in settled},
    )


def route_to(
    world: WorldModel,
    start: Position,
    target: Position,
    *,
    node_limit: int,
    blocked: frozenset[Position] = frozenset(),
    allow_unknown_endpoint: bool = False,
) -> Route | None:
    if start == target:
        return Route(0, None, None)
    passable = world.known_passable
    obstacles = world.known_obstacles
    if target not in passable and target != start and not allow_unknown_endpoint:
        return None
    if target in obstacles or target in blocked:
        return None
    distances = {start: 0}
    parents: dict[Position, tuple[Position, Direction]] = {}
    queue: list[tuple[int, int, int, int, int, Position]] = []
    start_h = manhattan(start, target)
    heappush(queue, (start_h, start_h, 0, start[0], start[1], start))
    found = False
    while queue and len(distances) < node_limit:
        _, _, queued_distance, _, _, current = heappop(queue)
        if queued_distance != distances[current]:
            continue
        if current == target:
            found = True
            break
        current_distance = distances[current]
        cx, cy = current
        for direction, dx, dy in _STEPS:
            neighbor = cx + dx, cy + dy
            next_distance = current_distance + 1
            if neighbor in obstacles or (neighbor in blocked and neighbor != start):
                continue
            if neighbor not in passable and neighbor != start and not (
                neighbor == target and allow_unknown_endpoint
            ):
                continue
            if next_distance >= distances.get(neighbor, 1 << 60):
                continue
            distances[neighbor] = next_distance
            parents[neighbor] = current, direction
            heuristic = manhattan(neighbor, target)
            heappush(
                queue,
                (
                    next_distance + heuristic,
                    heuristic,
                    next_distance,
                    neighbor[0],
                    neighbor[1],
                    neighbor,
                ),
            )
    if not found:
        return None
    distance = distances[target]
    current = target
    parent = parents.get(current)
    while parent is not None and parent[0] != start:
        current = parent[0]
        parent = parents.get(current)
    if parent is None:
        return None
    first_direction = parent[1]
    return Route(distance, first_direction, add_direction(start, first_direction))


def move_viability(
    world: WorldModel,
    origin: Position,
    destination: Position,
    *,
    target: Position | None = None,
    blocked: frozenset[Position] = frozenset(),
    node_limit: int = 256,
    lookahead_depth: int = 2,
    require_continuation: bool = False,
    require_open_area: bool = False,
    terminal_exception: str | None = None,
) -> MoveViability:
    """Classify a first step after removing the cell it just left.

    Ordinary navigation must not count the origin as an escape route.  A
    destination beside unexplored fog is not declared a dead end until the
    authoritative world model has actually observed the surrounding terrain.
    """

    global _viability_cache_world
    global _viability_cache
    if _viability_cache_world is not world:
        _viability_cache_world = world
        _viability_cache = {}
    cache_key = (
        origin,
        destination,
        target,
        blocked,
        node_limit,
        lookahead_depth,
        require_continuation,
        require_open_area,
        terminal_exception,
    )
    cached = _viability_cache.get(cache_key)
    if cached is not None:
        return cached

    obstacles = world.known_obstacles
    passable = world.known_passable
    blocked_cells = set(blocked)
    blocked_cells.discard(origin)
    blocked_cells.discard(destination)
    if target is not None:
        blocked_cells.discard(target)

    def known_open(cell: Position) -> bool:
        return (
            cell in passable
            and cell not in obstacles
            and cell not in blocked_cells
        )

    forward = tuple(
        neighbor
        for _, neighbor in cardinal_neighbors(destination)
        if neighbor != origin and known_open(neighbor)
    )
    forward_exits = len(forward)
    local_open = forward_exits >= 2
    unknown_frontier = False

    queue = deque(((destination, 0, origin),))
    visited = {origin, destination}
    while queue:
        cell, depth, parent = queue.popleft()
        onward: list[Position] = []
        for _, neighbor in cardinal_neighbors(cell):
            if neighbor == parent or neighbor in obstacles or neighbor in blocked_cells:
                continue
            if neighbor not in passable:
                if neighbor not in world.visible_cells:
                    unknown_frontier = True
                continue
            onward.append(neighbor)
        if len(onward) >= 2:
            local_open = True
        if depth >= lookahead_depth:
            continue
        for neighbor in onward:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, depth + 1, cell))

    continuation_reachable = False
    if target is not None:
        if destination == target:
            continuation_reachable = True
        else:
            onward_blocked = set(blocked_cells)
            onward_blocked.add(origin)
            onward_blocked.discard(destination)
            onward_blocked.discard(target)
            continuation_reachable = (
                route_to(
                    world,
                    destination,
                    target,
                    node_limit=max(1, node_limit),
                    blocked=frozenset(onward_blocked),
                    allow_unknown_endpoint=True,
                )
                is not None
            )

    has_forward_space = forward_exits > 0 or unknown_frontier
    viable = terminal_exception is not None or (
        has_forward_space
        and (not require_continuation or continuation_reachable)
        and (
            not require_open_area
            or local_open
            or unknown_frontier
            or continuation_reachable
        )
    )
    rejection_reason = None
    if not viable:
        rejection_reason = (
            "NO_ROUTE_CONTINUATION"
            if require_continuation and not continuation_reachable
            else "DEAD_END_INTERMEDIATE"
        )
    result = MoveViability(
        forward_exits=forward_exits,
        continuation_reachable=continuation_reachable,
        local_open=local_open,
        unknown_frontier=unknown_frontier,
        viable=viable,
        terminal_exception=terminal_exception,
        rejection_reason=rejection_reason,
    )
    _viability_cache[cache_key] = result
    return result


def weighted_route_to(
    world: WorldModel,
    start: Position,
    target: Position,
    *,
    node_limit: int,
    blocked: frozenset[Position] = frozenset(),
    cell_costs: Mapping[Position, int] | None = None,
    allow_unknown_endpoint: bool = False,
) -> Route | None:
    """A* route that prefers lower-risk cells without treating fog as truth."""

    if start == target:
        return Route(0, None, None)
    passable = world.known_passable
    obstacles = world.known_obstacles
    if target not in passable and target != start and not allow_unknown_endpoint:
        return None
    if target in obstacles or target in blocked:
        return None
    costs = cell_costs or {}
    best = {start: 0}
    steps = {start: 0}
    parents: dict[Position, tuple[Position, Direction]] = {}
    queue: list[tuple[int, int, int, int, int, Position]] = []
    start_h = manhattan(start, target)
    heappush(queue, (start_h, start_h, 0, start[0], start[1], start))
    found = False
    while queue and len(best) < node_limit:
        _, _, queued_cost, _, _, current = heappop(queue)
        if queued_cost != best[current]:
            continue
        if current == target:
            found = True
            break
        for direction, neighbor in cardinal_neighbors(current):
            if neighbor in obstacles or (neighbor in blocked and neighbor != start):
                continue
            if neighbor not in passable and neighbor != start and not (
                neighbor == target and allow_unknown_endpoint
            ):
                continue
            next_cost = best[current] + 1 + max(0, costs.get(neighbor, 0))
            if next_cost >= best.get(neighbor, 1 << 60):
                continue
            best[neighbor] = next_cost
            steps[neighbor] = steps[current] + 1
            parents[neighbor] = current, direction
            heuristic = manhattan(neighbor, target)
            heappush(
                queue,
                (
                    next_cost + heuristic,
                    heuristic,
                    next_cost,
                    neighbor[0],
                    neighbor[1],
                    neighbor,
                ),
            )
    if not found:
        return None
    current = target
    parent = parents.get(current)
    while parent is not None and parent[0] != start:
        current = parent[0]
        parent = parents.get(current)
    if parent is None:
        return None
    return Route(steps[target], parent[1], add_direction(start, parent[1]))


def weighted_progress_route(
    world: WorldModel,
    start: Position,
    target: Position,
    *,
    node_limit: int,
    blocked: frozenset[Position] = frozenset(),
    cell_costs: Mapping[Position, int] | None = None,
) -> tuple[Route, Position] | None:
    """Return a safe bounded-search segment toward a currently distant goal.

    Unlike ``weighted_route_to`` this routine does not claim that the final
    target was reached.  It returns the best explored waypoint and the exact
    first step to that waypoint, so a remote cargo Worker can keep making
    progress without a second small-budget proof of the entire route.
    """

    if start == target or node_limit <= 1:
        return None
    passable = world.known_passable
    obstacles = world.known_obstacles
    costs = cell_costs or {}
    best = {start: 0}
    steps = {start: 0}
    parents: dict[Position, tuple[Position, Direction]] = {}
    queue: list[tuple[int, int, int, int, int, Position]] = []
    start_h = manhattan(start, target)
    heappush(queue, (start_h, start_h, 0, start[0], start[1], start))
    while queue and len(best) < node_limit:
        _, _, queued_cost, _, _, current = heappop(queue)
        if queued_cost != best[current]:
            continue
        for direction, neighbor in cardinal_neighbors(current):
            if neighbor in obstacles or (neighbor in blocked and neighbor != start):
                continue
            if neighbor not in passable:
                continue
            next_cost = best[current] + 1 + max(0, costs.get(neighbor, 0))
            if next_cost >= best.get(neighbor, 1 << 60):
                continue
            best[neighbor] = next_cost
            steps[neighbor] = steps[current] + 1
            parents[neighbor] = current, direction
            heuristic = manhattan(neighbor, target)
            heappush(
                queue,
                (
                    next_cost + heuristic,
                    heuristic,
                    next_cost,
                    neighbor[0],
                    neighbor[1],
                    neighbor,
                ),
            )

    candidates = []
    for cell in best:
        if cell == start:
            continue
        parent = parents.get(cell)
        onward = sum(
            neighbor != (None if parent is None else parent[0])
            and neighbor in passable
            and neighbor not in obstacles
            and neighbor not in blocked
            for _, neighbor in cardinal_neighbors(cell)
        )
        unknown_frontier = any(
            neighbor not in passable
            and neighbor not in obstacles
            and neighbor not in world.visible_cells
            for _, neighbor in cardinal_neighbors(cell)
        )
        if onward == 0 and not unknown_frontier:
            continue
        candidates.append(
            (
                manhattan(cell, target),
                best[cell],
                -steps[cell],
                cell[0],
                cell[1],
                cell,
            )
        )
    if not candidates:
        return None
    *_, waypoint = min(candidates)
    current = waypoint
    parent = parents.get(current)
    while parent is not None and parent[0] != start:
        current = parent[0]
        parent = parents.get(current)
    if parent is None:
        return None
    route = Route(steps[waypoint], parent[1], add_direction(start, parent[1]))
    return route, waypoint


def route_from_field(
    start: Position,
    target: Position,
    distances: dict[Position, int],
    parents: dict[Position, tuple[Position, Direction]],
    *,
    obstacles: frozenset[Position],
    allow_unknown_endpoint: bool = False,
) -> Route | None:
    """Resolve a route from one reusable BFS field."""

    if start == target:
        return Route(0, None, None)
    if target not in distances:
        if not allow_unknown_endpoint or target in obstacles:
            return None
        options: list[tuple[int, int, Position, Direction]] = []
        for index, direction in enumerate(DIRECTION_ORDER):
            predecessor = add_direction(target, _opposite(direction))
            if predecessor in distances:
                options.append((distances[predecessor], index, predecessor, direction))
        if not options:
            return None
        _, _, predecessor, final_direction = min(options)
        distance = distances[predecessor] + 1
        if predecessor == start:
            return Route(distance, final_direction, target)
        current = predecessor
    else:
        distance = distances[target]
        current = target

    parent = parents.get(current)
    while parent is not None and parent[0] != start:
        current = parent[0]
        parent = parents.get(current)
    if parent is None:
        return None
    return Route(distance, parent[1], add_direction(start, parent[1]))


def _opposite(direction: Direction) -> Direction:
    mapping = {
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP,
        Direction.LEFT: Direction.RIGHT,
        Direction.RIGHT: Direction.LEFT,
    }
    return mapping[direction]


def information_gain(
    cell: Position,
    *,
    tick: int,
    last_visible: dict[Position, int],
    radius: int = 3,
    refresh_ticks: int = 64,
) -> int:
    gain = 0
    for visible in diamond(cell, radius):
        last = last_visible.get(visible)
        if last is None:
            gain += 4
        elif tick - last >= refresh_ticks:
            gain += 2
        elif tick - last >= refresh_ticks // 2:
            gain += 1
    return gain


def exploration_candidates(
    world: WorldModel,
    start: Position,
    *,
    distances: Mapping[Position, int],
    search_radius: int,
    limit: int,
    backoff: frozenset[Position],
) -> tuple[Position, ...]:
    """Build exploration targets from the Worker's proven reachable field.

    Unknown cells are only admitted when a reachable known predecessor exists.
    The old global-staleness fallback could fill the entire limit with remote
    cells that the bounded field had never reached, which left the Worker with
    no route and no persistent task.
    """

    known = world.known_passable
    reachable = set(distances) & known
    candidates: set[Position] = set()
    for cell in reachable:
        if manhattan(start, cell) > search_radius:
            continue
        for _, neighbor in cardinal_neighbors(cell):
            if (
                neighbor not in known
                and neighbor not in world.known_obstacles
                and neighbor not in backoff
            ):
                candidates.add(neighbor)
    if not candidates:
        stale = sorted(
            (
                (seen_tick, distances[cell], manhattan(start, cell), cell)
                for cell, seen_tick in world.cell_last_visible
                if cell in reachable and cell not in backoff and cell != start
            )
        )
        candidates.update(cell for _, _, _, cell in stale[:limit])
    ordered = sorted(
        candidates,
        key=lambda cell: (
            distances.get(cell, min(
                (distances.get(neighbor, 1 << 60) + 1 for _, neighbor in cardinal_neighbors(cell)),
                default=1 << 60,
            )),
            manhattan(start, cell),
            cell,
        ),
    )
    return tuple(ordered[:limit])


_SCOUT_SECTORS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)


def sector_scout_candidates(
    world: WorldModel,
    core: Position,
    *,
    sector_index: int,
    radius: int,
    tick: int,
    refresh_ticks: int,
    limit: int,
    backoff: frozenset[Position],
    claimed: frozenset[Position],
) -> tuple[Position, ...]:
    """Return deterministic targets around a stable sector/radius waypoint.

    These candidates are deliberately only *nominal*.  The caller proves each
    one with a bounded single-target A*, allowing a far scout to advance even
    when a reusable distance field was capped before reaching its destination.
    """

    dx, dy = _SCOUT_SECTORS[sector_index % len(_SCOUT_SECTORS)]
    scale = max(1, abs(dx) + abs(dy))
    nominal = (
        core[0] + dx * radius // scale,
        core[1] + dy * radius // scale,
    )
    known = world.known_passable
    last_visible = dict(world.cell_last_visible)
    candidates: set[Position] = set()
    # Probe inward along the stable sector ray.  This is independent of total
    # explored-map size and finds the nearest reachable-looking frontier near
    # the nominal ring without rescanning every known cell for every Worker.
    # A stable scouting band refreshes a narrow annulus; probing all the way
    # back toward Core silently turns a 30-cell slot into an inner patrol and
    # also lets successive assignments drift outward again.
    minimum = max(1, radius - 3)
    for distance in range(radius, minimum - 1, -1):
        waypoint = (
            core[0] + dx * distance // scale,
            core[1] + dy * distance // scale,
        )
        for cell in diamond(waypoint, 3):
            rel_x, rel_y = cell[0] - core[0], cell[1] - core[1]
            if (
                rel_x * dx + rel_y * dy <= 0
                or cell in world.known_obstacles
                or cell in backoff
                or cell in claimed
            ):
                continue
            if cell in known or any(
                neighbor in known for _, neighbor in cardinal_neighbors(cell)
            ):
                candidates.add(cell)
        if candidates:
            break
    # Exact information gain scans a radius-3 diamond.  Applying it to every
    # known cell for every newly assigned Worker made a restart Turn scale as
    # O(workers * map_cells * vision_area).  First retain a small geometric
    # and staleness-aware shortlist, then spend the exact scoring budget.
    shortlist = sorted(
        candidates,
        key=lambda cell: (
            int(cell in known),
            abs(manhattan(core, cell) - radius),
            manhattan(cell, nominal),
            last_visible.get(cell, -1),
            cell,
        ),
    )[: max(32, limit * 4)]
    scored = sorted(
        shortlist,
        key=lambda cell: (
            -information_gain(
                cell,
                tick=tick,
                last_visible=last_visible,
                refresh_ticks=refresh_ticks,
            ),
            manhattan(cell, nominal),
            abs(manhattan(core, cell) - radius),
            last_visible.get(cell, -1),
            cell,
        ),
    )
    return tuple(scored[:limit])
