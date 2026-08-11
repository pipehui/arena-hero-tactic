from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache

from arena_hero import Direction, Position, UnitType


DIRECTION_ORDER: tuple[Direction, ...] = (
    Direction.UP,
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
)


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def add_direction(position: Position, direction: Direction) -> Position:
    x, y = position
    if direction is Direction.UP:
        return x, y - 1
    if direction is Direction.RIGHT:
        return x + 1, y
    if direction is Direction.DOWN:
        return x, y + 1
    return x - 1, y


def direction_between(source: Position, destination: Position) -> Direction | None:
    delta = destination[0] - source[0], destination[1] - source[1]
    for direction in DIRECTION_ORDER:
        if direction.delta == delta:
            return direction
    return None


def cardinal_neighbors(position: Position) -> Iterator[tuple[Direction, Position]]:
    x, y = position
    yield Direction.UP, (x, y - 1)
    yield Direction.RIGHT, (x + 1, y)
    yield Direction.DOWN, (x, y + 1)
    yield Direction.LEFT, (x - 1, y)


def diamond(center: Position, radius: int) -> Iterator[Position]:
    cx, cy = center
    for dx in range(-radius, radius + 1):
        remaining = radius - abs(dx)
        for dy in range(-remaining, remaining + 1):
            yield cx + dx, cy + dy


def manhattan_ring(center: Position, radius: int) -> tuple[Position, ...]:
    if radius <= 0:
        return (center,)
    cx, cy = center
    points: list[Position] = []
    for offset in range(radius):
        points.append((cx + offset, cy - radius + offset))
    for offset in range(radius):
        points.append((cx + radius - offset, cy + offset))
    for offset in range(radius):
        points.append((cx - offset, cy + radius - offset))
    for offset in range(radius):
        points.append((cx - radius + offset, cy - offset))
    return tuple(points)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


@lru_cache(maxsize=512)
def _relative_supercover(dx: int, dy: int) -> tuple[Position, ...]:
    nx = abs(dx)
    ny = abs(dy)
    sx = _sign(dx)
    sy = _sign(dy)
    x = y = 0
    cells: list[Position] = [(x, y)]
    ix = iy = 0
    while ix < nx or iy < ny:
        left = (1 + 2 * ix) * ny
        right = (1 + 2 * iy) * nx
        if left == right:
            # At a perfect corner both touching side cells belong to the
            # supercover before the diagonal destination.
            if sx:
                cells.append((x + sx, y))
            if sy:
                cells.append((x, y + sy))
            x += sx
            y += sy
            ix += 1
            iy += 1
        elif left < right:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        cells.append((x, y))
    return tuple(dict.fromkeys(cells))


def supercover_line(start: Position, end: Position) -> tuple[Position, ...]:
    """Return every grid cell touched by the segment between cell centres."""

    relative = _relative_supercover(end[0] - start[0], end[1] - start[1])
    return tuple((start[0] + dx, start[1] + dy) for dx, dy in relative)


def vision_is_clear(
    source: Position,
    target: Position,
    obstacles: frozenset[Position] | set[Position],
) -> bool:
    if source == target:
        return True
    line = _relative_supercover(target[0] - source[0], target[1] - source[1])
    # The obstacle target itself is visible; only earlier cells block it.
    return not any(
        (source[0] + dx, source[1] + dy) in obstacles
        for dx, dy in line[1:-1]
    )


def unit_attack_cells(
    position: Position,
    unit_type: UnitType,
    obstacles: frozenset[Position] | set[Position],
) -> frozenset[Position]:
    """Return the exact cells an observed Unit can damage without moving."""

    if unit_type is UnitType.VANGUARD:
        return frozenset(
            add_direction(position, direction) for direction in DIRECTION_ORDER
        )
    if unit_type is not UnitType.RANGER:
        return frozenset()
    px, py = position
    cells: set[Position] = set()
    for dx, dy in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        for distance in range(1, 4):
            cell = px + dx * distance, py + dy * distance
            if cell in obstacles:
                break
            cells.add(cell)
    return frozenset(cells)


def ranger_line_is_clear(
    source: Position,
    target: Position,
    obstacles: frozenset[Position] | set[Position],
) -> bool:
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    distance = max(abs(dx), abs(dy))
    if not 1 <= distance <= 3:
        return False
    if not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
        return False
    step_x = _sign(dx)
    step_y = _sign(dy)
    for step in range(1, distance):
        if (source[0] + step_x * step, source[1] + step_y * step) in obstacles:
            return False
    return True


def ranger_firing_positions(
    target: Position,
    *,
    minimum_range: int = 2,
    maximum_range: int = 3,
) -> tuple[Position, ...]:
    """Return every axis/diagonal firing origin in deterministic order.

    Ranger range is the number of steps along an aligned ray, not Manhattan
    distance.  Keeping this geometry in one helper prevents formation and
    relocation planners from accidentally discarding valid ``(2, 2)`` and
    ``(3, 3)`` diagonal lines.
    """

    if minimum_range < 1 or maximum_range < minimum_range:
        raise ValueError("invalid Ranger firing range")
    tx, ty = target
    rays = (
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    )
    return tuple(
        (tx + dx * distance, ty + dy * distance)
        for distance in range(minimum_range, maximum_range + 1)
        for dx, dy in rays
    )


def count_open_neighbors(position: Position, obstacles: Iterable[Position]) -> int:
    blocked = set(obstacles)
    return sum(neighbor not in blocked for _, neighbor in cardinal_neighbors(position))
