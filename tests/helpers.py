from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from arena_hero import (
    Accepted,
    ChampionBeacon,
    CommandSource,
    CoreState,
    CoreView,
    Direction,
    PlayerState,
    PlayerStatus,
    ResolutionEvent,
    TerrainView,
    Turn,
    UnitType,
    UnitView,
)


def uid(value: int) -> UUID:
    return UUID(int=value)


def friendly_core(
    *,
    identifier: int = 10_000,
    position: tuple[int, int] = (0, 0),
    hp: int = 5,
    shield: int = 5,
    moving: bool = False,
    direction: Direction = Direction.RIGHT,
    progress: int = 1,
) -> CoreView:
    dx, dy = direction.delta
    return CoreView(
        kind="CORE",
        id=uid(identifier),
        controlled=True,
        owner_username="tester",
        position=position,
        hp=hp,
        shield=shield,
        state=CoreState.MOVING if moving else CoreState.NORMAL,
        move_direction=direction if moving else None,
        move_progress=progress if moving else None,
        move_required_ticks=4 if moving else None,
        destination=(position[0] + dx, position[1] + dy) if moving else None,
    )


def unit(
    identifier: int,
    unit_type: UnitType,
    position: tuple[int, int],
    *,
    hp: int | None = None,
    cargo: int = 0,
    controlled: bool = True,
) -> UnitView:
    maximum = {
        UnitType.WORKER: 2,
        UnitType.VANGUARD: 4,
        UnitType.RANGER: 2,
    }[unit_type]
    return UnitView(
        kind="UNIT",
        id=uid(identifier),
        controlled=controlled,
        position=position,
        hp=maximum if hp is None else hp,
        unit_type=unit_type,
        cargo=cargo if controlled and unit_type is UnitType.WORKER else None,
    )


def enemy_core(
    identifier: int,
    position: tuple[int, int],
    *,
    hp: int = 5,
    shield: int = 5,
) -> CoreView:
    return CoreView(
        kind="CORE",
        id=uid(identifier),
        controlled=False,
        owner_username="enemy",
        position=position,
        hp=hp,
        shield=shield,
        state=CoreState.NORMAL,
        move_direction=None,
        move_progress=None,
        move_required_ticks=None,
        destination=None,
    )


def make_turn(
    *,
    tick: int = 1,
    core: CoreView | None = None,
    units: tuple[UnitView, ...] = (),
    enemies: tuple[UnitView | CoreView, ...] = (),
    resources: int = 5,
    resource_cells: tuple[tuple[int, int], ...] = (),
    obstacle_cells: tuple[tuple[int, int], ...] = (),
    events: tuple[ResolutionEvent, ...] = (),
    respawning: bool = False,
    submitter=None,
    beacon: ChampionBeacon | None = None,
) -> Turn:
    if core is None and not respawning:
        core = friendly_core()
    objects: list[TerrainView | CoreView | UnitView] = []
    if core is not None:
        objects.append(core)
    objects.extend(units)
    objects.extend(enemies)
    if resource_cells:
        objects.append(TerrainView(kind="RESOURCE", positions=resource_cells))
    if obstacle_cells:
        objects.append(TerrainView(kind="OBSTACLE", positions=obstacle_cells))
    state = PlayerState(
        status=PlayerStatus.RESPAWNING if respawning else PlayerStatus.ACTIVE,
        respawn_at_tick=tick + 1 if respawning else None,
        resources=resources,
        population=len(units),
        champion_beacon=beacon
        or ChampionBeacon(position=(100, 100), status=None, carrier_id=None),
        objects=tuple(objects),
        events=events,
    )
    if submitter is None:
        submitter = lambda plan, key: Accepted(
            accepted=True,
            tick=tick,
            source=CommandSource.AGENT,
            received_at=datetime.now(timezone.utc),
        )
    return Turn(tick=tick, state=state, submitter=submitter)
