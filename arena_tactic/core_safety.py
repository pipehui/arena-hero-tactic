from __future__ import annotations

from dataclasses import dataclass

from arena_hero import CoreState, Direction, Position, UnitType

from .config import TacticConfig
from .geometry import (
    cardinal_neighbors,
    count_open_neighbors,
    manhattan,
    ranger_firing_positions,
    ranger_line_is_clear,
)
from .models import (
    ActionIntent,
    CoreEvacuationCampaign,
    CoreMoveCandidateEvaluation,
    IntentAction,
    UnitMission,
    WorldModel,
)
from .planning import MoveViability, move_viability, route_to
from .projection import EnemyProjection, TacticalMap
from .rules import CORE_BASE_SHIELD_CAP, CORE_BEACON_SHIELD_CAP, CORE_MAX_HP, UNIT_MAX_HP
from .state import TacticMemory


@dataclass(frozen=True, slots=True)
class _CorePressure:
    enemies: tuple[EnemyProjection, ...]
    nearby: tuple[EnemyProjection, ...]
    immediate: int
    future: int
    guard_count: int
    trigger: bool
    reason: str | None


class CoreSafetyPlanner:
    """Continuous Core evacuation and worker-verified peaceful relocation."""

    def __init__(self, config: TacticConfig, memory: TacticMemory) -> None:
        self.config = config
        self.memory = memory
        self._candidate_evaluations: tuple[CoreMoveCandidateEvaluation, ...] = ()
        self._no_escape_route = False

    def intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> tuple[CoreEvacuationCampaign, list[ActionIntent]]:
        self._candidate_evaluations = ()
        self._no_escape_route = False
        if world.core is None:
            return CoreEvacuationCampaign(False, None, 0, None, None), []
        core = world.core
        pressure = self._pressure(world, projection)
        self._sync_campaign(world.tick, pressure)
        self._sync_relocation_clearance(pressure)
        if core.state is CoreState.MOVING:
            return self._campaign(), self._moving_intents(world, projection)

        if self.memory.evacuation_active:
            intents = self._evacuation_intents(world, projection, pressure)
            return self._campaign(), intents
        return self._campaign(), self._peaceful_intents(world, projection)

    def _pressure(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> _CorePressure:
        assert world.core is not None
        core = world.core
        enemies = tuple(
            enemy
            for enemy in projection.enemies
            if enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and enemy.age <= self.config.home_defense_hold_ticks
        )
        nearby = tuple(
            enemy
            for enemy in enemies
            if manhattan(enemy.observed_position, core.position)
            <= self.config.core_retreat_radius
        )
        immediate = projection.immediate_attackers(core.position)
        future = projection.future_attackers(core.position)
        guard_count = sum(
            unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and unit.hp * 2 > UNIT_MAX_HP[unit.unit_type]
            and manhattan(unit.position, core.position)
            <= self.config.core_retreat_radius
            for unit in world.friendlies
        )
        losses = len(self.memory.recent_combat_loss_ticks)
        trigger = (
            immediate >= self.config.core_retreat_projected_attackers
            or future >= self.config.core_retreat_projected_attackers
            or (len(nearby) >= 2 and len(nearby) >= guard_count)
            or (core.shield <= 3 and immediate > 0)
            or losses >= 2
        )
        reason = None
        if immediate >= self.config.core_retreat_projected_attackers:
            reason = "MULTIPLE_CORE_ATTACKERS"
        elif future >= self.config.core_retreat_projected_attackers:
            reason = "PROJECTED_CORE_ATTACKERS"
        elif losses >= 2:
            reason = "RECENT_GUARD_LOSSES"
        elif trigger:
            reason = "LOCAL_FORCE_DISADVANTAGE"
        return _CorePressure(
            enemies,
            nearby,
            immediate,
            future,
            guard_count,
            trigger,
            reason,
        )

    def _sync_campaign(self, tick: int, pressure: _CorePressure) -> None:
        if pressure.trigger:
            if not self.memory.evacuation_active:
                self.memory.evacuation_started_tick = tick
            self.memory.evacuation_active = True
            self.memory.evacuation_safe_ticks = 0
            self.memory.evacuation_reason = pressure.reason
            return
        if not self.memory.evacuation_active:
            return
        self.memory.evacuation_safe_ticks += 1
        if self.memory.evacuation_safe_ticks >= self.config.core_retreat_safe_ticks:
            self.memory.evacuation_active = False
            self.memory.evacuation_reason = "SAFE_CLEARANCE"

    def _sync_relocation_clearance(self, pressure: _CorePressure) -> None:
        if not self.memory.evacuation_active and not pressure.nearby:
            self.memory.strategic_relocation_safe_ticks += 1
        else:
            self.memory.strategic_relocation_safe_ticks = 0

    def _moving_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> list[ActionIntent]:
        assert world.core is not None
        destination = world.core.destination
        if destination is None or not self._moving_destination_invalid(
            world,
            projection,
            destination,
        ):
            return []
        self.memory.failed_core_destinations[destination] = (
            world.tick + self.config.core_move_failure_ttl
        )
        dead_end = any(
            evaluation.destination == destination and not evaluation.viable
            for evaluation in self._candidate_evaluations
        )
        return [
            ActionIntent.simple(
                None,
                IntentAction.CANCEL_MOVE,
                UnitMission.CORE_SURVIVAL,
                0,
                target_position=destination,
                reason=(
                    "CORE_DEAD_END_DESTINATION"
                    if dead_end
                    else "MIGRATION_DESTINATION_INVALID"
                ),
            )
        ]

    def _evacuation_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        pressure: _CorePressure,
    ) -> list[ActionIntent]:
        assert world.core is not None
        core = world.core
        move = self._evacuation_move(
            world,
            projection,
            pressure.nearby or pressure.enemies,
        )
        if move is not None:
            return [move]
        if core.hp < CORE_MAX_HP and pressure.immediate < core.hp + core.shield:
            return [
                ActionIntent.simple(
                    None,
                    IntentAction.HEAL,
                    UnitMission.CORE_SURVIVAL,
                    0,
                    resource_cost=min(
                        CORE_MAX_HP - core.hp,
                        max(1, world.resources),
                    ),
                    reason="EVACUATION_BLOCKED_HEAL",
                ),
                self._no_escape_wait(),
            ]
        if core.shield < CORE_BASE_SHIELD_CAP:
            return [
                ActionIntent.simple(
                    None,
                    IntentAction.REPAIR_SHIELD,
                    UnitMission.CORE_SURVIVAL,
                    5,
                    resource_cost=1,
                    reason="EVACUATION_BLOCKED_REPAIR",
                ),
                self._no_escape_wait(),
            ]
        return [self._no_escape_wait()]

    @staticmethod
    def _no_escape_wait() -> ActionIntent:
        return ActionIntent.simple(
            None,
            IntentAction.WAIT,
            UnitMission.CORE_SURVIVAL,
            6,
            reason="NO_CORE_ESCAPE_ROUTE",
            metadata=(("dead_end_rejected", True),),
        )

    def _peaceful_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> list[ActionIntent]:
        assert world.core is not None
        core = world.core
        intents: list[ActionIntent] = []
        if core.hp < CORE_MAX_HP:
            missing = CORE_MAX_HP - core.hp
            intents.append(
                ActionIntent.simple(
                    None,
                    IntentAction.HEAL,
                    UnitMission.CORE_SURVIVAL,
                    10 if core.hp <= 2 else 44,
                    resource_cost=min(missing, max(1, world.resources)),
                    reason=(
                        "CORE_CRITICAL_HEAL"
                        if core.hp <= 2
                        else "CORE_MAINTENANCE_HEAL"
                    ),
                )
            )
        elif (
            self.memory.strategic_relocation_pending
            and self.memory.strategic_relocation_safe_ticks >= 8
        ):
            relocation = self._strategic_relocation(world, projection)
            if relocation is not None:
                intents.append(relocation)
        else:
            beacon_carrier = (
                None
                if world.beacon.carrier_id is None
                else world.friendly(world.beacon.carrier_id)
            )
            # Core repair resolves after combat.  Do not reserve a repair
            # above the base cap when the Unit carrying the Beacon is already
            # certain to die in the immutable combat snapshot.
            owns_beacon = world.beacon.carrier_id == core.id or (
                beacon_carrier is not None
                and projection.immediate_attackers(beacon_carrier.position)
                < beacon_carrier.hp
            )
            shield_cap = (
                CORE_BEACON_SHIELD_CAP if owns_beacon else CORE_BASE_SHIELD_CAP
            )
            if core.shield < shield_cap:
                intents.append(
                    ActionIntent.simple(
                        None,
                        IntentAction.REPAIR_SHIELD,
                        UnitMission.CORE_SURVIVAL,
                        78 if core.shield >= 3 else 70,
                        resource_cost=1,
                        reason=(
                            "BEACON_SHIELD_BUILDUP"
                            if owns_beacon and core.shield >= CORE_BASE_SHIELD_CAP
                            else "CORE_SHIELD_MAINTENANCE"
                        ),
                    )
                )
        return intents

    def _evacuation_move(self, world, projection, enemies):
        assert world.core is not None
        center = self._enemy_center(enemies)
        previous = (
            self.memory.core_position_history[-2]
            if len(self.memory.core_position_history) >= 2
            else None
        )
        options: list[
            tuple[tuple[int, ...], Direction, Position, MoveViability, int]
        ] = []
        evaluations: list[CoreMoveCandidateEvaluation] = []
        occupied = dict(world.occupied_cells)
        for index, (direction, destination) in enumerate(cardinal_neighbors(world.core.position)):
            if (
                destination in world.known_obstacles
                or destination in world.visible_resources
                or destination in self.memory.failed_core_destinations
                or destination in projection.hostile_occupied
                or occupied.get(destination, 0) > 1
            ):
                continue
            dynamic_blocked = set(projection.hostile_occupied)
            dynamic_blocked.update(world.visible_resources)
            dynamic_blocked.update(
                cell for cell, count in occupied.items() if count > 1
            )
            viability = move_viability(
                world,
                world.core.position,
                destination,
                blocked=frozenset(dynamic_blocked),
                node_limit=min(self.config.path_node_limit, 256),
                require_open_area=True,
            )
            service_exits = sum(
                neighbor in world.known_passable
                and neighbor not in world.known_obstacles
                and neighbor not in projection.hostile_occupied
                and neighbor not in world.visible_resources
                for _, neighbor in cardinal_neighbors(destination)
            )
            evaluation = CoreMoveCandidateEvaluation(
                direction=direction,
                destination=destination,
                forward_exits=viability.forward_exits,
                local_open=viability.local_open,
                unknown_frontier=viability.unknown_frontier,
                service_exits=service_exits,
                viable=viability.viable,
                rejection_reason=(
                    "CORE_DEAD_END_DESTINATION"
                    if not viability.viable
                    else None
                ),
            )
            evaluations.append(evaluation)
            if not viability.viable:
                continue
            immediate = projection.immediate_attackers(destination)
            future = projection.future_attackers(destination)
            cover = sum(
                neighbor in world.known_obstacles
                for _, neighbor in cardinal_neighbors(destination)
            )
            cargo_follow = min(
                (
                    manhattan(destination, unit.position)
                    for unit in world.friendlies
                    if unit.unit_type is UnitType.WORKER and unit.cargo > 0
                ),
                default=0,
            )
            away = 0 if center is None else -manhattan(destination, center)
            score = (
                immediate,
                future,
                -max(viability.forward_exits, int(viability.unknown_frontier)),
                away,
                -cover,
                -service_exits,
                cargo_follow,
                int(previous is not None and destination == previous),
                index,
            )
            options.append((score, direction, destination, viability, service_exits))
        self._candidate_evaluations = tuple(evaluations)
        if not options:
            self._no_escape_route = True
            return None
        score, direction, destination, viability, service_exits = min(
            options, key=lambda row: row[0]
        )
        self.memory.last_core_move_destination = destination
        return ActionIntent(
            actor_id=None,
            action=IntentAction.START_MOVE,
            mission=UnitMission.CORE_SURVIVAL,
            priority=0,
            direction=direction,
            target_position=destination,
            reserve_positions=(destination,),
            tie_break=score,
            reason=self.memory.evacuation_reason or "EVACUATE",
            metadata=viability.metadata + (("service_exits", service_exits),),
        )

    def _moving_destination_invalid(self, world, projection, destination):
        assert world.core is not None
        hard_invalid = (
            destination in world.known_obstacles
            or destination in world.visible_resources
            or destination in projection.hostile_occupied
            or dict(world.occupied_cells).get(destination, 0) > 1
        )
        materially_worse = (
            projection.future_attackers(destination)
            > projection.future_attackers(world.core.position) + 1
        )
        dynamic_blocked = set(projection.hostile_occupied)
        dynamic_blocked.update(world.visible_resources)
        dynamic_blocked.update(
            cell for cell, count in world.occupied_cells if count > 1
        )
        viability = move_viability(
            world,
            world.core.position,
            destination,
            blocked=frozenset(dynamic_blocked),
            node_limit=min(self.config.path_node_limit, 256),
            require_open_area=True,
        )
        service_exits = sum(
            neighbor in world.known_passable
            and neighbor not in world.known_obstacles
            and neighbor not in projection.hostile_occupied
            and neighbor not in world.visible_resources
            for _, neighbor in cardinal_neighbors(destination)
        )
        self._candidate_evaluations = (
            CoreMoveCandidateEvaluation(
                direction=next(
                    direction
                    for direction, cell in cardinal_neighbors(world.core.position)
                    if cell == destination
                ),
                destination=destination,
                forward_exits=viability.forward_exits,
                local_open=viability.local_open,
                unknown_frontier=viability.unknown_frontier,
                service_exits=service_exits,
                viable=viability.viable,
                rejection_reason=(
                    "CORE_DEAD_END_DESTINATION"
                    if not viability.viable
                    else None
                ),
            ),
        )
        return hard_invalid or materially_worse or not viability.viable

    def _strategic_relocation(self, world, projection):
        assert world.core is not None
        goal = self.memory.strategic_relocation_goal
        if goal is None or not self._site_valid(world, projection, goal):
            site_visibility = dict(projection.last_visible_ticks)
            # Schema <=10 checkpoints stored a Worker-only verification map.
            # Merge it for compatibility, but every current role now writes
            # to the global last-visible map used here.
            for cell, tick in self.memory.worker_cell_last_visible.items():
                site_visibility[cell] = max(site_visibility.get(cell, 0), tick)
            candidates = [
                cell
                for cell, seen_tick in site_visibility.items()
                if world.tick - seen_tick <= self.config.resource_memory_ttl
                and self.config.strategic_site_min_distance
                <= manhattan(cell, world.core.position)
                <= self.config.strategic_site_max_distance
                and self._site_valid(world, projection, cell)
            ]
            threat = self.memory.recent_home_threat_position
            goal = min(
                candidates,
                key=lambda cell: (
                    0 if threat is None else -manhattan(cell, threat),
                    -sum(
                        neighbor in world.known_obstacles
                        for _, neighbor in cardinal_neighbors(cell)
                    ),
                    -count_open_neighbors(cell, world.known_obstacles),
                    cell,
                ),
                default=None,
            )
            self.memory.strategic_relocation_goal = goal
        if goal is None:
            return None
        if goal == world.core.position:
            self.memory.strategic_relocation_pending = False
            self.memory.strategic_relocation_goal = None
            return None
        route = route_to(
            world,
            world.core.position,
            goal,
            node_limit=self.config.path_node_limit,
            blocked=frozenset(
                set(projection.hostile_occupied)
                | {position for position, _ in world.remembered_resources}
            ),
        )
        if route is None or route.first_direction is None:
            return None
        viability = move_viability(
            world,
            world.core.position,
            route.first_position,
            target=goal,
            blocked=frozenset(
                set(projection.hostile_occupied)
                | {position for position, _ in world.remembered_resources}
            ),
            node_limit=min(self.config.path_node_limit, 512),
            require_continuation=True,
        )
        if not viability.viable:
            return None
        return ActionIntent(
            actor_id=None,
            action=IntentAction.START_MOVE,
            mission=UnitMission.CORE_SURVIVAL,
            priority=5,
            direction=route.first_direction,
            target_position=route.first_position,
            reserve_positions=(route.first_position,),
            tie_break=(route.distance,),
            reason="WORKER_VERIFIED_STRATEGIC_RELOCATION",
            metadata=viability.metadata,
        )

    def _site_valid(self, world, projection, cell):
        if (
            cell not in world.known_passable
            or cell in world.known_obstacles
            or cell in world.visible_resources
            or cell in projection.hostile_occupied
            or count_open_neighbors(cell, world.known_obstacles) < 2
        ):
            return False
        service_exit_count = sum(
            neighbor in world.known_passable
            and neighbor not in world.known_obstacles
            for _, neighbor in cardinal_neighbors(cell)
        )
        if service_exit_count < 2:
            return False
        return any(
            ranger_line_is_clear(firing, cell, world.known_obstacles)
            for firing in ranger_firing_positions(cell)
            if firing in world.known_passable
        )

    def _campaign(self) -> CoreEvacuationCampaign:
        return CoreEvacuationCampaign(
            active=self.memory.evacuation_active,
            started_tick=self.memory.evacuation_started_tick,
            safe_ticks=self.memory.evacuation_safe_ticks,
            last_destination=self.memory.last_core_move_destination,
            reason=self.memory.evacuation_reason,
            candidate_evaluations=self._candidate_evaluations,
            no_escape_route=self._no_escape_route,
        )

    @staticmethod
    def _enemy_center(enemies) -> Position | None:
        if not enemies:
            return None
        return (
            round(sum(enemy.position[0] for enemy in enemies) / len(enemies)),
            round(sum(enemy.position[1] for enemy in enemies) / len(enemies)),
        )
