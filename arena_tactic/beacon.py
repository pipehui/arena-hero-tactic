from __future__ import annotations

from arena_hero import BeaconStatus, CoreState, UnitType

from .config import TacticConfig
from .geometry import manhattan, manhattan_ring
from .models import ActionIntent, IntentAction, UnitMission, WorldModel
from .planning import move_viability, weighted_route_to
from .projection import TacticalMap
from .state import TacticMemory
from .rules import UNIT_MAX_HP
from .worker_safety import WorkerSafetyEvaluator


class BeaconPlanner:
    """Conservative local Champion Beacon acquisition.

    The public coordinate is not enough to infer that the Beacon is available;
    acquisition starts only from a currently visible ``GROUND`` state.  The
    planner never strips the home defense pool merely to chase the objective.
    """

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
        protected: frozenset,
    ) -> list[ActionIntent]:
        if world.core is None:
            self._clear_mission()
            return []
        if world.beacon.carrier_id is not None:
            self._clear_mission()
            return self._secure_friendly_carrier(
                world,
                projection,
                protected,
            )
        if world.beacon.status is not BeaconStatus.GROUND:
            return self._continue_mission(world, projection, protected)
        target = world.beacon.position
        if world.core.position == target and world.core.state is CoreState.NORMAL:
            return [
                ActionIntent.simple(
                    None,
                    IntentAction.PICKUP_BEACON,
                    UnitMission.BEACON,
                    65,
                    target_position=target,
                    reason="CORE_ON_GROUND_BEACON",
                )
            ]
        on_cell = tuple(
            sorted(
                (unit for unit in world.friendlies if unit.position == target),
                key=lambda unit: unit.id.bytes,
            )
        )
        if on_cell:
            actor = on_cell[0]
            return [
                ActionIntent.simple(
                    actor.id,
                    IntentAction.PICKUP_BEACON,
                    UnitMission.BEACON,
                    65,
                    target_position=target,
                    reason="UNIT_ON_GROUND_BEACON",
                )
            ]

        combat_threat = any(
            enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.position, world.core.position)
            <= self.config.home_warning_radius
            for enemy in world.enemies
        )
        workers = tuple(
            unit for unit in world.friendlies if unit.unit_type is UnitType.WORKER
        )
        if (
            combat_threat
            or len(workers) < self.config.beacon_min_workers
            or manhattan(world.core.position, target) > self.config.beacon_acquire_radius
        ):
            return []
        eligible = tuple(
            sorted(
                (
                    worker
                    for worker in workers
                    if worker.cargo == 0
                    and worker.hp >= UNIT_MAX_HP[UnitType.WORKER]
                    and projection.immediate_attackers(worker.position) == 0
                ),
                key=lambda worker: (manhattan(worker.position, target), worker.id.bytes),
            )
        )
        for worker in eligible:
            blocked = (projection.hostile_occupied | protected) - {
                worker.position,
                target,
            }
            route = weighted_route_to(
                world,
                worker.position,
                target,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(blocked),
                cell_costs=self.worker_safety.route_costs(projection),
            )
            if route is None or route.first_direction is None:
                continue
            viability = move_viability(
                world,
                worker.position,
                route.first_position,
                target=target,
                blocked=frozenset(blocked),
                node_limit=min(self.config.path_node_limit, 512),
                require_continuation=route.first_position != target,
                require_open_area=route.first_position == target,
            )
            if not viability.viable:
                continue
            if not self._manual_direction_allowed(
                worker.id,
                route.first_direction,
                world.tick,
            ):
                continue
            immediate, future, remembered = projection.exposure(route.first_position)
            if immediate or future:
                continue
            self.memory.beacon_mission_actor_id = worker.id
            self.memory.beacon_mission_target = target
            return [
                ActionIntent.move(
                    worker.id,
                    UnitMission.BEACON,
                    65,
                    route.first_direction,
                    route.first_position,
                    risk=remembered,
                    tie_break=(route.distance,),
                    reason="LOCAL_SAFE_BEACON_ACQUIRE",
                    metadata=viability.metadata,
                ),
                ActionIntent.simple(
                    worker.id,
                    IntentAction.WAIT,
                    UnitMission.BEACON,
                    66,
                    target_position=target,
                    reason="BEACON_APPROACH_BLOCKED_THIS_TICK",
                ),
            ]
        return []

    def _secure_friendly_carrier(
        self,
        world: WorldModel,
        projection: TacticalMap,
        protected: frozenset,
    ) -> list[ActionIntent]:
        assert world.core is not None
        carrier_id = world.beacon.carrier_id
        if carrier_id is None or carrier_id == world.core.id:
            return []
        actor = world.friendly(carrier_id)
        if actor is None:
            return []
        # A loaded Worker must complete the higher-priority cargo workflow;
        # its next empty Tick will re-enter Beacon security automatically.
        if actor.unit_type is UnitType.WORKER and actor.cargo > 0:
            return []
        if (
            actor.unit_type is UnitType.WORKER
            and actor.hp < UNIT_MAX_HP[UnitType.WORKER]
        ):
            return []
        core_position = world.core.destination or world.core.position
        safe_cells = tuple(
            cell
            for cell in manhattan_ring(core_position, self.config.beacon_guard_radius)
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in protected
            and cell not in projection.hostile_occupied
            and projection.immediate_attackers(cell) == 0
        )
        if actor.position in safe_cells:
            return [
                ActionIntent.simple(
                    actor.id,
                    IntentAction.WAIT,
                    UnitMission.BEACON,
                    60,
                    target_position=actor.position,
                    reason="BEACON_SECURED_NEAR_CORE",
                )
            ]
        blocked = (
            projection.hostile_occupied
            | protected
            | frozenset(projection.immediate_damage)
        ) - {actor.position}
        routes = []
        for target in safe_cells:
            route = weighted_route_to(
                world,
                actor.position,
                target,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(blocked - {target}),
                cell_costs=self.worker_safety.route_costs(projection),
            )
            if route is None or route.first_direction is None:
                continue
            viability = move_viability(
                world,
                actor.position,
                route.first_position,
                target=target,
                blocked=frozenset(blocked - {target}),
                node_limit=min(self.config.path_node_limit, 512),
                require_continuation=route.first_position != target,
                require_open_area=route.first_position == target,
            )
            if not viability.viable:
                continue
            if not self._manual_direction_allowed(
                actor.id,
                route.first_direction,
                world.tick,
            ):
                continue
            immediate, future, remembered = projection.exposure(
                route.first_position
            )
            if immediate:
                continue
            score = (
                future,
                remembered,
                route.distance,
                target,
            )
            routes.append((score, route, target))
        if routes:
            score, route, target = min(routes, key=lambda item: item[0])
            return [
                ActionIntent.move(
                    actor.id,
                    UnitMission.BEACON,
                    60,
                    route.first_direction,
                    route.first_position,
                    risk=score[0] * 10 + score[1],
                    tie_break=(route.distance,),
                    reason="SECURE_BEACON_NEAR_CORE",
                    metadata=(("guard_cell", target),) + viability.metadata,
                ),
                ActionIntent.simple(
                    actor.id,
                    IntentAction.WAIT,
                    UnitMission.BEACON,
                    61,
                    target_position=target,
                    reason="BEACON_RETURN_BLOCKED_THIS_TICK",
                ),
            ]
        return [
            ActionIntent.simple(
                actor.id,
                IntentAction.WAIT,
                UnitMission.BEACON,
                61,
                reason="NO_SAFE_BEACON_GUARD_ROUTE",
            )
        ]

    def _continue_mission(
        self,
        world: WorldModel,
        projection: TacticalMap,
        protected: frozenset,
    ) -> list[ActionIntent]:
        actor_id = self.memory.beacon_mission_actor_id
        target = self.memory.beacon_mission_target
        if actor_id is None or target is None or target != world.beacon.position:
            self._clear_mission()
            return []
        actor = world.friendly(actor_id)
        if (
            actor is None
            or actor.unit_type is not UnitType.WORKER
            or actor.cargo > 0
            or actor.hp < UNIT_MAX_HP[UnitType.WORKER]
            or projection.immediate_attackers(actor.position)
            or actor.position == target
        ):
            self._clear_mission()
            return []
        blocked = (projection.hostile_occupied | protected) - {
            actor.position,
            target,
        }
        route = weighted_route_to(
            world,
            actor.position,
            target,
            node_limit=self.config.path_node_limit,
            blocked=frozenset(blocked),
            cell_costs=self.worker_safety.route_costs(projection),
        )
        if route is None or route.first_direction is None:
            self._clear_mission()
            return []
        viability = move_viability(
            world,
            actor.position,
            route.first_position,
            target=target,
            blocked=frozenset(blocked),
            node_limit=min(self.config.path_node_limit, 512),
            require_continuation=route.first_position != target,
            require_open_area=route.first_position == target,
        )
        if not viability.viable:
            self._clear_mission()
            return []
        if not self._manual_direction_allowed(
            actor.id,
            route.first_direction,
            world.tick,
        ):
            return [
                ActionIntent.simple(
                    actor.id,
                    IntentAction.WAIT,
                    UnitMission.BEACON,
                    65,
                    target_position=target,
                    reason="BEACON_MANUAL_LEASE_HOLD",
                )
            ]
        immediate, future, remembered = projection.exposure(route.first_position)
        if immediate or future:
            self._clear_mission()
            return []
        return [
            ActionIntent.move(
                actor.id,
                UnitMission.BEACON,
                65,
                route.first_direction,
                route.first_position,
                risk=remembered,
                tie_break=(route.distance,),
                reason="CONTINUE_BEACON_APPROACH",
                metadata=viability.metadata,
            ),
            ActionIntent.simple(
                actor.id,
                IntentAction.WAIT,
                UnitMission.BEACON,
                66,
                target_position=target,
                reason="BEACON_APPROACH_BLOCKED_THIS_TICK",
            ),
        ]

    def _clear_mission(self) -> None:
        self.memory.beacon_mission_actor_id = None
        self.memory.beacon_mission_target = None

    def _manual_direction_allowed(self, unit_id, direction, tick) -> bool:
        lease = self.memory.manual_move_leases.get(unit_id)
        if lease is None or tick > lease.expires_tick:
            return True
        opposite = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }
        return direction.value != opposite[lease.direction.value]
