from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from uuid import UUID

from arena_hero import Direction, Position, UnitType

from .config import TacticConfig
from .geometry import (
    DIRECTION_ORDER,
    cardinal_neighbors,
    direction_between,
    manhattan,
    ranger_firing_positions,
    ranger_line_is_clear,
)
from .models import (
    ActionIntent,
    EntitySnapshot,
    FireMission,
    IntentAction,
    ShotPlan,
    UnitMission,
    WorldModel,
)
from .projection import TacticalMap
from .planning import route_from_field, route_to, weighted_distance_field
from .rules import UNIT_MAX_HP
from .state import TacticMemory


@dataclass(slots=True)
class _FireAllocation:
    rangers: tuple[EntitySnapshot, ...]
    targets: tuple[EntitySnapshot, ...]
    missions: list[FireMission]
    legal: list[tuple[UUID, UUID, Position]]
    candidates_by_target: dict[UUID, tuple[Position, ...]]
    available: set[UUID]
    ranger_by_id: dict[UUID, EntitySnapshot]
    assignments: dict[UUID, tuple[UUID, Position, str]]
    mission_assignments: defaultdict[UUID, list[tuple[UUID, Position]]]
    core_fire_assignments: list[tuple[UUID, UUID, Position]]


class CombatPlanner:
    """Global Ranger fire control and Vanguard interception."""

    def __init__(self, config: TacticConfig, memory: TacticMemory) -> None:
        self.config = config
        self.memory = memory

    def sync_engagements(self, world: WorldModel, projection: TacticalMap) -> None:
        if world.core is None:
            self.memory.engaged_enemy_until.clear()
            return
        for enemy in world.enemies:
            if enemy.unit_type not in {UnitType.VANGUARD, UnitType.RANGER}:
                continue
            if self.target_is_urgent(world, projection, enemy, include_engagement=False):
                self.memory.engaged_enemy_until[enemy.id] = world.tick + 4
        for enemy_id, until in tuple(self.memory.engaged_enemy_until.items()):
            if until < world.tick:
                self.memory.engaged_enemy_until.pop(enemy_id, None)
        for key, feedback in tuple(self.memory.ranger_shot_feedback.items()):
            if world.tick > feedback.suppressed_until + self.config.ranger_miss_suppress_ticks:
                self.memory.ranger_shot_feedback.pop(key, None)
        for key, feedback in tuple(self.memory.vanguard_sweep_feedback.items()):
            if world.tick > feedback.suppressed_until + self.config.ranger_miss_suppress_ticks:
                self.memory.vanguard_sweep_feedback.pop(key, None)

    def fire_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> tuple[
        tuple[FireMission, ...],
        list[ActionIntent],
        tuple[tuple[UUID, UUID, Position], ...],
    ]:
        if world.core is None:
            self.memory.last_ranger_shots.clear()
            return (), [], ()
        state = self._prepare_fire_allocation(world, projection)
        self._allocate_urgent_packages(world, projection, state)
        self._allocate_urgent_remainders(world, projection, state)
        self._allocate_enemy_core_fire(world, state)
        self._allocate_opportunistic_fire(world, projection, state)

        intents, shot_plans = self._assigned_fire_intents(state)
        last_stand, last_stand_plans = self._last_stand_fire(
            world,
            projection,
            state,
        )
        intents.extend(last_stand)
        shot_plans.update(last_stand_plans)
        intents.extend(self._enemy_core_fire_intents(state))
        intents.extend(self._firing_stance_intents(world, projection, state))

        self.memory.last_ranger_shots = shot_plans
        return self._resolved_fire_missions(state), intents, tuple(state.legal)

    def _prepare_fire_allocation(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> _FireAllocation:
        rangers = tuple(
            unit for unit in world.friendlies if unit.unit_type is UnitType.RANGER
        )
        targets = tuple(
            sorted(
                world.enemies,
                key=lambda enemy: self.target_priority(world, projection, enemy),
            )
        )
        missions: list[FireMission] = []
        legal: list[tuple[UUID, UUID, Position]] = []
        candidates_by_target: dict[UUID, tuple[Position, ...]] = {}
        for enemy in targets:
            cells, confidence = self.enemy_candidate_cells(world, projection, enemy)
            urgent = self.target_is_urgent(world, projection, enemy)
            candidates_by_target[enemy.id] = cells
            for ranger in rangers:
                if (
                    ranger.hp * 2 <= UNIT_MAX_HP[UnitType.RANGER]
                    or not urgent
                    or enemy.unit_type is UnitType.WORKER
                ):
                    continue
                legal.extend(
                    (ranger.id, enemy.id, cell)
                    for cell in cells
                    if ranger_line_is_clear(
                        ranger.position,
                        cell,
                        world.known_obstacles,
                    )
                )
            missions.append(
                FireMission(
                    target_id=enemy.id,
                    target_type=enemy.unit_type,
                    target_kind="UNIT",
                    urgent=urgent,
                    confidence=confidence,
                    candidate_cells=cells,
                    required_hits=enemy.hp,
                )
            )
        return _FireAllocation(
            rangers=rangers,
            targets=targets,
            missions=missions,
            legal=legal,
            candidates_by_target=candidates_by_target,
            available={
                ranger.id
                for ranger in rangers
                if ranger.hp * 2 > UNIT_MAX_HP[UnitType.RANGER]
            },
            ranger_by_id={ranger.id: ranger for ranger in rangers},
            assignments={},
            mission_assignments=defaultdict(list),
            core_fire_assignments=[],
        )

    def _allocate_urgent_packages(self, world, projection, state) -> None:
        for mission in sorted(
            state.missions,
            key=lambda item: (
                not item.urgent,
                self.target_priority(world, projection, world.enemy(item.target_id)),
            ),
        ):
            if mission.urgent:
                self._assign_kill_packages(
                    world,
                    mission,
                    state.ranger_by_id,
                    state.available,
                    state.assignments,
                    state.mission_assignments,
                )

    def _allocate_urgent_remainders(self, world, projection, state) -> None:
        urgent_missions = {
            mission.target_id: mission
            for mission in state.missions
            if mission.urgent and mission.target_type is not UnitType.WORKER
        }
        for ranger_id in sorted(tuple(state.available), key=lambda item: item.bytes):
            ranger = state.ranger_by_id[ranger_id]
            options = []
            for target_id, mission in urgent_missions.items():
                coverage: dict[Position, int] = defaultdict(int)
                for _, assigned_cell in state.mission_assignments.get(target_id, ()):
                    coverage[assigned_cell] += 1
                primary = mission.candidate_cells[0]
                primary_is_lethal = coverage[primary] >= mission.required_hits
                for index, cell in enumerate(mission.candidate_cells):
                    if primary_is_lethal and index == 0:
                        continue
                    if not ranger_line_is_clear(
                        ranger.position,
                        cell,
                        world.known_obstacles,
                    ):
                        continue
                    options.append(
                        (
                            0 if index == 0 or primary_is_lethal else 1,
                            coverage[cell],
                            self.target_priority(
                                world,
                                projection,
                                world.enemy(target_id),
                            ),
                            index,
                            target_id,
                            cell,
                        )
                    )
            if not options:
                continue
            _, _, _, index, target_id, cell = min(options)
            reason = "URGENT_CROSS_COVERAGE" if index > 0 else "URGENT_REMAINDER"
            state.assignments[ranger_id] = target_id, cell, reason
            state.mission_assignments[target_id].append((ranger_id, cell))
            state.available.remove(ranger_id)

    def _allocate_enemy_core_fire(self, world, state) -> None:
        assert world.core is not None
        for enemy_core in sorted(
            world.enemy_cores,
            key=lambda item: (
                manhattan(item.position, world.core.position),
                item.hp + item.shield,
                item.id.bytes,
            ),
        ):
            expected = enemy_core.position
            if (
                enemy_core.destination is not None
                and enemy_core.move_progress is not None
                and enemy_core.move_required_ticks is not None
                and enemy_core.move_progress >= enemy_core.move_required_ticks - 1
            ):
                expected = enemy_core.destination
            shooters = [
                state.ranger_by_id[ranger_id]
                for ranger_id in sorted(tuple(state.available), key=lambda item: item.bytes)
                if ranger_line_is_clear(
                    state.ranger_by_id[ranger_id].position,
                    expected,
                    world.known_obstacles,
                )
            ][: enemy_core.hp + enemy_core.shield]
            assigned = tuple((ranger.id, expected) for ranger in shooters)
            for ranger in shooters:
                state.core_fire_assignments.append(
                    (ranger.id, enemy_core.id, expected)
                )
                state.available.remove(ranger.id)
            if assigned:
                state.missions.append(
                    FireMission(
                        target_id=enemy_core.id,
                        target_type=None,
                        target_kind="CORE",
                        urgent=False,
                        confidence="EXACT",
                        candidate_cells=(expected,),
                        required_hits=enemy_core.hp + enemy_core.shield,
                        assigned_shooters=tuple(item[0] for item in assigned),
                        assignments=assigned,
                    )
                )

    def _allocate_opportunistic_fire(self, world, projection, state) -> None:
        missions = {mission.target_id: mission for mission in state.missions}
        for ranger_id in sorted(tuple(state.available), key=lambda item: item.bytes):
            ranger = state.ranger_by_id[ranger_id]
            options = []
            for enemy in state.targets:
                if (
                    enemy.unit_type is UnitType.WORKER
                    and len(state.mission_assignments.get(enemy.id, ())) >= enemy.hp
                ):
                    continue
                mission = missions[enemy.id]
                coverage: dict[Position, int] = defaultdict(int)
                for _, assigned_cell in state.mission_assignments.get(enemy.id, ()):
                    coverage[assigned_cell] += 1
                for index, cell in enumerate(state.candidates_by_target[enemy.id]):
                    if not ranger_line_is_clear(
                        ranger.position,
                        cell,
                        world.known_obstacles,
                    ):
                        continue
                    # High-confidence movement deserves concentrated fire on
                    # its top prediction.  Otherwise spread opportunistic
                    # Rangers over distinct legal outcomes before duplicating
                    # an already-covered empty cell.
                    spread_rank = (
                        0 if mission.confidence == "HIGH" else coverage[cell]
                    )
                    options.append(
                        (
                            self.target_priority(world, projection, enemy),
                            spread_rank,
                            index,
                            enemy.id,
                            cell,
                        )
                    )
            if not options:
                continue
            _, _, _, target_id, cell = min(options)
            state.assignments[ranger_id] = target_id, cell, "OPPORTUNISTIC_FIRE"
            state.mission_assignments[target_id].append((ranger_id, cell))
            state.available.remove(ranger_id)

    @staticmethod
    def _assigned_fire_intents(state):
        intents: list[ActionIntent] = []
        plans: dict[UUID, ShotPlan] = {}
        for ranger_id, (target_id, cell, reason) in sorted(
            state.assignments.items(),
            key=lambda item: item[0].bytes,
        ):
            intents.append(
                ActionIntent.simple(
                    ranger_id,
                    IntentAction.SHOOT_CELL,
                    UnitMission.ATTACK,
                    30 if reason != "OPPORTUNISTIC_FIRE" else 50,
                    target_id=target_id,
                    expected_cell=cell,
                    target_position=cell,
                    reason=reason,
                )
            )
            plans[ranger_id] = ShotPlan(ranger_id, target_id, cell)
        return intents, plans

    def _last_stand_fire(self, world, projection, state):
        intents: list[ActionIntent] = []
        plans: dict[UUID, ShotPlan] = {}
        for ranger in state.rangers:
            if ranger.hp * 2 > UNIT_MAX_HP[UnitType.RANGER]:
                continue
            current_attackers = projection.immediate_attackers(ranger.position)
            if current_attackers == 0 or any(
                destination not in world.known_obstacles
                and destination not in projection.hostile_occupied
                and projection.immediate_attackers(destination) < current_attackers
                and projection.immediate_attackers(destination) < ranger.hp
                for _, destination in cardinal_neighbors(ranger.position)
            ):
                continue
            options = [
                (
                    self.target_priority(world, projection, enemy),
                    index,
                    enemy.id,
                    cell,
                )
                for enemy in state.targets
                if self.target_is_urgent(world, projection, enemy)
                for index, cell in enumerate(state.candidates_by_target[enemy.id])
                if ranger_line_is_clear(
                    ranger.position,
                    cell,
                    world.known_obstacles,
                )
            ]
            if not options:
                continue
            _, _, target_id, cell = min(options)
            intents.append(
                ActionIntent.simple(
                    ranger.id,
                    IntentAction.SHOOT_CELL,
                    UnitMission.ATTACK,
                    21,
                    target_id=target_id,
                    expected_cell=cell,
                    target_position=cell,
                    reason="LAST_STAND_FIRE",
                )
            )
            plans[ranger.id] = ShotPlan(ranger.id, target_id, cell)
        return intents, plans

    @staticmethod
    def _enemy_core_fire_intents(state):
        return [
            ActionIntent.simple(
                ranger_id,
                IntentAction.SHOOT,
                UnitMission.ATTACK,
                50,
                target_id=target_id,
                expected_cell=expected,
                target_position=expected,
                reason="VISIBLE_ENEMY_CORE_FIRE",
            )
            for ranger_id, target_id, expected in state.core_fire_assignments
        ]

    def _firing_stance_intents(self, world, projection, state):
        assert world.core is not None
        intents: list[ActionIntent] = []
        urgent_missions = tuple(
            mission
            for mission in state.missions
            if mission.urgent
            and (
                mission.target_type is not UnitType.WORKER
                or len(state.mission_assignments.get(mission.target_id, ()))
                < mission.required_hits
            )
        )
        for ranger_id in sorted(state.available, key=lambda item: item.bytes):
            ranger = state.ranger_by_id[ranger_id]
            blocked = frozenset(
                (
                    projection.hostile_occupied
                    | projection.service_positions
                )
                - {ranger.position}
            )
            distances, parents = weighted_distance_field(
                world,
                ranger.position,
                node_limit=self.config.path_node_limit,
                blocked=blocked,
            )
            options: dict[
                tuple[Direction, Position],
                tuple[tuple[int, ...], UUID, Position],
            ] = {}
            ranked_missions = sorted(
                urgent_missions,
                key=lambda mission: self.target_priority(
                    world,
                    projection,
                    world.enemy(mission.target_id),
                ),
            )
            for mission_rank, mission in enumerate(ranked_missions):
                enemy = world.enemy(mission.target_id)
                if enemy is None:
                    continue
                stance_cells = {
                    stance
                    for candidate in mission.candidate_cells
                    for stance in ranger_firing_positions(candidate)
                }
                for stance in stance_cells:
                    if (
                        stance not in world.known_passable
                        or stance in world.known_obstacles
                        or stance in projection.hostile_occupied
                        or stance in projection.service_positions
                        or manhattan(stance, world.core.position)
                        > self.config.home_pursuit_radius
                    ):
                        continue
                    coverage = sum(
                        ranger_line_is_clear(stance, cell, world.known_obstacles)
                        for cell in mission.candidate_cells
                    )
                    if not coverage:
                        continue
                    ranked_coverage = tuple(
                        index
                        for index, cell in enumerate(mission.candidate_cells)
                        if ranger_line_is_clear(stance, cell, world.known_obstacles)
                    )
                    route = route_from_field(
                        ranger.position,
                        stance,
                        distances,
                        parents,
                        obstacles=world.known_obstacles,
                    )
                    if route is None or route.first_direction is None:
                        continue
                    destination = route.first_position
                    direction = route.first_direction
                    if destination is None:
                        continue
                    score = (
                        mission_rank,
                        min(ranked_coverage),
                        -coverage,
                        projection.immediate_attackers(destination),
                        projection.future_attackers(destination),
                        route.distance,
                        manhattan(stance, world.core.position),
                        stance[0],
                        stance[1],
                        DIRECTION_ORDER.index(direction),
                    )
                    key = direction, destination
                    previous = options.get(key)
                    row = score, mission.target_id, stance
                    if previous is None or row[0] < previous[0]:
                        options[key] = row
            ranked = sorted(
                (
                    (score, direction, destination, target_id, stance)
                    for (direction, destination), (score, target_id, stance)
                    in options.items()
                ),
                key=lambda row: row[0],
            )
            for score, direction, destination, target_id, stance in ranked[:6]:
                intents.append(
                    ActionIntent.move(
                        ranger.id,
                        UnitMission.HOME_DEFENSE,
                        52,
                        direction,
                        destination,
                        risk=score[3] * 100 + score[4] * 10,
                        exclusive_destination=True,
                        tie_break=score,
                        reason="ADVANCE_TO_DYNAMIC_FIRE_LINE",
                        metadata=(
                            ("target_id", str(target_id)),
                            ("firing_stance", stance),
                            ("candidate_coverage", -score[2]),
                            ("route_distance", score[5]),
                        ),
                    )
                )
        return intents

    @staticmethod
    def _resolved_fire_missions(state):
        return tuple(
            replace(
                mission,
                assigned_shooters=tuple(
                    shooter
                    for shooter, _ in state.mission_assignments.get(
                        mission.target_id,
                        (),
                    )
                )
                if mission.target_kind == "UNIT"
                else mission.assigned_shooters,
                assignments=tuple(
                    state.mission_assignments.get(mission.target_id, ())
                )
                if mission.target_kind == "UNIT"
                else mission.assignments,
            )
            for mission in state.missions
        )

    def _assign_kill_packages(
        self,
        world: WorldModel,
        mission: FireMission,
        rangers: dict[UUID, EntitySnapshot],
        available: set[UUID],
        assignments: dict[UUID, tuple[UUID, Position, str]],
        mission_assignments: dict[UUID, list[tuple[UUID, Position]]],
    ) -> None:
        for candidate_index, cell in enumerate(mission.candidate_cells):
            shooters = sorted(
                (
                    ranger
                    for ranger_id, ranger in rangers.items()
                    if ranger_id in available
                    and ranger_line_is_clear(ranger.position, cell, world.known_obstacles)
                    and not self._shot_suppressed(
                        world.tick,
                        mission,
                        cell,
                    )
                ),
                key=lambda ranger: (
                    sum(
                        ranger_line_is_clear(
                            ranger.position,
                            candidate,
                            world.known_obstacles,
                        )
                        for candidate in mission.candidate_cells
                    ),
                    manhattan(ranger.position, cell),
                    ranger.id.bytes,
                ),
            )
            if len(shooters) >= mission.required_hits:
                package = shooters[: mission.required_hits]
            elif candidate_index == 0 and shooters:
                package = shooters
            else:
                continue
            for ranger in package:
                assignments[ranger.id] = mission.target_id, cell, "LETHAL_FIRE_PACKAGE"
                mission_assignments[mission.target_id].append((ranger.id, cell))
                available.remove(ranger.id)
            if mission.confidence == "HIGH":
                break

    def _shot_suppressed(
        self,
        tick: int,
        mission: FireMission,
        cell: Position,
    ) -> bool:
        if mission.confidence == "HIGH":
            return False
        feedback = self.memory.ranger_shot_feedback.get((mission.target_id, cell))
        return bool(
            feedback is not None
            and feedback.misses >= self.config.ranger_repeat_miss_limit
            and tick <= feedback.suppressed_until + self.config.ranger_miss_suppress_ticks
        )

    def enemy_candidate_cells(
        self,
        world: WorldModel,
        projection: TacticalMap,
        enemy: EntitySnapshot,
    ) -> tuple[tuple[Position, ...], str]:
        projected = projection.enemy(enemy.id)
        legal = list(projected.possible_positions if projected is not None else (enemy.position,))
        track = world.track(enemy.id)
        ranked: list[Position] = [enemy.position]
        confidence = "LOW"
        if track is not None and len(track.samples) >= 3:
            a, b, c = (sample[1] for sample in track.samples[-3:])
            if a == b == c:
                ranked = [c]
                confidence = "HIGH"
            elif a == c and a != b:
                ranked = [b, c]
                confidence = "HIGH"
            elif a == b and b != c:
                delta = c[0] - b[0], c[1] - b[1]
                ranked = [(c[0] + delta[0], c[1] + delta[1]), c, a]
                confidence = "MEDIUM"
            elif b == c and a != b:
                delta = b[0] - a[0], b[1] - a[1]
                ranked = [(c[0] + delta[0], c[1] + delta[1]), c, a]
                confidence = "MEDIUM"
            else:
                first_delta = b[0] - a[0], b[1] - a[1]
                second_delta = c[0] - b[0], c[1] - b[1]
                if first_delta == second_delta:
                    ranked = [(c[0] + second_delta[0], c[1] + second_delta[1]), c]
                    confidence = "HIGH"
                else:
                    ranked = [c]
        elif track is not None and len(track.samples) >= 2:
            previous, current = track.samples[-2][1], track.samples[-1][1]
            if previous == current:
                ranked = [current]
                confidence = "HIGH"
            else:
                delta = current[0] - previous[0], current[1] - previous[1]
                ranked = [(current[0] + delta[0], current[1] + delta[1]), current, previous]
                confidence = "MEDIUM"
        if world.core is not None and (
            self._attacks(enemy, world.core.position, world)
            or any(
                self._attacks(enemy, unit.position, world)
                for unit in world.friendlies
            )
        ):
            # Attacking consumes the enemy's whole action, so a unit with a
            # valuable current shot has a strong reason to remain in place.
            ranked = [enemy.position, *ranked]
            if confidence == "LOW":
                confidence = "MEDIUM"
        ordered: list[Position] = []
        for cell in (*ranked, *legal):
            if cell in legal and cell not in ordered:
                ordered.append(cell)
        feedback = self.memory.ranger_shot_feedback
        original_rank = {cell: index for index, cell in enumerate(ordered)}
        current_attacks = self._enemy_attacks_any_friendly(world, enemy, enemy.position)
        track_stationary = bool(
            track is not None
            and len(track.samples) >= 3
            and len({sample[1] for sample in track.samples[-3:]}) == 1
        )
        track_moved = bool(
            track is not None
            and len(track.samples) >= 2
            and track.samples[-2][1] != track.samples[-1][1]
        )
        mobile_target = (
            enemy.unit_type in {UnitType.WORKER, UnitType.VANGUARD}
            and not current_attacks
            and track_moved
            and not track_stationary
            and any(cell != enemy.position for cell in ordered)
        )
        ordered.sort(
            key=lambda cell: (
                int(
                    (enemy.id, cell) in feedback
                    and feedback[(enemy.id, cell)].misses
                    >= self.config.ranger_repeat_miss_limit
                    and confidence != "HIGH"
                ),
                int(mobile_target and cell == enemy.position),
                (
                    -self._enemy_action_value(world, enemy, cell)
                    if enemy.unit_type is UnitType.VANGUARD and mobile_target
                    else 0
                ),
                (
                    self._enemy_protected_distance(world, cell)
                    if enemy.unit_type is UnitType.VANGUARD and mobile_target
                    else 0
                ),
                original_rank[cell],
            )
        )
        return tuple(ordered[:5]), confidence

    def _enemy_attacks_any_friendly(
        self,
        world: WorldModel,
        enemy: EntitySnapshot,
        position: Position,
    ) -> bool:
        if world.core is not None and self._attacks_from(
            enemy.unit_type, position, world.core.position, world
        ):
            return True
        return any(
            self._attacks_from(enemy.unit_type, position, unit.position, world)
            for unit in world.friendlies
        )

    def _enemy_action_value(
        self,
        world: WorldModel,
        enemy: EntitySnapshot,
        position: Position,
    ) -> int:
        """Estimate the payoff of WAIT/attack versus a one-cell advance."""

        value = 0
        if world.core is not None and self._attacks_from(
            enemy.unit_type, position, world.core.position, world
        ):
            value += 12
        for unit in world.friendlies:
            if not self._attacks_from(
                enemy.unit_type, position, unit.position, world
            ):
                continue
            value += {
                UnitType.WORKER: 7,
                UnitType.RANGER: 6,
                UnitType.VANGUARD: 4,
            }[unit.unit_type]
        return value

    @staticmethod
    def _enemy_protected_distance(world: WorldModel, position: Position) -> int:
        targets = [unit.position for unit in world.friendlies]
        if world.core is not None:
            targets.append(world.core.position)
        return min((manhattan(position, cell) for cell in targets), default=0)

    @staticmethod
    def _attacks_from(
        unit_type: UnitType,
        source: Position,
        target: Position,
        world: WorldModel,
    ) -> bool:
        if unit_type is UnitType.VANGUARD:
            return manhattan(source, target) == 1
        if unit_type is UnitType.RANGER:
            return ranger_line_is_clear(source, target, world.known_obstacles)
        return False

    def target_priority(
        self,
        world: WorldModel,
        projection: TacticalMap,
        enemy: EntitySnapshot | None,
    ) -> tuple[Any, ...]:
        if enemy is None or world.core is None:
            return (99,)
        attacks_core = projection.immediate_attackers(world.core.position) > 0 and self._attacks(
            enemy, world.core.position, world
        )
        attacks_friend = any(self._attacks(enemy, unit.position, world) for unit in world.friendlies)
        carries_beacon = world.beacon.carrier_id == enemy.id
        type_rank = {
            UnitType.RANGER: 0,
            UnitType.VANGUARD: 1,
            UnitType.WORKER: 2,
        }[enemy.unit_type]
        return (
            not attacks_core,
            not attacks_friend,
            not carries_beacon,
            enemy.hp,
            type_rank,
            manhattan(enemy.position, world.core.position),
            enemy.id.bytes,
        )

    def target_is_urgent(
        self,
        world: WorldModel,
        projection: TacticalMap,
        enemy: EntitySnapshot,
        *,
        include_engagement: bool = True,
    ) -> bool:
        if world.core is None:
            return False
        if enemy.unit_type is UnitType.WORKER:
            return (
                manhattan(enemy.position, world.core.position)
                <= self.config.home_engage_radius
            )
        return (
            self._attacks(enemy, world.core.position, world)
            or any(self._attacks(enemy, unit.position, world) for unit in world.friendlies)
            or manhattan(enemy.position, world.core.position) <= self.config.home_engage_radius
            or (
                include_engagement
                and self.memory.engaged_enemy_until.get(enemy.id, 0) >= world.tick
                and manhattan(enemy.position, world.core.position)
                <= self.config.home_pursuit_radius
            )
        )

    @staticmethod
    def _attacks(enemy: EntitySnapshot, cell: Position, world: WorldModel) -> bool:
        if enemy.unit_type is UnitType.VANGUARD:
            return manhattan(enemy.position, cell) == 1
        if enemy.unit_type is UnitType.RANGER:
            return ranger_line_is_clear(enemy.position, cell, world.known_obstacles)
        return False

    def vanguard_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> list[ActionIntent]:
        if world.core is None:
            return []
        urgent = tuple(
            sorted(
                (
                    enemy
                    for enemy in world.enemies
                    if self.target_is_urgent(world, projection, enemy)
                ),
                key=lambda enemy: self.target_priority(world, projection, enemy),
            )
        )
        intents: list[ActionIntent] = []
        predicted_sweeps: dict[UUID, ShotPlan] = {}
        responder_counts: dict[UUID, int] = defaultdict(int)
        protected_units = tuple(
            unit
            for unit in world.friendlies
            if unit.unit_type in {UnitType.WORKER, UnitType.RANGER}
        )
        for vanguard in (
            unit for unit in world.friendlies if unit.unit_type is UnitType.VANGUARD
        ):
            adjacent = self._adjacent_sweep(vanguard, urgent)
            if adjacent is not None:
                intents.append(adjacent)
                continue
            if not urgent:
                continue
            target = self._vanguard_target(
                world,
                projection,
                vanguard,
                urgent,
                responder_counts,
            )
            if target is None:
                continue
            responder_counts[target.id] += 1
            candidate_cells, confidence = self.enemy_candidate_cells(world, projection, target)
            predicted = self._predicted_vanguard_sweep(
                world,
                vanguard,
                target,
                candidate_cells,
                confidence,
            )
            if predicted is not None:
                sweep_intent, sweep_plan = predicted
                intents.append(sweep_intent)
                predicted_sweeps[vanguard.id] = sweep_plan
            intents.extend(
                self._vanguard_intercept_intents(
                    world,
                    projection,
                    vanguard,
                    target,
                    candidate_cells,
                    protected_units,
                )
            )
        self.memory.last_vanguard_sweeps = predicted_sweeps
        return intents

    @staticmethod
    def _adjacent_sweep(vanguard, urgent):
        adjacent = next(
            (
                enemy
                for enemy in urgent
                if manhattan(vanguard.position, enemy.position) == 1
            ),
            None,
        )
        if adjacent is None:
            return None
        direction = direction_between(vanguard.position, adjacent.position)
        if direction is None:
            return None
        return ActionIntent(
            actor_id=vanguard.id,
            action=IntentAction.SWEEP,
            mission=UnitMission.ATTACK,
            priority=30,
            direction=direction,
            target_id=adjacent.id,
            target_position=adjacent.position,
            reason="ADJACENT_ENEMY_SWEEP",
        )

    def _vanguard_target(
        self,
        world,
        projection,
        vanguard,
        urgent,
        responder_counts,
    ):
        assert world.core is not None
        eligible = tuple(
            enemy
            for enemy in urgent
            if responder_counts[enemy.id]
            < (2 if enemy.unit_type in {UnitType.WORKER, UnitType.RANGER} else 3)
            and (
                manhattan(enemy.position, world.core.position)
                <= self.config.home_engage_radius
                or manhattan(vanguard.position, enemy.position)
                <= self.config.vanguard_engage_distance
            )
        )
        return min(
            eligible,
            key=lambda enemy: (
                self.target_priority(world, projection, enemy),
                manhattan(vanguard.position, enemy.position),
            ),
            default=None,
        )

    def _predicted_vanguard_sweep(
        self,
        world,
        vanguard,
        target,
        candidate_cells,
        confidence,
    ):
        if confidence != "HIGH" or len(candidate_cells) > 2:
            return None
        predicted = next(
            (
                cell
                for cell in candidate_cells
                if manhattan(vanguard.position, cell) == 1
            ),
            None,
        )
        if predicted is None or self._sweep_suppressed(
            world.tick,
            target.id,
            predicted,
        ):
            return None
        direction = direction_between(vanguard.position, predicted)
        if direction is None:
            return None
        return (
            ActionIntent(
                actor_id=vanguard.id,
                action=IntentAction.SWEEP,
                mission=UnitMission.ATTACK,
                priority=31,
                direction=direction,
                target_id=target.id,
                target_position=predicted,
                reason="HIGH_CONFIDENCE_INTERCEPT_SWEEP",
            ),
            ShotPlan(vanguard.id, target.id, predicted),
        )

    def _vanguard_intercept_intents(
        self,
        world,
        projection,
        vanguard,
        target,
        candidate_cells,
        protected_units,
    ):
        assert world.core is not None
        protected = min(
            (world.core.position, *(unit.position for unit in protected_units)),
            key=lambda cell: manhattan(target.position, cell),
        )
        options: list[tuple[tuple[int, ...], Direction, Position]] = []
        reposition: list[tuple[tuple[int, ...], Direction, Position]] = []
        occupied = projection.occupied_cells
        current_distance = min(
            manhattan(vanguard.position, cell) for cell in candidate_cells
        )
        current_route_block = min(
            manhattan(cell, vanguard.position)
            + manhattan(vanguard.position, protected)
            for cell in candidate_cells
        )
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(vanguard.position)
        ):
            if (
                destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or destination in projection.service_positions
            ):
                continue
            immediate = projection.immediate_attackers(destination)
            future = projection.future_attackers(destination)
            exposure_budget = (
                1
                if vanguard.hp == UNIT_MAX_HP[UnitType.VANGUARD]
                else 0
            )
            if immediate >= vanguard.hp or immediate > exposure_budget:
                continue
            route_block = min(
                manhattan(cell, destination) + manhattan(destination, protected)
                for cell in candidate_cells
            )
            candidate_distance = min(
                manhattan(destination, cell) for cell in candidate_cells
            )
            improves_intercept = (
                candidate_distance < current_distance
                or route_block < current_route_block
            )
            if improves_intercept:
                score = (
                    candidate_distance,
                    route_block,
                    immediate,
                    future,
                    occupied.get(destination, 0),
                    self.memory.congestion_counts.get(destination, 0),
                    index,
                )
                options.append((score, direction, destination))
            elif (
                candidate_distance <= current_distance + 1
                and route_block <= current_route_block + 2
                and manhattan(destination, world.core.position)
                <= self.config.home_pursuit_radius
            ):
                # When the best intercept cell is part of a friendly movement
                # chain, retaining only strict-improvement steps makes every
                # rear Vanguard fall through to WAIT.  A bounded lateral step
                # opens the chain while keeping the unit on the interception
                # corridor and inside the home pursuit envelope.
                score = (
                    immediate,
                    future,
                    occupied.get(destination, 0),
                    candidate_distance,
                    route_block,
                    self.memory.congestion_counts.get(destination, 0),
                    index,
                )
                reposition.append((score, direction, destination))
        intents = [
            ActionIntent.move(
                vanguard.id,
                UnitMission.ATTACK,
                50,
                direction,
                destination,
                risk=score[2] * 100 + score[3] * 10,
                exclusive_destination=True,
                tie_break=score,
                reason="ROUTE_INTERCEPT_ADVANCE",
                metadata=(
                    ("target_id", str(target.id)),
                    ("candidate_distance", score[0]),
                    ("route_block", score[1]),
                    ("immediate_attackers", score[2]),
                    ("future_attackers", score[3]),
                    ("intercept_improved", True),
                ),
            )
            for score, direction, destination in sorted(options)[:4]
        ]
        intents.extend(
            ActionIntent.move(
                vanguard.id,
                UnitMission.ATTACK,
                53,
                direction,
                destination,
                risk=score[0] * 100 + score[1] * 10,
                exclusive_destination=True,
                tie_break=score,
                reason="INTERCEPT_REPOSITION",
                metadata=(
                    ("target_id", str(target.id)),
                    ("candidate_distance", score[3]),
                    ("route_block", score[4]),
                ),
            )
            for score, direction, destination in sorted(reposition)[:4]
        )
        if options or reposition:
            intents.append(
                ActionIntent.simple(
                    vanguard.id,
                    IntentAction.WAIT,
                    UnitMission.ATTACK,
                    54,
                    reason="INTERCEPT_MOVE_BLOCKED_THIS_TICK",
                    target_id=target.id,
                )
            )
        elif (
            manhattan(vanguard.position, target.position)
            <= self.config.vanguard_engage_distance
        ):
            intents.append(
                ActionIntent.simple(
                    vanguard.id,
                    IntentAction.WAIT,
                    UnitMission.ATTACK,
                    54,
                    reason=(
                        "HOLD_INTERCEPT_LINE"
                        if projection.immediate_attackers(vanguard.position) < vanguard.hp
                        else "NO_SURVIVABLE_ACTION"
                    ),
                    target_id=target.id,
                )
            )
        return intents

    def _sweep_suppressed(
        self,
        tick: int,
        target_id: UUID,
        cell: Position,
    ) -> bool:
        feedback = self.memory.vanguard_sweep_feedback.get((target_id, cell))
        return bool(
            feedback is not None
            and feedback.misses >= 1
            and tick <= feedback.suppressed_until + self.config.ranger_miss_suppress_ticks
        )
