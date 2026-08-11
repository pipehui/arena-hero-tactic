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
    EnemyActionEstimate,
    EnemyRangerFireEstimate,
    EntitySnapshot,
    FireMission,
    HomeCombatAssignment,
    IntentAction,
    ShotPlan,
    ScreeningGroupState,
    UnitMission,
    VanguardIntent,
    VanguardIntentEstimate,
    VanguardAssignmentCandidate,
    VanguardInterceptTask,
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
        self._sync_screening_groups(world, projection)
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

    def _sync_screening_groups(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> None:
        """Maintain sticky 2V+2R intercept groups outside the home battle line."""

        if world.core is None:
            self.memory.screening_groups.clear()
            return
        friend_by_id = {unit.id: unit for unit in world.friendlies}
        visible_by_id = {enemy.id: enemy for enemy in world.enemies}
        inner_threat = any(
            enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.position, world.core.position)
            <= self.config.home_engage_radius
            for enemy in world.enemies
        )
        for target_id, group in tuple(self.memory.screening_groups.items()):
            target = visible_by_id.get(target_id)
            members = tuple(
                friend_by_id.get(member_id)
                for member_id in (*group.vanguard_ids, *group.ranger_ids)
            )
            invalid_member = any(
                member is None
                or member.hp * 2 <= UNIT_MAX_HP[member.unit_type]
                for member in members
            )
            if invalid_member:
                self.memory.screening_groups.pop(target_id, None)
                continue
            if target is None:
                if world.tick - group.last_seen_tick > self.config.outer_screen_fog_ttl:
                    self.memory.screening_groups.pop(target_id, None)
                continue
            distance = manhattan(target.position, world.core.position)
            if distance <= self.config.home_engage_radius:
                self.memory.screening_groups[target_id] = replace(
                    group,
                    last_seen_tick=world.tick,
                    last_distance=distance,
                    outward_ticks=0,
                    phase="HOME_HANDOFF",
                )
                continue
            outward = group.outward_ticks + 1 if distance > group.last_distance else 0
            near_friendly = any(
                manhattan(target.position, friendly.position) <= 6
                for friendly in world.friendlies
            )
            minimum_hold = world.tick - group.started_tick < self.config.outer_screen_hold_ticks
            if (
                distance > self.config.outer_screen_continue_radius
                or (outward >= 2 and distance > self.config.home_pursuit_radius and not near_friendly)
            ) and not minimum_hold:
                self.memory.screening_groups.pop(target_id, None)
                continue
            self.memory.screening_groups[target_id] = ScreeningGroupState(
                target_id=target_id,
                vanguard_ids=group.vanguard_ids,
                ranger_ids=group.ranger_ids,
                started_tick=group.started_tick,
                last_seen_tick=world.tick,
                last_distance=distance,
                outward_ticks=outward,
                phase="PURSUING" if distance > self.config.outer_screen_max_radius else "INTERCEPTING",
            )

        available_vanguards = self._screening_candidates(
            world, UnitType.VANGUARD
        )
        available_rangers = self._screening_candidates(world, UnitType.RANGER)
        used = {
            member_id
            for group in self.memory.screening_groups.values()
            for member_id in (*group.vanguard_ids, *group.ranger_ids)
        }
        available_vanguards = [unit for unit in available_vanguards if unit.id not in used]
        available_rangers = [unit for unit in available_rangers if unit.id not in used]
        remaining_slots = min(
            max(
                0,
                (
                    len(available_vanguards)
                    - self.config.outer_screen_home_vanguard_reserve
                )
                // self.config.outer_screen_vanguards,
            ),
            max(
                0,
                (
                    len(available_rangers)
                    - self.config.outer_screen_home_ranger_reserve
                )
                // self.config.outer_screen_rangers,
            ),
        )
        if remaining_slots <= 0 or inner_threat:
            return
        targets = tuple(
            sorted(
                (
                    enemy
                    for enemy in world.enemies
                    if enemy.id not in self.memory.screening_groups
                    and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                    and self._screening_trigger(world, projection, enemy)
                ),
                key=lambda enemy: self.target_priority(world, projection, enemy),
            )
        )
        for target in targets[:remaining_slots]:
            chosen_vanguards = tuple(
                sorted(
                    available_vanguards,
                    key=lambda unit: (
                        manhattan(unit.position, target.position),
                        unit.id.bytes,
                    ),
                )[: self.config.outer_screen_vanguards]
            )
            chosen_rangers = tuple(
                sorted(
                    available_rangers,
                    key=lambda unit: (
                        manhattan(unit.position, target.position),
                        unit.id.bytes,
                    ),
                )[: self.config.outer_screen_rangers]
            )
            if (
                len(chosen_vanguards) < self.config.outer_screen_vanguards
                or len(chosen_rangers) < self.config.outer_screen_rangers
            ):
                break
            group = ScreeningGroupState(
                target_id=target.id,
                vanguard_ids=(chosen_vanguards[0].id, chosen_vanguards[1].id),
                ranger_ids=(chosen_rangers[0].id, chosen_rangers[1].id),
                started_tick=world.tick,
                last_seen_tick=world.tick,
                last_distance=manhattan(target.position, world.core.position),
            )
            self.memory.screening_groups[target.id] = group
            used.update((*group.vanguard_ids, *group.ranger_ids))
            available_vanguards = [
                unit for unit in available_vanguards if unit.id not in used
            ]
            available_rangers = [
                unit for unit in available_rangers if unit.id not in used
            ]

    def _screening_candidates(
        self,
        world: WorldModel,
        unit_type: UnitType,
    ) -> list[EntitySnapshot]:
        raid_ids = set(self.memory.raid_member_ids)
        return [
            unit
            for unit in world.friendlies
            if unit.unit_type is unit_type
            and unit.hp * 2 > UNIT_MAX_HP[unit.unit_type]
            and unit.id not in raid_ids
            and unit.id != self.memory.beacon_mission_actor_id
            and not (
                unit.id in self.memory.unit_missions
                and self.memory.unit_missions[unit.id].mission is UnitMission.RECOVER
            )
        ]

    def _screening_trigger(
        self,
        world: WorldModel,
        projection: TacticalMap,
        enemy: EntitySnapshot,
    ) -> bool:
        assert world.core is not None
        distance = manhattan(enemy.position, world.core.position)
        if not (
            self.config.outer_screen_min_radius < distance
            <= self.config.outer_screen_max_radius
        ):
            return False
        close_to_friendly = any(
            manhattan(enemy.position, unit.position)
            <= self.config.outer_screen_acquire_distance
            for unit in world.friendlies
        )
        track = world.track(enemy.id)
        moving_inward = bool(
            track is not None
            and len(track.samples) >= 3
            and manhattan(track.samples[-1][1], world.core.position)
            < manhattan(track.samples[-2][1], world.core.position)
            < manhattan(track.samples[-3][1], world.core.position)
        )
        projected = projection.enemy(enemy.id)
        next_step_attack = bool(
            projected is not None
            and (
                world.core.position in projected.future_attack_cells
                or any(
                    unit.position in projected.future_attack_cells
                    for unit in world.friendlies
                )
            )
        )
        return close_to_friendly or moving_inward or next_step_attack

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
            prediction = self.enemy_prediction(world, projection, enemy)
            cells = prediction.candidate_cells
            confidence = prediction.confidence
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
                    prediction_mode=(
                        prediction.intent.value
                        if isinstance(prediction, VanguardIntentEstimate)
                        else (
                            "RANGER_" + prediction.confidence
                            if isinstance(prediction, EnemyRangerFireEstimate)
                            else "GENERIC"
                        )
                    ),
                    candidate_roles=prediction.candidate_roles,
                    evidence=prediction.evidence,
                    split_fire=(
                        isinstance(prediction, VanguardIntentEstimate)
                        and prediction.intent is VanguardIntent.UNCERTAIN
                    )
                    or (
                        isinstance(prediction, EnemyRangerFireEstimate)
                        and "MOVING" in prediction.evidence
                        and prediction.firing_position is not None
                    ),
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
                if mission.split_fire:
                    self._assign_split_coverage(
                        world,
                        mission,
                        state.ranger_by_id,
                        state.available,
                        state.assignments,
                        state.mission_assignments,
                    )
                else:
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
                screening = self.memory.screening_groups.get(mission.target_id)
                if screening is not None and ranger_id not in screening.ranger_ids:
                    # Home-reserve Rangers may fire from their current cell,
                    # but only the assigned pair may leave the inner defense
                    # to improve an outer screening line.
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

    def _assign_split_coverage(
        self,
        world: WorldModel,
        mission: FireMission,
        rangers: dict[UUID, EntitySnapshot],
        available: set[UUID],
        assignments: dict[UUID, tuple[UUID, Position, str]],
        mission_assignments: dict[UUID, list[tuple[UUID, Position]]],
    ) -> None:
        """Cover distinct plausible outcomes before duplicating a shot."""

        for cell in mission.candidate_cells:
            shooters = sorted(
                (
                    ranger
                    for ranger_id, ranger in rangers.items()
                    if ranger_id in available
                    and ranger_line_is_clear(
                        ranger.position,
                        cell,
                        world.known_obstacles,
                    )
                    and not self._shot_suppressed(world.tick, mission, cell)
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
            if not shooters:
                continue
            ranger = shooters[0]
            assignments[ranger.id] = mission.target_id, cell, "INTENT_SPLIT_COVERAGE"
            mission_assignments[mission.target_id].append((ranger.id, cell))
            available.remove(ranger.id)
            if len(mission_assignments[mission.target_id]) >= 2:
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
        prediction = self.enemy_prediction(world, projection, enemy)
        return prediction.candidate_cells, prediction.confidence

    def enemy_prediction(
        self,
        world: WorldModel,
        projection: TacticalMap,
        enemy: EntitySnapshot,
    ) -> VanguardIntentEstimate | EnemyRangerFireEstimate | EnemyActionEstimate:
        projected = projection.enemy(enemy.id)
        legal = list(projected.possible_positions if projected is not None else (enemy.position,))
        if enemy.position not in legal:
            legal.insert(0, enemy.position)
        if enemy.unit_type is UnitType.VANGUARD:
            return self._vanguard_intent_estimate(world, enemy, legal)
        if enemy.unit_type is UnitType.RANGER:
            return self._enemy_ranger_fire_estimate(world, enemy, legal)
        ranked, confidence = self._trajectory_candidates(world, enemy, legal)
        current_attacks = self._enemy_attacks_any_friendly(world, enemy, enemy.position)
        if current_attacks:
            ranked = [enemy.position, *ranked]
        ordered = self._dedupe_legal(ranked, legal)
        return EnemyActionEstimate(
            target_id=enemy.id,
            confidence=confidence,
            candidate_cells=tuple(ordered[:5]),
            candidate_roles=tuple(
                "CURRENT" if cell == enemy.position else "MOTION"
                for cell in ordered[:5]
            ),
            evidence=("CURRENT_ATTACK" if current_attacks else "TRAJECTORY_ONLY",),
        )

    def _trajectory_candidates(
        self,
        world: WorldModel,
        enemy: EntitySnapshot,
        legal: list[Position],
    ) -> tuple[list[Position], str]:
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
        return self._dedupe_legal(ranked, legal), confidence

    def _vanguard_intent_estimate(
        self,
        world: WorldModel,
        enemy: EntitySnapshot,
        legal: list[Position],
    ) -> VanguardIntentEstimate:
        track = world.track(enemy.id)
        current_attacks = self._enemy_attacks_any_friendly(
            world, enemy, enemy.position
        )
        trajectory, _ = self._trajectory_candidates(world, enemy, legal)
        if current_attacks:
            ranked = [
                enemy.position,
                *sorted(
                    (cell for cell in legal if cell != enemy.position),
                    key=lambda cell: (
                        -self._enemy_action_value(world, enemy, cell),
                        self._enemy_protected_distance(world, cell),
                        cell,
                    ),
                ),
            ]
            ordered = self._dedupe_legal(ranked, legal)
            return VanguardIntentEstimate(
                enemy.id,
                VanguardIntent.ATTACKING,
                "HIGH",
                tuple(ordered[:5]),
                tuple(
                    "CURRENT_ATTACK" if cell == enemy.position else "ATTACK_EXIT"
                    for cell in ordered[:5]
                ),
                ("CURRENT_ATTACK_AVAILABLE",),
            )

        if track is not None and len(track.samples) >= 3:
            a, b, c = (sample[1] for sample in track.samples[-3:])
            distances = tuple(
                self._enemy_protected_distance(world, cell) for cell in (a, b, c)
            )
            delta = c[0] - b[0], c[1] - b[1]
            continuation = c[0] + delta[0], c[1] + delta[1]
            if (
                a != b
                and b != c
                and distances[0] <= distances[1] <= distances[2]
                and continuation in legal
            ):
                ordered = self._dedupe_legal(
                    [continuation, c, *trajectory, *legal], legal
                )
                return VanguardIntentEstimate(
                    enemy.id,
                    VanguardIntent.RETREATING,
                    "HIGH",
                    tuple(ordered[:5]),
                    tuple(
                        "RETREAT_CONTINUATION" if cell == continuation else "RETREAT_ALTERNATE"
                        for cell in ordered[:5]
                    ),
                    ("TWO_TICK_OUTWARD_MOTION",),
                )

        moved = bool(
            track is not None
            and len(track.samples) >= 2
            and track.samples[-2][1] != track.samples[-1][1]
        )
        previous = track.samples[-2][1] if moved and track is not None else enemy.position
        rangers = tuple(
            unit
            for unit in world.friendlies
            if unit.unit_type is UnitType.RANGER
            and unit.hp * 2 > UNIT_MAX_HP[UnitType.RANGER]
        )
        entered_blind_spot = moved and any(
            ranger_line_is_clear(ranger.position, previous, world.known_obstacles)
            and not ranger_line_is_clear(
                ranger.position, enemy.position, world.known_obstacles
            )
            for ranger in rangers
        )
        closer = self._enemy_protected_distance(
            world, enemy.position
        ) <= self._enemy_protected_distance(world, previous)
        next_attack = any(
            self._enemy_action_value(world, enemy, cell) > 0 for cell in legal
        )
        if entered_blind_spot and (closer or next_attack):
            motion = trajectory[0] if trajectory else enemy.position
            ranked = sorted(
                legal,
                key=lambda cell: (
                    -self._enemy_action_value(world, enemy, cell),
                    sum(
                        ranger_line_is_clear(
                            ranger.position, cell, world.known_obstacles
                        )
                        for ranger in rangers
                    ),
                    self._enemy_protected_distance(world, cell),
                    int(cell != motion),
                    cell,
                ),
            )
            ordered = self._dedupe_legal([*ranked, *trajectory], legal)
            return VanguardIntentEstimate(
                enemy.id,
                VanguardIntent.BLIND_SPOT_APPROACH,
                "MEDIUM",
                tuple(ordered[:5]),
                tuple(
                    "BLIND_APPROACH" if cell != enemy.position else "CURRENT_BLIND_CELL"
                    for cell in ordered[:5]
                ),
                ("LEFT_RANGER_FIRE_LINE", "PROTECTED_TARGET_CLOSING"),
            )

        motion = next((cell for cell in trajectory if cell != enemy.position), None)
        approach = min(
            (cell for cell in legal if cell != enemy.position),
            key=lambda cell: (
                self._enemy_protected_distance(world, cell),
                -self._enemy_action_value(world, enemy, cell),
                cell,
            ),
            default=None,
        )
        lateral = sorted(
            (cell for cell in legal if cell not in {enemy.position, motion, approach}),
            key=lambda cell: (
                self._enemy_protected_distance(world, cell),
                cell,
            ),
        )
        ordered = self._dedupe_legal(
            [cell for cell in (approach, motion, *lateral, enemy.position) if cell is not None],
            legal,
        )
        return VanguardIntentEstimate(
            enemy.id,
            VanguardIntent.UNCERTAIN,
            "LOW",
            tuple(ordered[:5]),
            tuple(
                "APPROACH_ANGLE"
                if cell == approach
                else "MOTION_ANGLE"
                if cell == motion
                else "WAIT"
                if cell == enemy.position
                else "LATERAL_ANGLE"
                for cell in ordered[:5]
            ),
            ("INTENT_NOT_RESOLVED",),
        )

    def _enemy_ranger_fire_estimate(
        self,
        world: WorldModel,
        enemy: EntitySnapshot,
        legal: list[Position],
    ) -> EnemyRangerFireEstimate:
        track = world.track(enemy.id)
        stationary = bool(
            track is not None
            and len(track.samples) >= 3
            and len({sample[1] for sample in track.samples[-3:]}) == 1
        )
        moving = bool(
            track is not None
            and len(track.samples) >= 2
            and track.samples[-2][1] != track.samples[-1][1]
        )
        current_attacks = self._enemy_attacks_any_friendly(
            world, enemy, enemy.position
        )
        firing_position = min(
            (cell for cell in legal if cell != enemy.position),
            key=lambda cell: (
                -self._enemy_action_value(world, enemy, cell),
                self._enemy_protected_distance(world, cell),
                cell,
            ),
            default=None,
        )
        if stationary:
            candidates = [enemy.position, *(cell for cell in legal if cell != enemy.position)]
            confidence = "HIGH"
            evidence = ("STATIONARY",)
        elif moving and firing_position is not None:
            candidates = (
                [enemy.position, firing_position]
                if current_attacks
                else [firing_position, enemy.position]
            )
            candidates.extend(cell for cell in legal if cell not in candidates)
            confidence = "MEDIUM"
            evidence = ("MOVING", "CURRENT_AND_FIRING_POSITION")
        else:
            candidates = [enemy.position, *(cell for cell in legal if cell != enemy.position)]
            confidence = "MEDIUM" if current_attacks else "LOW"
            evidence = ("CURRENT_ATTACK_AVAILABLE",) if current_attacks else ("LIMITED_TRACK",)
        ordered = self._dedupe_legal(candidates, legal)
        return EnemyRangerFireEstimate(
            target_id=enemy.id,
            confidence=confidence,
            current_cell=enemy.position,
            firing_position=firing_position,
            candidate_cells=tuple(ordered[:5]),
            candidate_roles=tuple(
                "CURRENT"
                if cell == enemy.position
                else "NEXT_FIRE_POSITION"
                if cell == firing_position
                else "ALTERNATE"
                for cell in ordered[:5]
            ),
            evidence=evidence,
        )

    @staticmethod
    def _dedupe_legal(
        ranked: list[Position],
        legal: list[Position],
    ) -> list[Position]:
        ordered: list[Position] = []
        for cell in (*ranked, *legal):
            if cell in legal and cell not in ordered:
                ordered.append(cell)
        return ordered

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
                enemy.id in self.memory.screening_groups
                and manhattan(enemy.position, world.core.position)
                <= self.config.outer_screen_continue_radius
            )
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

    def home_combat_assignment(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> HomeCombatAssignment:
        """Assign home Vanguards globally before formation planners run.

        The old per-unit loop let low UUID defenders consume every responder
        slot even when they were much farther from the contact.  This pass
        first covers distinct threat sectors, then adds one extra responder
        per target, using an actual bounded path field as the primary cost.
        """

        if world.core is None:
            return HomeCombatAssignment()
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
        vanguards = tuple(
            unit
            for unit in world.friendlies
            if unit.unit_type is UnitType.VANGUARD
            and unit.hp * 2 > UNIT_MAX_HP[UnitType.VANGUARD]
            and unit.id not in self.memory.raid_member_ids
            and unit.id != self.memory.beacon_mission_actor_id
        )
        if not urgent or not vanguards:
            return HomeCombatAssignment(
                unassigned_vanguards=tuple(unit.id for unit in vanguards),
                uncovered_targets=tuple(enemy.id for enemy in urgent),
            )

        fields: dict[UUID, dict[Position, int]] = {}
        blocked = projection.hostile_occupied | projection.service_positions
        for vanguard in vanguards:
            fields[vanguard.id] = weighted_distance_field(
                world,
                vanguard.position,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(blocked - {vanguard.position}),
            )[0]

        predictions = {
            enemy.id: self.enemy_candidate_cells(world, projection, enemy)[0]
            for enemy in urgent
        }
        profiles: dict[
            tuple[UUID, UUID],
            tuple[Position, tuple[int, ...]],
        ] = {}
        for enemy in urgent:
            screening = self.memory.screening_groups.get(enemy.id)
            for vanguard in vanguards:
                eligible = (
                    manhattan(enemy.position, world.core.position)
                    <= self.config.home_engage_radius
                    or manhattan(vanguard.position, enemy.position)
                    <= self.config.vanguard_engage_distance
                    or (
                        screening is not None
                        and vanguard.id in screening.vanguard_ids
                    )
                )
                if eligible:
                    profiles[(vanguard.id, enemy.id)] = self._vanguard_assignment_cost(
                        world,
                        projection,
                        vanguard,
                        enemy,
                        predictions[enemy.id],
                        fields[vanguard.id],
                    )
        sectors: dict[Direction, list[EntitySnapshot]] = defaultdict(list)
        for enemy in urgent:
            sectors[self._home_sector(world.core.position, enemy.position)].append(enemy)
        for enemies in sectors.values():
            enemies.sort(key=lambda enemy: self.target_priority(world, projection, enemy))

        assigned: set[UUID] = set()
        response_counts: defaultdict[UUID, int] = defaultdict(int)
        tasks: list[VanguardInterceptTask] = []

        def assign(target: EntitySnapshot, *, phase: str | None = None) -> bool:
            screening = self.memory.screening_groups.get(target.id)
            available = [
                unit
                for unit in vanguards
                if unit.id not in assigned
                and (unit.id, target.id) in profiles
            ]
            if not available:
                return False
            if screening is not None and screening.phase == "HOME_HANDOFF":
                preserved = [
                    unit for unit in available if unit.id in screening.vanguard_ids
                ]
                if preserved:
                    available = preserved
                    phase = "HOME_HANDOFF"
            candidates = predictions[target.id]
            rows = []
            for vanguard in available:
                intercept, cost = profiles[(vanguard.id, target.id)]
                rows.append((cost, vanguard.id.bytes, vanguard, intercept))
            cost, _, vanguard, intercept = min(rows, key=lambda row: row[:2])
            assigned.add(vanguard.id)
            response_counts[target.id] += 1
            tasks.append(
                VanguardInterceptTask(
                    vanguard_id=vanguard.id,
                    target_id=target.id,
                    sector=self._home_sector(world.core.position, target.position),
                    phase=phase or "HOME_INTERCEPT",
                    intercept_cell=intercept,
                    candidate_cells=candidates,
                    cost=cost,
                )
            )
            return True

        # One defender per active direction before any direction receives a
        # second responder.
        sector_heads = sorted(
            (enemies[0] for enemies in sectors.values()),
            key=lambda enemy: self.target_priority(world, projection, enemy),
        )
        for enemy in sector_heads:
            assign(enemy)
        # Keep an outer group's two closest blockers through the 14 -> 13
        # handoff, then give every ordinary target at most one extra blocker.
        for enemy in urgent:
            limit = 2
            while response_counts[enemy.id] < limit and assign(enemy):
                pass

        covered = {task.target_id for task in tasks}
        selected_pairs = {(task.vanguard_id, task.target_id) for task in tasks}
        candidate_rows = tuple(
            VanguardAssignmentCandidate(
                vanguard_id=vanguard.id,
                target_id=enemy.id,
                cost=(
                    None
                    if (vanguard.id, enemy.id) not in profiles
                    else profiles[(vanguard.id, enemy.id)][1]
                ),
                selected=(vanguard.id, enemy.id) in selected_pairs,
                reason=(
                    "SELECTED"
                    if (vanguard.id, enemy.id) in selected_pairs
                    else "OUT_OF_RESPONSE_RANGE"
                    if (vanguard.id, enemy.id) not in profiles
                    else "ASSIGNED_TO_HIGHER_PRIORITY_THREAT"
                    if vanguard.id in assigned
                    else "TARGET_RESPONSE_LIMIT_FILLED"
                    if response_counts[enemy.id] >= 2
                    else "HIGHER_COST"
                ),
            )
            for enemy in urgent
            for vanguard in vanguards
        )
        return HomeCombatAssignment(
            tasks=tuple(tasks),
            candidates=candidate_rows,
            unassigned_vanguards=tuple(
                unit.id for unit in vanguards if unit.id not in assigned
            ),
            uncovered_targets=tuple(
                enemy.id for enemy in urgent if enemy.id not in covered
            ),
        )

    def _vanguard_assignment_cost(
        self,
        world: WorldModel,
        projection: TacticalMap,
        vanguard: EntitySnapshot,
        target: EntitySnapshot,
        candidates: tuple[Position, ...],
        distances: dict[Position, int],
    ) -> tuple[Position, tuple[int, ...]]:
        assert world.core is not None
        intercepts = {
            neighbor
            for candidate in candidates
            for _, neighbor in cardinal_neighbors(candidate)
            if neighbor in world.known_passable
            and neighbor not in world.known_obstacles
            and neighbor not in projection.hostile_occupied
            and neighbor not in projection.service_positions
            and manhattan(neighbor, world.core.position)
            <= self.config.home_pursuit_radius
        }
        if any(manhattan(vanguard.position, cell) == 1 for cell in candidates):
            intercepts.add(vanguard.position)
        if not intercepts:
            intercepts.add(vanguard.position)
        screening = self.memory.screening_groups.get(target.id)
        rows = []
        for cell in intercepts:
            coverage = sum(
                len(candidates) - rank
                for rank, candidate in enumerate(candidates)
                if manhattan(cell, candidate) == 1
            )
            path_cost = distances.get(cell, 1 << 20)
            rows.append(
                (
                    (
                        int(cell != vanguard.position or not coverage),
                        path_cost,
                        -coverage,
                        int(screening is None or vanguard.id not in screening.vanguard_ids),
                        projection.immediate_attackers(cell),
                        projection.future_attackers(cell),
                        manhattan(cell, world.core.position),
                        cell[0],
                        cell[1],
                    ),
                    cell,
                )
            )
        cost, intercept = min(rows, key=lambda row: row[0])
        return intercept, cost

    @staticmethod
    def _home_sector(core: Position, threat: Position) -> Direction:
        dx, dy = threat[0] - core[0], threat[1] - core[1]
        if abs(dx) > abs(dy):
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        return Direction.DOWN if dy > 0 else Direction.UP

    def vanguard_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        assignment: HomeCombatAssignment | None = None,
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
        protected_units = tuple(
            unit
            for unit in world.friendlies
            if unit.unit_type in {UnitType.WORKER, UnitType.RANGER}
        )
        assignment = assignment or self.home_combat_assignment(world, projection)
        tasks_by_vanguard = {task.vanguard_id: task for task in assignment.tasks}
        for vanguard in (
            unit for unit in world.friendlies if unit.unit_type is UnitType.VANGUARD
        ):
            adjacent = self._adjacent_sweep(vanguard, urgent)
            if adjacent is not None:
                intents.append(adjacent)
                continue
            task = tasks_by_vanguard.get(vanguard.id)
            if not urgent or task is None:
                continue
            target = world.enemy(task.target_id)
            if target is None:
                continue
            candidate_cells = task.candidate_cells
            confidence = self.enemy_candidate_cells(world, projection, target)[1]
            joint = self._joint_vanguard_sweep(
                world,
                projection,
                vanguard,
                urgent,
            )
            if joint is not None:
                sweep_intent, sweep_plan = joint
                intents.append(sweep_intent)
                predicted_sweeps[vanguard.id] = sweep_plan
                continue
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
                    task.intercept_cell,
                )
            )
        self.memory.last_vanguard_sweeps = predicted_sweeps
        return intents

    def _joint_vanguard_sweep(
        self,
        world: WorldModel,
        projection: TacticalMap,
        vanguard: EntitySnapshot,
        urgent: tuple[EntitySnapshot, ...],
    ) -> tuple[ActionIntent, ShotPlan] | None:
        rows: list[tuple[tuple[int, ...], Position, tuple[UUID, ...]]] = []
        for direction_index, (_, cell) in enumerate(cardinal_neighbors(vanguard.position)):
            converging: list[UUID] = []
            rank_sum = 0
            for enemy in urgent:
                candidates, _ = self.enemy_candidate_cells(world, projection, enemy)
                if cell not in candidates:
                    continue
                rank = candidates.index(cell)
                if rank <= 1:
                    converging.append(enemy.id)
                    rank_sum += rank
            if len(converging) < 2:
                continue
            rows.append(
                (
                    (-len(converging), rank_sum, direction_index, cell[0], cell[1]),
                    cell,
                    tuple(sorted(converging, key=lambda item: item.bytes)),
                )
            )
        if not rows:
            return None
        score, cell, targets = min(rows, key=lambda row: row[0])
        # Fresh multi-enemy convergence is stronger evidence than an old miss
        # on the same cell, so it deliberately overrides short suppression.
        direction = direction_between(vanguard.position, cell)
        if direction is None:
            return None
        representative = targets[0]
        return (
            ActionIntent(
                actor_id=vanguard.id,
                action=IntentAction.SWEEP,
                mission=UnitMission.ATTACK,
                priority=29,
                direction=direction,
                target_id=representative,
                target_position=cell,
                reason="MULTI_ENEMY_CONVERGENCE_SWEEP",
                metadata=(
                    ("converging_targets", tuple(str(item) for item in targets)),
                    ("joint_score", score),
                ),
            ),
            ShotPlan(vanguard.id, representative, cell),
        )

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
            vanguard.id,
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
        assigned_intercept,
    ):
        assert world.core is not None
        protected = min(
            (world.core.position, *(unit.position for unit in protected_units)),
            key=lambda cell: manhattan(target.position, cell),
        )
        options: list[tuple[tuple[int, ...], Direction, Position]] = []
        occupied = projection.occupied_cells
        current_route_block = min(
            manhattan(cell, vanguard.position)
            + manhattan(vanguard.position, protected)
            for cell in candidate_cells
        )
        path_blocks = frozenset(
            (projection.hostile_occupied | projection.service_positions)
            - {vanguard.position, assigned_intercept}
        )
        current_path = route_to(
            world,
            vanguard.position,
            assigned_intercept,
            node_limit=self.config.path_node_limit,
            blocked=path_blocks,
        )
        current_path_cost = (
            0
            if vanguard.position == assigned_intercept
            else (1 << 20 if current_path is None else current_path.distance)
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
            next_path = route_to(
                world,
                destination,
                assigned_intercept,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(path_blocks - {destination}),
            )
            next_path_cost = (
                0
                if destination == assigned_intercept
                else (1 << 20 if next_path is None else next_path.distance)
            )
            current_coverage = sum(
                len(candidate_cells) - rank
                for rank, cell in enumerate(candidate_cells)
                if manhattan(vanguard.position, cell) == 1
            )
            candidate_coverage = sum(
                len(candidate_cells) - rank
                for rank, cell in enumerate(candidate_cells)
                if manhattan(destination, cell) == 1
            )
            improves_intercept = (
                next_path_cost < current_path_cost
                or route_block < current_route_block
                or candidate_coverage > current_coverage
            )
            if improves_intercept:
                score = (
                    next_path_cost,
                    -candidate_coverage,
                    route_block,
                    candidate_distance,
                    immediate,
                    future,
                    occupied.get(destination, 0),
                    self.memory.congestion_counts.get(destination, 0),
                    index,
                )
                options.append((score, direction, destination))
        intents = [
            ActionIntent.move(
                vanguard.id,
                UnitMission.ATTACK,
                50,
                direction,
                destination,
                risk=score[4] * 100 + score[5] * 10,
                exclusive_destination=True,
                tie_break=score,
                reason="ROUTE_INTERCEPT_ADVANCE",
                metadata=(
                    ("target_id", str(target.id)),
                    ("candidate_distance", score[3]),
                    ("route_block", score[2]),
                    ("immediate_attackers", score[4]),
                    ("future_attackers", score[5]),
                    ("intercept_improved", True),
                    ("intercept_path_before", current_path_cost),
                    ("intercept_path_after", score[0]),
                    ("candidate_coverage_before", current_coverage),
                    ("candidate_coverage_after", -score[1]),
                ),
            )
            for score, direction, destination in sorted(options)[:4]
        ]
        if options:
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
