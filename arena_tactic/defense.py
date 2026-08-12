from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from uuid import UUID

from arena_hero import Direction, Position, UnitType

from .combat import CombatPlanner
from .config import TacticConfig
from .geometry import (
    DIRECTION_ORDER,
    cardinal_neighbors,
    count_open_neighbors,
    manhattan,
    manhattan_ring,
    ranger_firing_positions,
    ranger_line_is_clear,
)
from .models import (
    ActionIntent,
    EntitySnapshot,
    IntentAction,
    SquadRendezvousLease,
    SquadState,
    UnitMission,
    WorldModel,
)
from .planning import move_viability, path_to, route_to
from .projection import TacticalMap
from .rules import UNIT_MAX_HP
from .state import TacticMemory


class DefensePlanner:
    """Terrain-aware home defense and persistent peaceful squad patrol."""

    def __init__(
        self,
        config: TacticConfig,
        memory: TacticMemory,
        combat: CombatPlanner,
    ) -> None:
        self.config = config
        self.memory = memory
        self.combat = combat

    def intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        protected: frozenset[Position],
        assigned_vanguard_ids: frozenset[UUID] = frozenset(),
    ) -> list[ActionIntent]:
        if world.core is None:
            return []
        living_combatants = tuple(
            unit
            for unit in world.friendlies
            if unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
        )
        healthy = tuple(
            unit
            for unit in living_combatants
            if unit.hp * 2 > UNIT_MAX_HP[unit.unit_type]
        )
        visible_by_id = {enemy.id: enemy for enemy in world.enemies}
        current_threat_intel = tuple(
            enemy
            for enemy in projection.enemies
            if enemy.visible_now
            and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.observed_position, world.core.position)
            <= self.config.home_warning_radius
        )
        threats = tuple(
            visible_by_id[enemy.enemy_id]
            for enemy in current_threat_intel
            if enemy.enemy_id in visible_by_id
        )
        if threats:
            self.memory.home_defense_alert_until = max(
                self.memory.home_defense_alert_until,
                world.tick + self.config.home_defense_hold_ticks,
            )
        elif world.tick <= self.memory.home_defense_alert_until:
            # A one-Tick vision gap must not dissolve a gathered defense.
            # Recent tracks are used only as formation anchors; combat still
            # requires a currently visible target in CombatPlanner.
            threats = tuple(
                EntitySnapshot(
                    id=enemy.enemy_id,
                    position=enemy.observed_position,
                    hp=1,
                    unit_type=enemy.unit_type,
                    controlled=False,
                )
                for enemy in projection.enemies
                if not enemy.visible_now
                and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
                and enemy.age <= self.config.home_defense_hold_ticks
                and manhattan(enemy.observed_position, world.core.position)
                <= self.config.home_warning_radius
            )
        self._sync_squads(living_combatants)
        screening_intents = self._screening_intents(
            world,
            projection,
            healthy,
            protected,
        )
        if threats:
            screening_ids = {
                member_id
                for group in self.memory.screening_groups.values()
                if group.phase != "HOME_HANDOFF"
                for member_id in (*group.vanguard_ids, *group.ranger_ids)
            }
            home_pool = tuple(
                unit
                for unit in healthy
                if unit.id not in screening_ids
                and unit.id not in assigned_vanguard_ids
            )
            return [
                *screening_intents,
                *self._sector_defense(
                    world,
                    projection,
                    home_pool,
                    threats,
                    protected,
                ),
            ]
        if screening_intents:
            screening_ids = {
                member_id
                for group in self.memory.screening_groups.values()
                if group.phase != "HOME_HANDOFF"
                for member_id in (*group.vanguard_ids, *group.ranger_ids)
            }
            home_pool = tuple(
                unit
                for unit in healthy
                if unit.id not in screening_ids
                and unit.id not in assigned_vanguard_ids
            )
            return [
                *screening_intents,
                *self._peaceful_patrol(world, projection, home_pool, protected),
            ]
        return self._peaceful_patrol(
            world,
            projection,
            tuple(unit for unit in healthy if unit.id not in assigned_vanguard_ids),
            protected,
        )

    def _screening_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        healthy: tuple[EntitySnapshot, ...],
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        """Give each active outer screen two blockers and two firing roles."""

        if world.core is None or not self.memory.screening_groups:
            return []
        members = {unit.id: unit for unit in healthy}
        intents: list[ActionIntent] = []
        reserved: set[Position] = set(protected)
        for group in sorted(
            self.memory.screening_groups.values(),
            key=lambda item: (item.started_tick, item.target_id.bytes),
        ):
            if group.phase == "HOME_HANDOFF":
                continue
            target = world.enemy(group.target_id)
            if target is None:
                track = world.track(group.target_id)
                target_position = None if track is None else track.position
                candidates = () if target_position is None else (target_position,)
            else:
                target_position = target.position
                candidates, _ = self.combat.enemy_candidate_cells(
                    world, projection, target
                )
            if target_position is None or not candidates:
                continue
            vanguards = [members.get(unit_id) for unit_id in group.vanguard_ids]
            rangers = [members.get(unit_id) for unit_id in group.ranger_ids]
            if any(unit is None for unit in (*vanguards, *rangers)):
                continue

            blocker_candidates = tuple(
                sorted(
                    {
                        cell
                        for candidate in candidates[:3]
                        for _, cell in cardinal_neighbors(candidate)
                        if cell in world.known_passable
                        and cell not in world.known_obstacles
                        and cell not in projection.hostile_occupied
                        and cell not in reserved
                        and manhattan(cell, world.core.position)
                        <= self.config.outer_screen_continue_radius
                    },
                    key=lambda cell: (
                        projection.immediate_attackers(cell),
                        projection.future_attackers(cell),
                        manhattan(cell, world.core.position),
                        min(manhattan(cell, candidate) for candidate in candidates),
                        cell,
                    ),
                )
            )
            for index, vanguard in enumerate(vanguards):
                assert vanguard is not None
                target_cell = next(
                    (cell for cell in blocker_candidates if cell not in reserved),
                    target_position,
                )
                reserved.add(target_cell)
                intents.extend(
                    self._move_or_wait(
                        world,
                        projection,
                        vanguard,
                        target_cell,
                        frozenset(reserved - {target_cell}),
                        "OUTER_SCREEN_BLOCKER" if index == 0 else "OUTER_SCREEN_FLANKER",
                        mission=UnitMission.HOME_DEFENSE,
                        move_priority=47,
                        wait_priority=49,
                    )
                )

            # Ranger attack, contact keeping and firing-line movement are all
            # owned by CombatPlanner.  Emitting another positioning intent
            # here used to let a generic zero-risk step override the dynamic
            # firing line and even walk the last observer out of vision.
        return intents

    def _sector_defense(
        self,
        world: WorldModel,
        projection: TacticalMap,
        combatants: tuple[EntitySnapshot, ...],
        threats: tuple[EntitySnapshot, ...],
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        assert world.core is not None
        fronts = self._sector_fronts(
            world,
            projection,
            threats,
            protected,
        )
        tasks = self._assign_sector_tasks(
            world,
            projection,
            combatants,
            fronts,
            protected,
        )
        intents: list[ActionIntent] = []
        for unit in combatants:
            task = tasks.get(unit.id)
            if task is None:
                intents.append(
                    ActionIntent.simple(
                        unit.id,
                        IntentAction.WAIT,
                        UnitMission.HOME_DEFENSE,
                        58,
                        reason="DEFENSE_POOL_RESERVE",
                    )
                )
                continue
            target, role = task
            if unit.position == target:
                intents.append(
                    ActionIntent.simple(
                        unit.id,
                        IntentAction.WAIT,
                        UnitMission.HOME_DEFENSE,
                        58,
                        target_position=target,
                        reason=f"{role}_HOLD",
                    )
                )
                continue
            intents.extend(
                self._move_or_wait(
                    world,
                    projection,
                    unit,
                    target,
                    protected,
                    role,
                    mission=UnitMission.HOME_DEFENSE,
                    move_priority=55,
                    wait_priority=59,
                )
            )
        return intents

    def _sector_fronts(
        self,
        world: WorldModel,
        projection: TacticalMap,
        threats: tuple[EntitySnapshot, ...],
        protected: frozenset[Position],
    ) -> tuple[tuple[Direction, Position, tuple[EntitySnapshot, ...]], ...]:
        assert world.core is not None
        sectors: dict[Direction, list[EntitySnapshot]] = defaultdict(list)
        for threat in threats:
            sectors[self._sector_direction(world.core.position, threat.position)].append(threat)
        active = tuple(
            sorted(
                sectors,
                key=lambda direction: DIRECTION_ORDER.index(direction),
            )
        )
        active_keys = {direction.value for direction in active}
        for key in tuple(self.memory.defense_sector_anchors):
            if key not in active_keys:
                self.memory.defense_sector_anchors.pop(key, None)
        front_positions: list[tuple[Direction, Position, tuple[EntitySnapshot, ...]]] = []
        for direction in active:
            sector_threats = tuple(
                sorted(
                    sectors[direction],
                    key=lambda enemy: self.combat.target_priority(world, projection, enemy),
                )
            )
            primary = sector_threats[0]
            path = path_to(
                world,
                primary.position,
                world.core.position,
                node_limit=self.config.path_node_limit,
                blocked=projection.hostile_occupied - {primary.position},
            )
            if path is not None:
                viable = [
                    cell
                    for cell in path[1:-1]
                    if self.config.peaceful_squad_radii[0]
                    <= manhattan(cell, world.core.position)
                    <= self.config.home_engage_radius
                    and cell not in protected
                    and cell not in projection.hostile_occupied
                ]
                front = viable[-1] if viable else None
            else:
                front = None
            if front is None:
                desired = self._advance(
                    world.core.position,
                    direction,
                    self.config.peaceful_squad_radii[0],
                )
                front = min(
                    (
                        cell
                        for cell in world.known_passable
                        if cell not in world.known_obstacles
                        and cell not in protected
                        and cell not in projection.hostile_occupied
                        and manhattan(cell, world.core.position)
                        <= self.config.home_engage_radius
                    ),
                    key=lambda cell: (manhattan(cell, desired), cell),
                    default=world.core.position,
                )
            anchor_key = direction.value
            previous_anchor = self.memory.defense_sector_anchors.get(anchor_key)
            if previous_anchor is not None:
                old_cell, assigned_tick = previous_anchor
                old_valid = (
                    old_cell in world.known_passable
                    and old_cell not in world.known_obstacles
                    and old_cell not in protected
                    and old_cell not in projection.hostile_occupied
                    and manhattan(old_cell, world.core.position)
                    <= self.config.home_engage_radius
                )
                if old_valid and (
                    world.tick - assigned_tick < 4
                    or manhattan(old_cell, front) <= 1
                ):
                    front = old_cell
            if previous_anchor is None or previous_anchor[0] != front:
                self.memory.defense_sector_anchors[anchor_key] = (
                    front,
                    world.tick,
                )
            front_positions.append((direction, front, sector_threats))
        return tuple(front_positions)

    def _assign_sector_tasks(
        self,
        world: WorldModel,
        projection: TacticalMap,
        combatants: tuple[EntitySnapshot, ...],
        front_positions: tuple[
            tuple[Direction, Position, tuple[EntitySnapshot, ...]],
            ...,
        ],
        protected: frozenset[Position],
    ) -> dict[UUID, tuple[Position, str]]:
        assert world.core is not None
        vanguards = sorted(
            (unit for unit in combatants if unit.unit_type is UnitType.VANGUARD),
            key=lambda unit: unit.id.bytes,
        )
        rangers = sorted(
            (unit for unit in combatants if unit.unit_type is UnitType.RANGER),
            key=lambda unit: unit.id.bytes,
        )
        tasks: dict[UUID, tuple[Position, str]] = {}
        reserved: set[Position] = set()
        for index, (_, front, sector_threats) in enumerate(front_positions):
            if vanguards:
                vanguard = min(
                    vanguards,
                    key=lambda unit: (manhattan(unit.position, front), unit.id.bytes),
                )
                tasks[vanguard.id] = front, "SECTOR_FRONTLINE"
                vanguards.remove(vanguard)
                reserved.add(front)
            firing_cells = self._firing_band(
                world,
                projection,
                front,
                protected | frozenset(reserved),
            )
            desired_rangers = max(1, len(sector_threats))
            for cell in firing_cells[:desired_rangers]:
                if not rangers:
                    break
                ranger = min(
                    rangers,
                    key=lambda unit: (manhattan(unit.position, cell), unit.id.bytes),
                )
                tasks[ranger.id] = cell, "SECTOR_FIRE_LINE"
                rangers.remove(ranger)
                reserved.add(cell)

        reserve_targets = [
            cell
            for cell in manhattan_ring(world.core.position, self.config.peaceful_squad_radii[0])
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in protected
            and cell not in reserved
        ]
        for unit in (*vanguards, *rangers):
            if reserve_targets:
                target = min(
                    reserve_targets,
                    key=lambda cell: (
                        projection.future_attackers(cell),
                        manhattan(unit.position, cell),
                        cell,
                    ),
                )
                tasks[unit.id] = target, "SECTOR_RESERVE"
                reserve_targets.remove(target)
        return tasks

    def _peaceful_patrol(
        self,
        world: WorldModel,
        projection: TacticalMap,
        healthy: tuple[EntitySnapshot, ...],
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        assert world.core is not None
        members, by_radius, paired_ids = self._active_patrol_squads(healthy)
        intents = self._paired_patrol_intents(
            world,
            projection,
            members,
            by_radius,
            protected,
        )
        intents.extend(
            self._reserve_patrol_intents(
                world,
                projection,
                healthy,
                paired_ids,
                protected,
            )
        )
        return intents

    def _active_patrol_squads(
        self,
        healthy: tuple[EntitySnapshot, ...],
    ) -> tuple[
        dict[UUID, EntitySnapshot],
        dict[int, list[SquadState]],
        set[UUID],
    ]:
        members = {unit.id: unit for unit in healthy}
        paired_ids: set[UUID] = set()
        by_radius: dict[int, list[SquadState]] = defaultdict(list)
        for squad in sorted(
            self.memory.squad_states.values(),
            key=lambda item: (item.radius, item.sector_index),
        ):
            if (
                squad.vanguard_id not in members
                or squad.ranger_id not in members
            ):
                # Pair identity survives treatment, but its healthy member is
                # temporarily available to the unpaired home reserve.
                continue
            by_radius[squad.radius].append(squad)
            paired_ids.update((squad.vanguard_id, squad.ranger_id))
        return members, by_radius, paired_ids

    def _paired_patrol_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        members: dict[UUID, EntitySnapshot],
        by_radius: dict[int, list[SquadState]],
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        assert world.core is not None
        intents: list[ActionIntent] = []
        for radius, squads in sorted(by_radius.items()):
            ring = manhattan_ring(world.core.position, radius)
            for local_index, squad in enumerate(squads):
                vanguard = members.get(squad.vanguard_id)
                ranger = members.get(squad.ranger_id)
                if vanguard is None or ranger is None:
                    continue
                start = local_index * len(ring) // len(squads)
                end = (local_index + 1) * len(ring) // len(squads)
                sector = ring[start:end] or ring
                intents.extend(
                    self._squad_patrol_intents(
                        world,
                        projection,
                        (squad.vanguard_id, squad.ranger_id),
                        squad,
                        vanguard,
                        ranger,
                        sector,
                        ring,
                        radius,
                        protected,
                    )
                )
        return intents

    def _squad_patrol_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        squad_key: tuple[UUID, UUID],
        squad: SquadState,
        vanguard: EntitySnapshot,
        ranger: EntitySnapshot,
        sector: tuple[Position, ...],
        ring: tuple[Position, ...],
        radius: int,
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        assert world.core is not None
        anchors = self._patrol_anchor_candidates(
            world,
            projection,
            vanguard,
            sector,
            ring,
            radius,
            protected,
        )
        anchor_set = set(anchors)
        anchor = (
            squad.patrol_anchor
            if squad.patrol_anchor in anchor_set
            else None
        )
        support = squad.support_target if anchor is not None else None
        completed = bool(
            anchor is not None
            and vanguard.position == anchor
            and (support is None or ranger.position == support)
        )
        if completed:
            anchor = None
            support = None
        if anchor is None:
            recent_anchor_cells = set(
                self.memory.position_history.get(vanguard.id, ())[-4:]
            )
            anchor = min(
                anchors,
                key=lambda cell: (
                    int(cell in recent_anchor_cells),
                    self.memory.visit_counts.get(cell, 0),
                    projection.future_attackers(cell),
                    manhattan(vanguard.position, cell),
                    cell,
                ),
                default=None,
            )
        if anchor is None:
            return [
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    UnitMission.PATROL,
                    72,
                    target_position=unit.position,
                    reason="PATROL_SECTOR_UNOBSERVED_HOLD",
                )
                for unit in (vanguard, ranger)
            ]
        if (
            manhattan(vanguard.position, ranger.position)
            > self.config.squad_max_separation
        ):
            return self._reassembly_intents(
                world,
                projection,
                vanguard,
                ranger,
                protected,
            )
        intents = self._move_or_wait(
            world,
            projection,
            vanguard,
            anchor,
            protected,
            "VANGUARD_ANCHOR",
        )
        firing_band = tuple(
            cell
            for cell in self._firing_band(
                world,
                projection,
                anchor,
                protected,
            )
            if manhattan(cell, anchor) <= self.config.squad_max_separation
        )
        if support not in firing_band:
            recent_support_cells = set(
                self.memory.position_history.get(ranger.id, ())[-4:]
            )
            support = min(
                firing_band,
                key=lambda cell: (
                    projection.immediate_attackers(cell),
                    projection.future_attackers(cell),
                    int(cell in recent_support_cells),
                    self.memory.visit_counts.get(cell, 0),
                    manhattan(ranger.position, cell),
                    manhattan(cell, world.core.position),
                    cell,
                ),
                default=None,
            )
        self.memory.squad_states[squad_key] = replace(
            squad,
            patrol_anchor=anchor,
            support_target=support,
            target_assigned_tick=(
                squad.target_assigned_tick
                if squad.patrol_anchor == anchor
                and squad.support_target == support
                else world.tick
            ),
        )
        if support is not None:
            intents.extend(
                self._move_or_wait(
                    world,
                    projection,
                    ranger,
                    support,
                    protected,
                    "RANGER_SUPPORT",
                )
            )
        else:
            intents.append(
                ActionIntent.simple(
                    ranger.id,
                    IntentAction.WAIT,
                    UnitMission.PATROL,
                    72,
                    target_position=ranger.position,
                    reason="RANGER_SUPPORT_UNAVAILABLE_HOLD",
                )
            )
        return intents

    def _reassembly_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        vanguard: EntitySnapshot,
        ranger: EntitySnapshot,
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        assert world.core is not None
        key = (vanguard.id, ranger.id)
        separation = manhattan(vanguard.position, ranger.position)
        previous = self.memory.squad_rendezvous_leases.get(key)
        stalled = 0
        if previous is not None:
            progressed = separation < previous.best_separation
            stalled = 0 if progressed else previous.stalled_ticks + 1

        blocked = frozenset(
            (protected | projection.hostile_occupied)
            - {vanguard.position, ranger.position}
        )
        between = path_to(
            world,
            vanguard.position,
            ranger.position,
            node_limit=self.config.path_node_limit,
            blocked=blocked,
        )
        rendezvous = (
            previous.rendezvous
            if previous is not None
            and previous.rendezvous in world.known_passable
            and previous.rendezvous not in blocked
            else (
                between[len(between) // 2]
                if between
                else ranger.position
            )
        )
        assigned_tick = (
            previous.assigned_tick
            if previous is not None and previous.rendezvous == rendezvous
            else world.tick
        )
        self.memory.squad_rendezvous_leases[key] = SquadRendezvousLease(
            vanguard_id=vanguard.id,
            ranger_id=ranger.id,
            rendezvous=rendezvous,
            assigned_tick=assigned_tick,
            best_separation=min(
                separation,
                previous.best_separation if previous is not None else separation,
            ),
            stalled_ticks=stalled,
            last_vanguard_position=vanguard.position,
            last_ranger_position=ranger.position,
        )

        # Let the outer Vanguard hold while the Ranger is genuinely closing.
        # If authoritative positions fail to improve for two Ticks, move the
        # holder toward the mutually reachable midpoint instead of extending
        # an unbounded PARTNER_HOLD lease.
        if stalled >= self.config.squad_reassembly_no_progress_ticks:
            intents = self._move_or_wait(
                world,
                projection,
                vanguard,
                rendezvous,
                protected,
                "SQUAD_REASSEMBLE_RENDEZVOUS",
            )
            intents.append(
                ActionIntent.simple(
                    ranger.id,
                    IntentAction.WAIT,
                    UnitMission.PATROL,
                    72,
                    target_position=rendezvous,
                    reason="SQUAD_REASSEMBLE_RENDEZVOUS_HOLD",
                    metadata=(
                        ("stalled_ticks", stalled),
                        ("rendezvous", rendezvous),
                    ),
                )
            )
            if stalled >= self.config.squad_reassembly_break_ticks:
                # The pair identity may be rebuilt on the next Tick.  With
                # more than one squad this lets nearest healthy partners be
                # matched again; with one pair the new midpoint still avoids
                # freezing either member indefinitely.
                self.memory.squad_states.pop(key, None)
            return intents

        ranger_route = self._route(
            world,
            projection,
            ranger,
            vanguard.position,
            protected,
        )
        if ranger_route is not None and ranger_route.first_direction is not None:
            mover, holder, target = ranger, vanguard, vanguard.position
        else:
            mover, holder, target = vanguard, ranger, ranger.position
        intents = self._move_or_wait(
            world,
            projection,
            mover,
            target,
            protected,
            "SQUAD_REASSEMBLE",
        )
        intents.append(
            ActionIntent.simple(
                holder.id,
                IntentAction.WAIT,
                UnitMission.PATROL,
                72,
                target_position=holder.position,
                reason="SQUAD_REASSEMBLE_PARTNER_HOLD",
            )
        )
        return intents

    def _reserve_patrol_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        healthy: tuple[EntitySnapshot, ...],
        paired_ids: set[UUID],
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        assert world.core is not None
        intents: list[ActionIntent] = []
        reserve_ring = [
            cell
            for cell in manhattan_ring(
                world.core.position,
                self.config.peaceful_squad_radii[0],
            )
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in protected
        ]
        for unit in sorted(
            (unit for unit in healthy if unit.id not in paired_ids),
            key=lambda item: item.id.bytes,
        ):
            if not reserve_ring:
                intents.append(
                    ActionIntent.simple(
                        unit.id,
                        IntentAction.WAIT,
                        UnitMission.PATROL,
                        72,
                        target_position=unit.position,
                        reason="UNPAIRED_RESERVE_UNAVAILABLE_HOLD",
                    )
                )
                continue
            target = min(
                reserve_ring,
                key=lambda cell: (
                    self.memory.visit_counts.get(cell, 0),
                    manhattan(unit.position, cell),
                    cell,
                ),
            )
            intents.extend(
                self._move_or_wait(
                    world,
                    projection,
                    unit,
                    target,
                    protected,
                    "UNPAIRED_HOME_RESERVE_PATROL",
                )
            )
            reserve_ring.remove(target)
        return intents

    def _patrol_anchor_candidates(
        self,
        world: WorldModel,
        projection: TacticalMap,
        vanguard: EntitySnapshot,
        sector: tuple[Position, ...],
        ring: tuple[Position, ...],
        radius: int,
        protected: frozenset[Position],
    ) -> list[Position]:
        assert world.core is not None

        def usable(cell: Position) -> bool:
            return (
                cell in world.known_passable
                and cell not in world.known_obstacles
                and cell not in protected
                and cell not in projection.hostile_occupied
                and count_open_neighbors(cell, world.known_obstacles) >= 2
            )

        primary = [cell for cell in sector if usable(cell)]
        if primary:
            return primary
        same_layer = [cell for cell in ring if usable(cell)]
        if same_layer:
            return same_layer
        inner_floor = max(2, radius - 5)
        observed_fallback = [
            cell
            for cell in world.known_passable
            if inner_floor <= manhattan(cell, world.core.position) <= radius
            and usable(cell)
        ]
        if observed_fallback:
            return observed_fallback
        return [vanguard.position] if usable(vanguard.position) else []

    def _move_or_wait(
        self,
        world,
        projection,
        unit,
        target,
        protected,
        reason,
        *,
        mission=UnitMission.PATROL,
        move_priority=70,
        wait_priority=72,
    ):
        if unit.position == target:
            return [
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    mission,
                    wait_priority,
                    target_position=target,
                    reason=f"{reason}_HOLD",
                )
            ]
        preferred = self._route(world, projection, unit, target, protected)
        if preferred is None or preferred.first_direction is None:
            return [
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    mission,
                    wait_priority,
                    target_position=target,
                    reason=f"{reason}_ROUTE_BLOCKED",
                )
            ]
        moves: list[ActionIntent] = []
        rejected_dead_end = False
        rejected_no_progress = False
        for index, direction in enumerate(DIRECTION_ORDER):
            dx, dy = direction.delta
            destination = unit.position[0] + dx, unit.position[1] + dy
            if (
                destination in world.known_obstacles
                or destination not in world.known_passable
                or destination in projection.hostile_occupied
                or destination in protected
            ):
                continue
            terminal_exception = self._combat_terminal_exception(
                world,
                unit,
                destination,
                mission,
            )
            viability = move_viability(
                world,
                unit.position,
                destination,
                target=target,
                blocked=frozenset(protected | projection.hostile_occupied),
                node_limit=min(self.config.path_node_limit, 512),
                require_continuation=(
                    terminal_exception is None and destination != target
                ),
                terminal_exception=terminal_exception,
            )
            if not viability.viable:
                rejected_dead_end = True
                continue
            if destination == target:
                remaining_distance = 0
            else:
                remaining = route_to(
                    world,
                    destination,
                    target,
                    node_limit=min(self.config.path_node_limit, 512),
                    blocked=frozenset(
                        (protected | projection.hostile_occupied | {unit.position})
                        - {destination, target}
                    ),
                )
                remaining_distance = None if remaining is None else remaining.distance
            if (
                remaining_distance is None
                or remaining_distance >= preferred.distance
            ):
                rejected_no_progress = True
                continue
            is_preferred = (
                preferred.first_direction is direction
            )
            moves.append(
                ActionIntent.move(
                    unit.id,
                    mission,
                    move_priority,
                    direction,
                    destination,
                    risk=self._risk(projection, destination),
                    exclusive_destination=True,
                    tie_break=(
                        0 if is_preferred else 1,
                        remaining_distance,
                        index,
                    ),
                    reason=reason,
                    metadata=viability.metadata
                    + (
                        ("route_distance_before", preferred.distance),
                        ("route_distance_after", remaining_distance),
                    ),
                )
            )
        moves.sort(key=ActionIntent.sort_key)
        moves.append(
            ActionIntent.simple(
                unit.id,
                IntentAction.WAIT,
                mission,
                wait_priority,
                target_position=target,
                reason=(
                    f"{reason}_NO_VIABLE_CONTINUATION"
                    if rejected_dead_end and not moves
                    else f"{reason}_ROUTE_BLOCKED_THIS_TICK"
                ),
                metadata=(
                    ("dead_end_rejected", rejected_dead_end),
                    ("no_progress_rejected", rejected_no_progress),
                ),
            )
        )
        return moves

    @staticmethod
    def _combat_terminal_exception(world, unit, destination, mission):
        if mission not in {UnitMission.ATTACK, UnitMission.HOME_DEFENSE}:
            return None
        if unit.unit_type is UnitType.VANGUARD and any(
            manhattan(destination, enemy.position) == 1
            for enemy in world.enemies
        ):
            return "ATTACK"
        if unit.unit_type is UnitType.RANGER and any(
            destination in ranger_firing_positions(enemy.position)
            and ranger_line_is_clear(
                destination,
                enemy.position,
                world.known_obstacles,
            )
            for enemy in world.enemies
        ):
            return "ATTACK"
        return None

    def _firing_band(self, world, projection, anchor, protected):
        assert world.core is not None
        return tuple(
            sorted(
                {
                    cell
                    for cell in ranger_firing_positions(anchor)
                    if cell in world.known_passable
                    and cell not in world.known_obstacles
                    and cell not in protected
                    and cell not in projection.hostile_occupied
                    and count_open_neighbors(cell, world.known_obstacles) >= 2
                    and ranger_line_is_clear(cell, anchor, world.known_obstacles)
                },
                key=lambda cell: (
                    projection.immediate_attackers(cell),
                    projection.future_attackers(cell),
                    manhattan(cell, world.core.position),
                    cell,
                ),
            )
        )

    def _route(self, world, projection, unit, target, protected):
        blocked = (projection.hostile_occupied | protected) - {unit.position, target}
        return route_to(
            world,
            unit.position,
            target,
            node_limit=self.config.path_node_limit,
            blocked=frozenset(blocked),
        )

    def _sync_squads(self, combatants: tuple[EntitySnapshot, ...]) -> None:
        living = {unit.id for unit in combatants}
        for key, squad in tuple(self.memory.squad_states.items()):
            if squad.vanguard_id not in living or squad.ranger_id not in living:
                self.memory.squad_states.pop(key, None)
        active_pairs = {
            (squad.vanguard_id, squad.ranger_id)
            for squad in self.memory.squad_states.values()
        }
        for key in tuple(self.memory.squad_rendezvous_leases):
            if key not in active_pairs or not set(key) <= living:
                self.memory.squad_rendezvous_leases.pop(key, None)
        paired_v = {squad.vanguard_id for squad in self.memory.squad_states.values()}
        paired_r = {squad.ranger_id for squad in self.memory.squad_states.values()}
        vanguards = sorted(
            (
                unit
                for unit in combatants
                if unit.unit_type is UnitType.VANGUARD and unit.id not in paired_v
            ),
            key=lambda unit: unit.id.bytes,
        )
        rangers = sorted(
            (
                unit
                for unit in combatants
                if unit.unit_type is UnitType.RANGER and unit.id not in paired_r
            ),
            key=lambda unit: unit.id.bytes,
        )
        while vanguards and rangers:
            _, _, _, vanguard, ranger = min(
                (
                    (manhattan(v.position, r.position), v.id.bytes, r.id.bytes, v, r)
                    for v in vanguards
                    for r in rangers
                ),
                key=lambda item: item[:3],
            )
            key = vanguard.id, ranger.id
            self.memory.squad_states[key] = SquadState(
                vanguard.id,
                ranger.id,
                self.config.peaceful_squad_radii[0],
                0,
            )
            vanguards.remove(vanguard)
            rangers.remove(ranger)
        ordered = sorted(
            self.memory.squad_states.items(),
            key=lambda item: (item[0][0].bytes, item[0][1].bytes),
        )
        total = len(ordered)
        if total <= 2:
            counts = (1, max(0, total - 1), 0)
        else:
            base = total // 3
            counts = (base, base, total - base * 2)
        first_end, second_end = counts[0], counts[0] + counts[1]
        for index, (key, squad) in enumerate(ordered):
            layer = 0 if index < first_end else (1 if index < second_end else 2)
            layer = min(layer, len(self.config.peaceful_squad_radii) - 1)
            self.memory.squad_states[key] = replace(
                squad,
                radius=self.config.peaceful_squad_radii[layer],
                sector_index=index,
            )

    @staticmethod
    def _sector_direction(core: Position, threat: Position) -> Direction:
        dx, dy = threat[0] - core[0], threat[1] - core[1]
        if abs(dx) > abs(dy):
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        return Direction.DOWN if dy > 0 else Direction.UP

    @staticmethod
    def _advance(origin: Position, direction: Direction, distance: int) -> Position:
        dx, dy = direction.delta
        return origin[0] + dx * distance, origin[1] + dy * distance

    @staticmethod
    def _risk(projection: TacticalMap, cell: Position | None) -> int:
        if cell is None:
            return 0
        immediate, future, remembered = projection.exposure(cell)
        return immediate * 100 + future * 10 + remembered
