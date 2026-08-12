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
    FormationMoveFeedback,
    IntentAction,
    IntentResolution,
    PairingCooldown,
    PeacefulFormationAssignment,
    SquadFormationBundle,
    SquadFormationLease,
    SquadRendezvousLease,
    SquadState,
    UnitMission,
    WorldModel,
)
from .planning import Route, move_viability, path_to, route_to
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
        self._sync_squads(living_combatants, world.tick)
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
        self.memory.defense_reserve_leases.clear()
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

    def observe_resolution(
        self,
        world: WorldModel,
        resolution: IntentResolution,
    ) -> None:
        """Persist only the resolver evidence needed to unstick next Turn."""

        formation_actors = {
            actor_id
            for key in self.memory.squad_states
            for actor_id in key
        }
        selected_by_actor = {
            intent.actor_id: intent
            for intent in resolution.selected
            if intent.actor_id is not None
        }
        conflicts: dict[UUID, str] = {}
        conflict_priority = {
            "HEAD_ON_SWAP": 0,
            "CELL_CAPACITY": 1,
            "RESERVATION_CONFLICT": 2,
        }
        for rejected in resolution.rejected:
            actor_id = rejected.intent.actor_id
            if actor_id not in formation_actors:
                continue
            if rejected.reason not in conflict_priority:
                continue
            previous = conflicts.get(actor_id)
            if (
                previous is None
                or conflict_priority[rejected.reason]
                < conflict_priority[previous]
            ):
                conflicts[actor_id] = rejected.reason
        for actor_id in tuple(self.memory.formation_move_feedback):
            if actor_id not in formation_actors:
                self.memory.formation_move_feedback.pop(actor_id, None)
        for actor_id in formation_actors:
            selected = selected_by_actor.get(actor_id)
            if selected is None or selected.mission not in {
                UnitMission.PATROL,
                UnitMission.HOME_DEFENSE,
            }:
                self.memory.formation_move_feedback.pop(actor_id, None)
                continue
            previous = self.memory.formation_move_feedback.get(actor_id)
            rejection = conflicts.get(actor_id)
            blocked_now = (
                rejection is not None
                or "ROUTE_BLOCKED" in selected.reason
                or selected.reason in {
                    "NO_VIABLE_FORMATION_MOVE",
                    "NO_VIABLE_RENDEZVOUS",
                }
            )
            consecutive_blocked = (
                previous.consecutive_blocked_ticks + 1
                if blocked_now
                and previous is not None
                and previous.tick == world.tick - 1
                and previous.target_position == selected.target_position
                else int(blocked_now)
            )
            self.memory.formation_move_feedback[actor_id] = FormationMoveFeedback(
                actor_id=actor_id,
                tick=world.tick,
                action=selected.action.value,
                reason=selected.reason,
                target_position=selected.target_position,
                rejection_reason=rejection,
                consecutive_blocked_ticks=consecutive_blocked,
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
                        target_position=unit.position,
                        reason="NO_VIABLE_FORMATION_MOVE",
                        metadata=(
                            ("hold_class", "NO_VIABLE_MOVE"),
                            ("defense_active", True),
                        ),
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
                    metadata=(
                        ("hold_class", "TACTICAL_HOLD"),
                        ("formation_role", role),
                        ("lease_ticks", self.config.tactical_position_lease_ticks),
                    ),
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

        remaining_ids = {unit.id for unit in (*vanguards, *rangers)}
        for unit_id in tuple(self.memory.defense_reserve_leases):
            if unit_id not in remaining_ids:
                self.memory.defense_reserve_leases.pop(unit_id, None)
        occupied = {
            unit.position: unit.id
            for unit in world.friendlies
        }
        threat_cells = {
            enemy.position
            for _, _, enemies in front_positions
            for enemy in enemies
        }
        front_cells = {front for _, front, _ in front_positions}
        reserve_targets = [
            cell
            for cell in world.known_passable
            if 3 <= manhattan(cell, world.core.position) <= self.config.home_engage_radius
            and cell not in world.known_obstacles
            and cell not in protected
            and cell not in reserved
            and cell not in projection.hostile_occupied
            and count_open_neighbors(cell, world.known_obstacles) >= 2
            and (
                manhattan(cell, world.core.position) <= 5
                or min(
                    (manhattan(cell, front) for front in front_cells),
                    default=0,
                ) <= 6
            )
        ]
        for unit in (*vanguards, *rangers):
            available = [
                cell
                for cell in reserve_targets
                if occupied.get(cell, unit.id) == unit.id
                or cell not in occupied
            ]
            previous = self.memory.defense_reserve_leases.get(unit.id)
            previous_target = None
            if previous is not None:
                previous_cell, assigned_tick, _ = previous
                if (
                    previous_cell in available
                    and world.tick - assigned_tick
                    < self.config.tactical_position_lease_ticks
                ):
                    previous_target = previous_cell
            scored: list[tuple[tuple[int, ...], Position, str]] = []
            for cell in available:
                route = route_to(
                    world,
                    unit.position,
                    cell,
                    node_limit=min(self.config.path_node_limit, 512),
                    blocked=frozenset(
                        (protected | projection.hostile_occupied | reserved)
                        - {unit.position, cell}
                    ),
                )
                if route is None:
                    continue
                if unit.unit_type is UnitType.RANGER:
                    coverage = sum(
                        ranger_line_is_clear(cell, threat, world.known_obstacles)
                        and cell in ranger_firing_positions(threat)
                        for threat in threat_cells
                    )
                else:
                    coverage = sum(
                        manhattan(cell, threat) <= 2
                        for threat in threat_cells
                    )
                front_distance = min(
                    (manhattan(cell, front) for front in front_cells),
                    default=0,
                )
                role = (
                    "SECTOR_RESERVE"
                    if coverage
                    else (
                        "TACTICAL_SECOND_LINE"
                        if front_distance <= 4
                        else "TACTICAL_CORE_GUARD"
                    )
                )
                scored.append(
                    (
                        (
                            int(cell != previous_target),
                            -coverage,
                            front_distance,
                            projection.immediate_attackers(cell),
                            projection.future_attackers(cell),
                            route.distance,
                            manhattan(cell, world.core.position),
                            cell[0],
                            cell[1],
                        ),
                        cell,
                        role,
                    )
                )
            if not scored:
                continue
            _, target, role = min(scored, key=lambda item: item[0])
            tasks[unit.id] = target, role
            reserve_targets.remove(target)
            old = self.memory.defense_reserve_leases.get(unit.id)
            self.memory.defense_reserve_leases[unit.id] = (
                target,
                old[1] if old is not None and old[0] == target else world.tick,
                role,
            )
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
        patrol_entries: list[
            tuple[
                SquadState,
                EntitySnapshot,
                EntitySnapshot,
                tuple[Position, ...],
                tuple[Position, ...],
            ]
        ] = []
        for radius, squads in sorted(by_radius.items()):
            ring = manhattan_ring(world.core.position, radius)
            for local_index, squad in enumerate(squads):
                vanguard = members.get(squad.vanguard_id)
                ranger = members.get(squad.ranger_id)
                if vanguard is None or ranger is None:
                    continue
                if (
                    manhattan(vanguard.position, ranger.position)
                    > self.config.squad_max_separation
                ):
                    intents.extend(
                        self._reassembly_intents(
                            world,
                            projection,
                            vanguard,
                            ranger,
                            protected,
                        )
                    )
                    continue
                start = local_index * len(ring) // len(squads)
                end = (local_index + 1) * len(ring) // len(squads)
                sector = ring[start:end] or ring
                patrol_entries.append((squad, vanguard, ranger, sector, ring))

        route_cache: dict[
            tuple[Position, Position, frozenset[Position]], Route | None
        ] = {}
        occupied_sets: dict[Position, set[UUID]] = defaultdict(set)
        for unit in world.friendlies:
            occupied_sets[unit.position].add(unit.id)
        occupied_by = {
            position: frozenset(actor_ids)
            for position, actor_ids in occupied_sets.items()
        }
        candidate_map: dict[
            tuple[UUID, UUID], tuple[SquadFormationBundle, ...]
        ] = {}
        entry_by_key = {
            (squad.vanguard_id, squad.ranger_id): (
                squad,
                vanguard,
                ranger,
                sector,
                ring,
            )
            for squad, vanguard, ranger, sector, ring in patrol_entries
        }
        for squad, vanguard, ranger, sector, ring in patrol_entries:
            key = squad.vanguard_id, squad.ranger_id
            self._refresh_formation_lease(
                world,
                projection,
                squad,
                vanguard,
                ranger,
                protected,
                route_cache,
            )
            candidate_map[key] = self._formation_candidates(
                world,
                projection,
                squad,
                vanguard,
                ranger,
                sector,
                ring,
                protected,
                occupied_by,
                route_cache,
            )

        order = sorted(
            entry_by_key,
            key=lambda key: (
                len(candidate_map[key]),
                -(
                    self.memory.squad_formation_leases.get(key).stalled_ticks
                    if key in self.memory.squad_formation_leases
                    else 0
                ),
                entry_by_key[key][0].radius,
                key[0].bytes,
                key[1].bytes,
            ),
        )
        assigned: dict[tuple[UUID, UUID], SquadFormationBundle] = {}
        claimed_slots: set[Position] = set()
        claimed_steps: set[Position] = set()
        rejected: list[tuple[UUID, UUID, Position, Position, str]] = []
        for key in order:
            chosen = None
            for candidate in candidate_map[key]:
                slots = {candidate.anchor, candidate.support}
                steps = {
                    cell
                    for cell in (
                        candidate.vanguard_first_position,
                        candidate.ranger_first_position,
                    )
                    if cell is not None
                }
                if slots & claimed_slots:
                    rejected.append((*key, candidate.anchor, candidate.support, "SLOT_RESERVED"))
                    continue
                if steps & claimed_steps or steps & claimed_slots:
                    rejected.append((*key, candidate.anchor, candidate.support, "FIRST_STEP_RESERVED"))
                    continue
                if slots & claimed_steps:
                    rejected.append((*key, candidate.anchor, candidate.support, "SLOT_BLOCKS_FIRST_STEP"))
                    continue
                chosen = candidate
                break
            if chosen is None:
                continue
            assigned[key] = chosen
            claimed_slots.update((chosen.anchor, chosen.support))
            claimed_steps.update(
                cell
                for cell in (
                    chosen.vanguard_first_position,
                    chosen.ranger_first_position,
                )
                if cell is not None
            )

        vacating_actors = {
            actor_id
            for bundle in assigned.values()
            for actor_id, destination in (
                (bundle.vanguard_id, bundle.vanguard_first_position),
                (bundle.ranger_id, bundle.ranger_first_position),
            )
            if destination is not None
        }
        if vacating_actors:
            relaxed_occupancy = {
                position: remaining
                for position, actor_ids in occupied_by.items()
                if (remaining := actor_ids - vacating_actors)
            }
            for key in tuple(item for item in order if item not in assigned):
                squad, vanguard, ranger, sector, ring = entry_by_key[key]
                extra = self._formation_candidates(
                    world,
                    projection,
                    squad,
                    vanguard,
                    ranger,
                    sector,
                    ring,
                    protected,
                    relaxed_occupancy,
                    route_cache,
                )
                merged = tuple(
                    sorted(
                        {*(candidate_map[key]), *extra},
                        key=lambda item: item.score,
                    )[: self.config.formation_candidate_limit]
                )
                candidate_map[key] = merged
                chosen = next(
                    (
                        candidate
                        for candidate in merged
                        if not self._formation_bundle_conflicts(
                            candidate,
                            tuple(assigned.values()),
                        )
                    ),
                    None,
                )
                if chosen is None:
                    continue
                assigned[key] = chosen
                claimed_slots.update((chosen.anchor, chosen.support))
                claimed_steps.update(
                    cell
                    for cell in (
                        chosen.vanguard_first_position,
                        chosen.ranger_first_position,
                    )
                    if cell is not None
                )

        # One deterministic augmenting repair avoids UUID-order starvation:
        # an unassigned squad may take a contested bundle when the current
        # owner has a different feasible bundle that does not disturb any
        # other assignment.  Bound the repair to one displaced squad so the
        # live command window remains predictable.
        for missing_key in tuple(key for key in order if key not in assigned):
            repaired = False
            for candidate in candidate_map[missing_key]:
                for victim_key in sorted(
                    assigned,
                    key=lambda item: (item[0].bytes, item[1].bytes),
                ):
                    others = tuple(
                        bundle
                        for key, bundle in assigned.items()
                        if key != victim_key
                    )
                    if self._formation_bundle_conflicts(candidate, others):
                        continue
                    alternative = next(
                        (
                            bundle
                            for bundle in candidate_map[victim_key]
                            if bundle != assigned[victim_key]
                            and not self._formation_bundle_conflicts(
                                bundle,
                                (*others, candidate),
                            )
                        ),
                        None,
                    )
                    if alternative is None:
                        continue
                    assigned[missing_key] = candidate
                    assigned[victim_key] = alternative
                    repaired = True
                    break
                if repaired:
                    break
        claimed_slots = {
            cell
            for bundle in assigned.values()
            for cell in (bundle.anchor, bundle.support)
        }
        claimed_steps = {
            cell
            for bundle in assigned.values()
            for cell in (
                bundle.vanguard_first_position,
                bundle.ranger_first_position,
            )
            if cell is not None
        }

        all_reserved = frozenset(claimed_slots | claimed_steps)
        for key, bundle in sorted(
            assigned.items(),
            key=lambda item: (item[0][0].bytes, item[0][1].bytes),
        ):
            squad, vanguard, ranger, _, _ = entry_by_key[key]
            self._store_formation_bundle(world, squad, bundle, vanguard, ranger)
            own = {
                bundle.anchor,
                bundle.support,
                bundle.vanguard_first_position,
                bundle.ranger_first_position,
            }
            bundle_protected = frozenset(
                (set(protected) | set(all_reserved)) - own - {None}
            )
            intents.extend(
                self._formation_member_intents(
                    world,
                    projection,
                    vanguard,
                    bundle.anchor,
                    ranger,
                    bundle_protected,
                    "VANGUARD_ANCHOR",
                )
            )
            intents.extend(
                self._formation_member_intents(
                    world,
                    projection,
                    ranger,
                    bundle.support,
                    vanguard,
                    bundle_protected,
                    "RANGER_SUPPORT",
                )
            )

        unassigned = tuple(key for key in order if key not in assigned)
        for key in unassigned:
            _, vanguard, ranger, _, _ = entry_by_key[key]
            self.memory.squad_formation_leases.pop(key, None)
            for unit in (vanguard, ranger):
                intents.append(
                    ActionIntent.simple(
                        unit.id,
                        IntentAction.WAIT,
                        UnitMission.PATROL,
                        72,
                        target_position=unit.position,
                        reason="NO_VIABLE_FORMATION_MOVE",
                        metadata=(
                            ("hold_class", "NO_VIABLE_MOVE"),
                            ("candidate_count", len(candidate_map[key])),
                        ),
                    )
                )
        self.memory.peaceful_formation_assignment = PeacefulFormationAssignment(
            tick=world.tick,
            bundles=tuple(
                assigned[key]
                for key in sorted(assigned, key=lambda item: (item[0].bytes, item[1].bytes))
            ),
            reserved_positions=tuple(sorted(all_reserved)),
            unassigned_squads=unassigned,
            rejected=tuple(rejected[:64]),
        )
        return intents

    @staticmethod
    def _formation_bundle_conflicts(
        candidate: SquadFormationBundle,
        assigned: tuple[SquadFormationBundle, ...],
    ) -> bool:
        candidate_slots = {candidate.anchor, candidate.support}
        candidate_steps = {
            cell
            for cell in (
                candidate.vanguard_first_position,
                candidate.ranger_first_position,
            )
            if cell is not None
        }
        for bundle in assigned:
            slots = {bundle.anchor, bundle.support}
            steps = {
                cell
                for cell in (
                    bundle.vanguard_first_position,
                    bundle.ranger_first_position,
                )
                if cell is not None
            }
            if (
                candidate_slots & slots
                or candidate_steps & steps
                or candidate_steps & slots
                or candidate_slots & steps
            ):
                return True
            candidate_moves = {
                candidate.vanguard_origin: candidate.vanguard_first_position,
                candidate.ranger_origin: candidate.ranger_first_position,
            }
            assigned_moves = {
                bundle.vanguard_origin: bundle.vanguard_first_position,
                bundle.ranger_origin: bundle.ranger_first_position,
            }
            if any(
                destination is not None
                and other_destination is not None
                and destination == other_origin
                and other_destination == origin
                for origin, destination in candidate_moves.items()
                for other_origin, other_destination in assigned_moves.items()
            ):
                return True
        return False

    def _formation_candidates(
        self,
        world: WorldModel,
        projection: TacticalMap,
        squad: SquadState,
        vanguard: EntitySnapshot,
        ranger: EntitySnapshot,
        sector: tuple[Position, ...],
        ring: tuple[Position, ...],
        protected: frozenset[Position],
        occupied_by: dict[Position, frozenset[UUID]],
        route_cache: dict[tuple[Position, Position, frozenset[Position]], Route | None],
    ) -> tuple[SquadFormationBundle, ...]:
        key = squad.vanguard_id, squad.ranger_id
        occupied_elsewhere = frozenset(
            position
            for position, actor_ids in occupied_by.items()
            if any(actor_id not in key for actor_id in actor_ids)
        )
        blocked = frozenset(
            protected | projection.hostile_occupied | occupied_elsewhere
        )
        anchors = self._patrol_anchor_candidates(
            world,
            projection,
            vanguard,
            sector,
            ring,
            squad.radius,
            blocked,
        )
        recent_anchor = set(self.memory.position_history.get(vanguard.id, ())[-4:])
        recent_support = set(self.memory.position_history.get(ranger.id, ())[-4:])
        anchors = sorted(
            anchors,
            key=lambda cell: (
                int(cell in recent_anchor),
                self.memory.visit_counts.get(cell, 0),
                projection.future_attackers(cell),
                manhattan(vanguard.position, cell),
                cell,
            ),
        )[: self.config.formation_candidate_limit * 2]
        candidates: list[SquadFormationBundle] = []
        for anchor in anchors:
            vanguard_route = self._cached_formation_route(
                world,
                vanguard.position,
                anchor,
                blocked - {anchor},
                route_cache,
            )
            if vanguard_route is None:
                continue
            firing_band = tuple(
                cell
                for cell in self._firing_band(
                    world,
                    projection,
                    anchor,
                    blocked - {anchor},
                )
                if 2 <= manhattan(cell, anchor) <= 3
            )
            for support in firing_band:
                if support == anchor:
                    continue
                backoff_key = (*key, anchor, support)
                if self.memory.squad_target_backoffs.get(backoff_key, 0) >= world.tick:
                    continue
                ranger_route = self._cached_formation_route(
                    world,
                    ranger.position,
                    support,
                    blocked - {support},
                    route_cache,
                )
                if ranger_route is None:
                    continue
                if (
                    vanguard_route.first_position == ranger.position
                    and ranger_route.first_position == vanguard.position
                ):
                    continue
                if vanguard_route.first_position == ranger_route.first_position:
                    continue
                stable = int(
                    squad.patrol_anchor != anchor
                    or squad.support_target != support
                )
                score = (
                    stable,
                    int(anchor in recent_anchor) + int(support in recent_support),
                    self.memory.visit_counts.get(anchor, 0)
                    + self.memory.visit_counts.get(support, 0),
                    max(vanguard_route.distance, ranger_route.distance),
                    vanguard_route.distance + ranger_route.distance,
                    projection.future_attackers(anchor)
                    + projection.future_attackers(support),
                    anchor[0],
                    anchor[1],
                    support[0],
                    support[1],
                )
                candidates.append(
                    SquadFormationBundle(
                        vanguard_id=vanguard.id,
                        ranger_id=ranger.id,
                        vanguard_origin=vanguard.position,
                        ranger_origin=ranger.position,
                        anchor=anchor,
                        support=support,
                        vanguard_route_distance=vanguard_route.distance,
                        ranger_route_distance=ranger_route.distance,
                        vanguard_first_direction=vanguard_route.first_direction,
                        vanguard_first_position=vanguard_route.first_position,
                        ranger_first_direction=ranger_route.first_direction,
                        ranger_first_position=ranger_route.first_position,
                        score=score,
                    )
                )
        return tuple(
            sorted(candidates, key=lambda item: item.score)[
                : self.config.formation_candidate_limit
            ]
        )

    def _cached_formation_route(
        self,
        world: WorldModel,
        start: Position,
        target: Position,
        blocked: frozenset[Position],
        cache: dict[tuple[Position, Position, frozenset[Position]], Route | None],
    ) -> Route | None:
        key = start, target, blocked
        if key not in cache:
            cache[key] = route_to(
                world,
                start,
                target,
                node_limit=min(self.config.path_node_limit, 512),
                blocked=blocked - {start, target},
            )
        return cache[key]

    def _refresh_formation_lease(
        self,
        world: WorldModel,
        projection: TacticalMap,
        squad: SquadState,
        vanguard: EntitySnapshot,
        ranger: EntitySnapshot,
        protected: frozenset[Position],
        route_cache: dict[tuple[Position, Position, frozenset[Position]], Route | None],
    ) -> None:
        key = squad.vanguard_id, squad.ranger_id
        previous = self.memory.squad_formation_leases.get(key)
        if previous is None or (
            squad.patrol_anchor != previous.anchor
            or squad.support_target != previous.support
        ):
            return
        blocked = frozenset(protected | projection.hostile_occupied)
        v_route = self._cached_formation_route(
            world,
            vanguard.position,
            previous.anchor,
            blocked - {vanguard.position, previous.anchor},
            route_cache,
        )
        r_route = self._cached_formation_route(
            world,
            ranger.position,
            previous.support,
            blocked - {ranger.position, previous.support},
            route_cache,
        )
        v_distance = None if v_route is None else v_route.distance
        r_distance = None if r_route is None else r_route.distance
        v_arrived = previous.vanguard_arrived or vanguard.position == previous.anchor
        r_arrived = previous.ranger_arrived or ranger.position == previous.support
        progressed = (
            (v_arrived and not previous.vanguard_arrived)
            or (r_arrived and not previous.ranger_arrived)
            or (
                v_distance is not None
                and previous.vanguard_best_distance is not None
                and v_distance < previous.vanguard_best_distance
            )
            or (
                r_distance is not None
                and previous.ranger_best_distance is not None
                and r_distance < previous.ranger_best_distance
            )
        )
        consecutive = world.tick == previous.last_evaluated_tick + 1
        stalled = 0 if progressed or not consecutive else previous.stalled_ticks + 1
        feedback = tuple(
            item
            for actor_id in key
            if (
                item := self.memory.formation_move_feedback.get(actor_id)
            ) is not None
            and item.tick == world.tick - 1
        )
        rejection = next(
            (
                item.rejection_reason
                for item in feedback
                if item.rejection_reason
                in {"CELL_CAPACITY", "RESERVATION_CONFLICT", "HEAD_ON_SWAP"}
            ),
            None,
        )
        blocked_ticks = (
            previous.blocked_ticks + 1
            if rejection is not None and consecutive
            else 0
        )
        one_arrived = v_arrived != r_arrived
        hold_ticks = (
            previous.partner_hold_ticks + 1
            if one_arrived and consecutive
            else (1 if one_arrived else 0)
        )
        completed = v_arrived and r_arrived
        invalid = (
            stalled >= self.config.formation_target_stall_ticks
            or blocked_ticks >= self.config.formation_target_stall_ticks
            or hold_ticks > self.config.formation_partner_hold_ticks
        )
        if completed or invalid:
            if invalid:
                self.memory.squad_target_backoffs[
                    (*key, previous.anchor, previous.support)
                ] = world.tick + self.config.formation_target_backoff_ticks
            self.memory.squad_formation_leases.pop(key, None)
            self.memory.squad_states[key] = replace(
                squad,
                patrol_anchor=None,
                support_target=None,
                target_assigned_tick=None,
            )
            return
        self.memory.squad_formation_leases[key] = replace(
            previous,
            last_evaluated_tick=world.tick,
            vanguard_best_distance=self._best_distance(
                previous.vanguard_best_distance, v_distance
            ),
            ranger_best_distance=self._best_distance(
                previous.ranger_best_distance, r_distance
            ),
            vanguard_arrived=v_arrived,
            ranger_arrived=r_arrived,
            stalled_ticks=stalled,
            blocked_ticks=blocked_ticks,
            partner_hold_ticks=hold_ticks,
            last_vanguard_position=vanguard.position,
            last_ranger_position=ranger.position,
            last_rejection_reason=rejection,
        )

    @staticmethod
    def _best_distance(previous: int | None, current: int | None) -> int | None:
        if previous is None:
            return current
        if current is None:
            return previous
        return min(previous, current)

    def _store_formation_bundle(
        self,
        world: WorldModel,
        squad: SquadState,
        bundle: SquadFormationBundle,
        vanguard: EntitySnapshot,
        ranger: EntitySnapshot,
    ) -> None:
        key = squad.vanguard_id, squad.ranger_id
        existing = self.memory.squad_formation_leases.get(key)
        same = (
            existing is not None
            and existing.anchor == bundle.anchor
            and existing.support == bundle.support
        )
        if not same:
            existing = SquadFormationLease(
                vanguard_id=vanguard.id,
                ranger_id=ranger.id,
                anchor=bundle.anchor,
                support=bundle.support,
                assigned_tick=world.tick,
                last_evaluated_tick=world.tick,
                vanguard_best_distance=bundle.vanguard_route_distance,
                ranger_best_distance=bundle.ranger_route_distance,
                vanguard_arrived=vanguard.position == bundle.anchor,
                ranger_arrived=ranger.position == bundle.support,
                partner_hold_ticks=int(
                    (vanguard.position == bundle.anchor)
                    != (ranger.position == bundle.support)
                ),
                last_vanguard_position=vanguard.position,
                last_ranger_position=ranger.position,
            )
            self.memory.squad_formation_leases[key] = existing
        self.memory.squad_states[key] = replace(
            squad,
            patrol_anchor=bundle.anchor,
            support_target=bundle.support,
            target_assigned_tick=(
                squad.target_assigned_tick if same else world.tick
            ),
        )

    def _formation_member_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        unit: EntitySnapshot,
        target: Position,
        partner: EntitySnapshot,
        protected: frozenset[Position],
        reason: str,
    ) -> list[ActionIntent]:
        if unit.position == target:
            return [
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    UnitMission.PATROL,
                    72,
                    target_position=target,
                    reason="WAIT_FOR_PARTNER_PROGRESS",
                    metadata=(
                        ("hold_class", "PARTNER_PROGRESS_HOLD"),
                        ("partner_id", str(partner.id)),
                        ("formation_role", reason),
                    ),
                )
            ]
        return self._move_or_wait(
            world,
            projection,
            unit,
            target,
            protected,
            reason,
        )

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
        blocked = frozenset(
            (
                protected
                | projection.hostile_occupied
                | {
                    unit.position
                    for unit in world.friendlies
                    if unit.id not in key
                }
            )
            - {vanguard.position, ranger.position}
        )
        between = path_to(
            world,
            vanguard.position,
            ranger.position,
            node_limit=self.config.path_node_limit,
            blocked=blocked,
        )
        candidates: set[Position] = set()
        if between:
            middle = len(between) // 2
            candidates.update(between[max(0, middle - 2) : middle + 3])
            for cell in tuple(candidates):
                candidates.update(neighbor for _, neighbor in cardinal_neighbors(cell))
        if previous is not None:
            candidates.add(previous.rendezvous)
        candidates.update((vanguard.position, ranger.position))
        viable: list[tuple[tuple[int, ...], Position, Route, Route]] = []
        for cell in candidates:
            if (
                cell not in world.known_passable
                or cell in world.known_obstacles
                or cell in blocked
                or count_open_neighbors(cell, world.known_obstacles) < 2
            ):
                continue
            v_route = route_to(
                world,
                vanguard.position,
                cell,
                node_limit=min(self.config.path_node_limit, 512),
                blocked=blocked - {cell},
            )
            r_route = route_to(
                world,
                ranger.position,
                cell,
                node_limit=min(self.config.path_node_limit, 512),
                blocked=blocked - {cell},
            )
            if v_route is None or r_route is None:
                continue
            stable = int(
                previous is None
                or previous.rendezvous != cell
                or previous.stalled_ticks
                >= self.config.squad_reassembly_no_progress_ticks
            )
            viable.append(
                (
                    (
                        stable,
                        max(v_route.distance, r_route.distance),
                        v_route.distance + r_route.distance,
                        projection.future_attackers(cell),
                        cell[0],
                        cell[1],
                    ),
                    cell,
                    v_route,
                    r_route,
                )
            )
        if viable:
            _, rendezvous, v_route, r_route = min(viable, key=lambda item: item[0])
            combined_distance = v_route.distance + r_route.distance
        else:
            rendezvous = previous.rendezvous if previous is not None else vanguard.position
            v_route = r_route = None
            combined_distance = None
        progressed = bool(
            previous is not None
            and (
                separation < previous.best_separation
                or (
                    combined_distance is not None
                    and previous.best_route_distance is not None
                    and combined_distance < previous.best_route_distance
                )
            )
        )
        stalled = (
            0
            if previous is None or progressed
            else previous.stalled_ticks + 1
        )
        if stalled >= self.config.squad_reassembly_break_ticks:
            cooldown = PairingCooldown(
                vanguard_id=vanguard.id,
                ranger_id=ranger.id,
                expires_tick=world.tick + self.config.formation_pair_cooldown_ticks,
            )
            self.memory.squad_pairing_cooldowns[key] = cooldown
            self.memory.squad_states.pop(key, None)
            self.memory.squad_formation_leases.pop(key, None)
            self.memory.squad_rendezvous_leases.pop(key, None)
            return [
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    UnitMission.PATROL,
                    72,
                    target_position=unit.position,
                    reason="PAIRING_COOLDOWN_REASSIGN",
                    metadata=(
                        ("hold_class", "BLOCKED_WAIT"),
                        ("cooldown_until", cooldown.expires_tick),
                    ),
                )
                for unit in (vanguard, ranger)
            ]
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
            best_route_distance=(
                combined_distance
                if previous is None or previous.best_route_distance is None
                else (
                    previous.best_route_distance
                    if combined_distance is None
                    else min(previous.best_route_distance, combined_distance)
                )
            ),
            stalled_ticks=stalled,
            last_vanguard_position=vanguard.position,
            last_ranger_position=ranger.position,
        )
        if v_route is None or r_route is None:
            return [
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    UnitMission.PATROL,
                    72,
                    target_position=unit.position,
                    reason="NO_VIABLE_RENDEZVOUS",
                    metadata=(("hold_class", "NO_VIABLE_MOVE"),),
                )
                for unit in (vanguard, ranger)
            ]
        intents: list[ActionIntent] = []
        for unit in (vanguard, ranger):
            if unit.position == rendezvous:
                intents.append(
                    ActionIntent.simple(
                        unit.id,
                        IntentAction.WAIT,
                        UnitMission.PATROL,
                        72,
                        target_position=rendezvous,
                        reason="WAIT_FOR_RENDEZVOUS_PROGRESS",
                        metadata=(
                            ("hold_class", "PARTNER_PROGRESS_HOLD"),
                            ("stalled_ticks", stalled),
                        ),
                    )
                )
            else:
                intents.extend(
                    self._move_or_wait(
                        world,
                        projection,
                        unit,
                        rendezvous,
                        protected,
                        "SQUAD_REASSEMBLE_RENDEZVOUS",
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
        occupied = {unit.position for unit in world.friendlies}
        reserve_ring = [
            cell
            for radius in self.config.peaceful_squad_radii
            for cell in manhattan_ring(world.core.position, radius)
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in protected
            and cell not in projection.hostile_occupied
            and cell not in occupied
            and count_open_neighbors(cell, world.known_obstacles) >= 2
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
                        reason="NO_VIABLE_FORMATION_MOVE",
                        metadata=(("hold_class", "NO_VIABLE_MOVE"),),
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
        feedback = self.memory.formation_move_feedback.get(unit.id)
        if (
            mission in {UnitMission.PATROL, UnitMission.HOME_DEFENSE}
            and feedback is not None
            and feedback.tick == world.tick - 1
            and feedback.target_position == target
            and feedback.consecutive_blocked_ticks
            >= self.config.formation_target_stall_ticks
        ):
            history = self.memory.position_history.get(unit.id, ())
            previous_position = next(
                (
                    position
                    for position in reversed(history[:-1])
                    if position != unit.position
                ),
                None,
            )
            occupied = {
                other.position
                for other in world.friendlies
                if other.id != unit.id
            }
            normal_destinations = {
                intent.target_position
                for intent in moves
                if intent.action is IntentAction.MOVE
            }
            for index, direction in enumerate(DIRECTION_ORDER):
                dx, dy = direction.delta
                destination = unit.position[0] + dx, unit.position[1] + dy
                if (
                    destination == previous_position
                    or destination == target
                    or destination in normal_destinations
                    or destination in occupied
                    or destination in world.known_obstacles
                    or destination not in world.known_passable
                    or destination in projection.hostile_occupied
                    or destination in protected
                ):
                    continue
                viability = move_viability(
                    world,
                    unit.position,
                    destination,
                    target=target,
                    blocked=frozenset(
                        protected | projection.hostile_occupied | occupied
                    ),
                    node_limit=min(self.config.path_node_limit, 512),
                    require_continuation=True,
                )
                if not viability.viable:
                    continue
                continuation = route_to(
                    world,
                    destination,
                    target,
                    node_limit=min(self.config.path_node_limit, 512),
                    blocked=frozenset(
                        (
                            protected
                            | projection.hostile_occupied
                            | occupied
                            | {unit.position}
                        )
                        - {destination, target}
                    ),
                )
                if (
                    continuation is None
                    or continuation.distance > preferred.distance + 1
                ):
                    continue
                free_exits = sum(
                    neighbor in world.known_passable
                    and neighbor not in world.known_obstacles
                    and neighbor not in protected
                    and neighbor not in projection.hostile_occupied
                    and neighbor not in occupied
                    for _, neighbor in cardinal_neighbors(destination)
                )
                if free_exits < 2:
                    continue
                moves.append(
                    ActionIntent.move(
                        unit.id,
                        mission,
                        move_priority + 1,
                        direction,
                        destination,
                        risk=self._risk(projection, destination),
                        exclusive_destination=True,
                        tie_break=(
                            continuation.distance,
                            -free_exits,
                            index,
                        ),
                        reason="FORMATION_YIELD",
                        metadata=viability.metadata
                        + (
                            ("yield_for", feedback.reason),
                            (
                                "blocked_ticks",
                                feedback.consecutive_blocked_ticks,
                            ),
                            ("route_distance_before", preferred.distance),
                            ("route_distance_after", continuation.distance),
                            (
                                "yield_expires_tick",
                                world.tick + self.config.formation_yield_ticks,
                            ),
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

    def _sync_squads(
        self,
        combatants: tuple[EntitySnapshot, ...],
        tick: int,
    ) -> None:
        living = {unit.id for unit in combatants}
        for key, cooldown in tuple(self.memory.squad_pairing_cooldowns.items()):
            if cooldown.expires_tick < tick or not set(key) <= living:
                self.memory.squad_pairing_cooldowns.pop(key, None)
        for key, expires_tick in tuple(self.memory.squad_target_backoffs.items()):
            if expires_tick < tick or key[0] not in living or key[1] not in living:
                self.memory.squad_target_backoffs.pop(key, None)
        for key, squad in tuple(self.memory.squad_states.items()):
            if squad.vanguard_id not in living or squad.ranger_id not in living:
                self.memory.squad_states.pop(key, None)
                self.memory.squad_formation_leases.pop(key, None)
                self.memory.squad_rendezvous_leases.pop(key, None)
        active_pairs = {
            (squad.vanguard_id, squad.ranger_id)
            for squad in self.memory.squad_states.values()
        }
        for key in tuple(self.memory.squad_rendezvous_leases):
            if key not in active_pairs or not set(key) <= living:
                self.memory.squad_rendezvous_leases.pop(key, None)
        for key in tuple(self.memory.squad_formation_leases):
            if key not in active_pairs or not set(key) <= living:
                self.memory.squad_formation_leases.pop(key, None)
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
            candidates = tuple(
                (
                    manhattan(v.position, r.position),
                    v.id.bytes,
                    r.id.bytes,
                    v,
                    r,
                )
                for v in vanguards
                for r in rangers
                if (
                    cooldown := self.memory.squad_pairing_cooldowns.get(
                        (v.id, r.id)
                    )
                ) is None
                or cooldown.expires_tick < tick
            )
            if not candidates:
                break
            _, _, _, vanguard, ranger = min(
                candidates,
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
