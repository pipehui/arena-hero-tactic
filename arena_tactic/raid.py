from __future__ import annotations

from math import ceil
from arena_hero import Direction, Position, UnitType

from .config import TacticConfig
from .geometry import (
    cardinal_neighbors,
    direction_between,
    manhattan,
    manhattan_ring,
    ranger_firing_positions,
    ranger_line_is_clear,
)
from .models import (
    ActionIntent,
    EnemyCoreIntel,
    EntitySnapshot,
    HomeCounterSiegeDecision,
    IntentAction,
    UnitMission,
    WorldModel,
)
from .planning import route_to
from .projection import TacticalMap
from .rules import UNIT_MAX_HP
from .state import TacticMemory


class RaidPlanner:
    """Persistent surplus-only enemy Core expedition state machine."""

    def __init__(self, config: TacticConfig, memory: TacticMemory) -> None:
        self.config = config
        self.memory = memory

    def intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        if world.core is None:
            self._clear()
            return []
        home_threat = self.memory.home_defense_alert_until >= world.tick or any(
            enemy.visible_now
            and enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.observed_position, world.core.position)
            <= self.config.home_warning_radius
            for enemy in projection.enemies
        )
        living = {unit.id: unit for unit in world.friendlies}
        members = tuple(
            living[member_id]
            for member_id in self.memory.raid_member_ids
            if member_id in living
        )
        target = self._active_target(world)

        if self.memory.raid_phase != "IDLE":
            interruption = self._interruption_reason(world, members, target, home_threat)
            if interruption is not None:
                self.memory.raid_phase = "RETURNING"
                self.memory.raid_interrupted_tick = world.tick
        if self.memory.raid_phase == "RETURNING":
            return self._return_intents(world, projection, members, protected)

        if self.memory.raid_phase == "IDLE":
            containment = self._containment_active(world)
            target = self._choose_target(world, projection, home_threat, containment)
            if target is None:
                return self._confirmation_intents(
                    world,
                    projection,
                    home_threat,
                )
            selected = self._select_members(world, target, containment=containment)
            if not selected:
                return []
            self.memory.raid_target_id = target.id
            self.memory.raid_last_seen_tick = target.last_seen_tick
            self.memory.raid_last_position = target.position
            self.memory.raid_member_ids = tuple(unit.id for unit in selected)
            self.memory.raid_phase = "ASSEMBLING"
            self.memory.raid_containment_mode = containment
            members = selected

        assert target is not None
        visible = next((core for core in world.enemy_cores if core.id == target.id), None)
        if visible is not None:
            self.memory.raid_last_seen_tick = world.tick
            self.memory.raid_last_position = visible.position
            target = self.memory.enemy_core_intel[target.id]

        if self.memory.raid_phase == "ASSEMBLING":
            if self._assembled(members):
                self.memory.raid_phase = "ADVANCING"
            else:
                return self._assemble_intents(world, projection, members, protected)
        if visible is not None:
            self.memory.raid_phase = "SIEGING"
            expected = visible.position
            if (
                visible.destination is not None
                and visible.move_progress is not None
                and visible.move_required_ticks is not None
                and visible.move_progress >= visible.move_required_ticks - 1
            ):
                expected = visible.destination
            return self._siege_intents(
                world,
                projection,
                members,
                expected,
                visible.id,
                protected,
            )
        if self.memory.raid_last_position is None:
            self.memory.raid_phase = "RETURNING"
            self.memory.raid_interrupted_tick = world.tick
            return self._return_intents(world, projection, members, protected)
        age = world.tick - (self.memory.raid_last_seen_tick or world.tick)
        if any(
            manhattan(unit.position, self.memory.raid_last_position) <= 2
            for unit in members
        ):
            self.memory.raid_phase = "SEARCHING"
            return self._search_intents(world, projection, members, age, protected)
        self.memory.raid_phase = "ADVANCING"
        return self._advance_intents(
            world,
            projection,
            members,
            self.memory.raid_last_position,
            protected,
        )

    def counter_siege_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        protected: frozenset[Position],
    ) -> tuple[HomeCounterSiegeDecision, list[ActionIntent]]:
        """Continue a local battle into the hostile Core that supplied it.

        This deliberately is not a normal expedition: the target has already
        entered the home-defense envelope, so the force-surplus launch gate
        would create a blind spot exactly when freshly spawned defenders are
        being cleared.  One healthy V+R pair remains at home and every other
        healthy combatant keeps pressure on the source Core.
        """

        if world.core is None:
            self._clear_counter_siege()
            return HomeCounterSiegeDecision(reason="CORE_UNAVAILABLE"), []
        visible = {
            core.id: core
            for core in world.enemy_cores
            if manhattan(core.position, world.core.position) <= 24
        }
        active = visible.get(self.memory.counter_siege_target_id)
        if active is None and self.memory.counter_siege_target_id is not None:
            age = world.tick - (
                self.memory.counter_siege_last_seen_tick or world.tick
            )
            if age > 4 or self.memory.counter_siege_last_position is None:
                self._clear_counter_siege()
            elif (
                manhattan(
                    self.memory.counter_siege_last_position,
                    world.core.position,
                )
                > 24
            ):
                self._clear_counter_siege()
        if self.memory.counter_siege_target_id is None:
            if self.memory.home_defense_alert_until < world.tick:
                return HomeCounterSiegeDecision(reason="NO_RECENT_HOME_BATTLE"), []
            candidates = tuple(
                sorted(
                    (
                        core
                        for core in visible.values()
                        if manhattan(core.position, world.core.position) <= 18
                    ),
                    key=lambda core: (
                        manhattan(core.position, world.core.position),
                        core.hp + core.shield,
                        core.id.bytes,
                    ),
                )
            )
            if not candidates:
                return HomeCounterSiegeDecision(reason="NO_LOCAL_ENEMY_CORE"), []
            active = candidates[0]
            self.memory.counter_siege_target_id = active.id
            self.memory.counter_siege_phase = "PRESSING"
        if active is not None:
            self.memory.counter_siege_last_seen_tick = world.tick
            self.memory.counter_siege_last_position = active.position
        target_id = self.memory.counter_siege_target_id
        target_position = (
            active.position
            if active is not None
            else self.memory.counter_siege_last_position
        )
        if target_id is None or target_position is None:
            self._clear_counter_siege()
            return HomeCounterSiegeDecision(reason="TARGET_LOST"), []

        healthy_vanguards = tuple(
            sorted(
                (
                    unit
                    for unit in world.friendlies
                    if unit.unit_type is UnitType.VANGUARD
                    and unit.hp * 2 > UNIT_MAX_HP[UnitType.VANGUARD]
                    and unit.id != world.beacon.carrier_id
                ),
                key=lambda unit: (
                    manhattan(unit.position, world.core.position),
                    unit.id.bytes,
                ),
            )
        )
        healthy_rangers = tuple(
            sorted(
                (
                    unit
                    for unit in world.friendlies
                    if unit.unit_type is UnitType.RANGER
                    and unit.hp * 2 > UNIT_MAX_HP[UnitType.RANGER]
                    and unit.id != world.beacon.carrier_id
                ),
                key=lambda unit: (
                    manhattan(unit.position, world.core.position),
                    unit.id.bytes,
                ),
            )
        )
        if not healthy_vanguards or not healthy_rangers:
            decision = HomeCounterSiegeDecision(
                phase="HOLDING",
                target_id=target_id,
                target_position=target_position,
                last_seen_tick=self.memory.counter_siege_last_seen_tick,
                reason="HOME_RESERVE_UNAVAILABLE",
            )
            return decision, []
        living_vanguard_ids = {unit.id for unit in healthy_vanguards}
        living_ranger_ids = {unit.id for unit in healthy_rangers}
        previous_reserve = tuple(
            item
            for item in self.memory.counter_siege_reserve_ids
            if item in living_vanguard_ids | living_ranger_ids
        )
        reserve_vanguard = next(
            (item for item in previous_reserve if item in living_vanguard_ids),
            healthy_vanguards[0].id,
        )
        reserve_ranger = next(
            (item for item in previous_reserve if item in living_ranger_ids),
            healthy_rangers[0].id,
        )
        reserve = (reserve_vanguard, reserve_ranger)
        self.memory.counter_siege_reserve_ids = reserve
        members = tuple(
            unit
            for unit in (*healthy_vanguards, *healthy_rangers)
            if unit.id not in reserve
        )
        if not members:
            decision = HomeCounterSiegeDecision(
                phase="HOLDING",
                target_id=target_id,
                target_position=target_position,
                reserve_ids=reserve,
                last_seen_tick=self.memory.counter_siege_last_seen_tick,
                reason="ONLY_HOME_PAIR_AVAILABLE",
            )
            return decision, []
        self.memory.counter_siege_member_ids = tuple(unit.id for unit in members)
        intents: list[ActionIntent] = []
        for unit in sorted(members, key=lambda item: item.id.bytes):
            intents.extend(
                self._counter_siege_unit_intents(
                    world,
                    projection,
                    unit,
                    target_id,
                    target_position,
                    protected,
                )
            )
        decision = HomeCounterSiegeDecision(
            phase="PRESSING",
            target_id=target_id,
            target_position=target_position,
            member_ids=tuple(unit.id for unit in members),
            reserve_ids=reserve,
            last_seen_tick=self.memory.counter_siege_last_seen_tick,
            reason="LOCAL_THREAT_SOURCE",
        )
        return decision, intents

    def _counter_siege_unit_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        unit: EntitySnapshot,
        target_id,
        target: Position,
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        if unit.unit_type is UnitType.RANGER and ranger_line_is_clear(
            unit.position, target, world.known_obstacles
        ):
            return [
                ActionIntent.simple(
                    unit.id,
                    IntentAction.SHOOT,
                    UnitMission.COUNTER_SIEGE,
                    37,
                    target_id=target_id,
                    expected_cell=target,
                    target_position=target,
                    reason="COUNTER_SIEGE_CORE_FIRE",
                )
            ]
        if unit.unit_type is UnitType.VANGUARD and manhattan(unit.position, target) == 1:
            direction = direction_between(unit.position, target)
            if direction is not None:
                return [
                    ActionIntent(
                        actor_id=unit.id,
                        action=IntentAction.SWEEP,
                        mission=UnitMission.COUNTER_SIEGE,
                        priority=37,
                        direction=direction,
                        target_id=target_id,
                        target_position=target,
                        reason="COUNTER_SIEGE_CORE_SWEEP",
                    )
                ]
        destinations = (
            tuple(
                cell
                for cell in ranger_firing_positions(target)
                if ranger_line_is_clear(cell, target, world.known_obstacles)
            )
            if unit.unit_type is UnitType.RANGER
            else tuple(cell for _, cell in cardinal_neighbors(target))
        )
        routes = []
        for destination in destinations:
            if (
                destination not in world.known_passable
                or destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or destination in protected
            ):
                continue
            route = route_to(
                world,
                unit.position,
                destination,
                node_limit=self.config.path_node_limit,
                blocked=frozenset(
                    (projection.hostile_occupied | protected)
                    - {unit.position, destination}
                ),
            )
            if route is not None and route.first_direction is not None:
                routes.append((route.distance, destination, route))
        if not routes:
            return [
                ActionIntent.simple(
                    unit.id,
                    IntentAction.WAIT,
                    UnitMission.COUNTER_SIEGE,
                    53,
                    target_id=target_id,
                    target_position=target,
                    reason="COUNTER_SIEGE_NO_REACHABLE_POSITION",
                )
            ]
        _, destination, route = min(routes)
        return [
            ActionIntent.move(
                unit.id,
                UnitMission.COUNTER_SIEGE,
                52,
                route.first_direction,
                route.first_position,
                risk=projection.future_attackers(route.first_position) * 10,
                exclusive_destination=True,
                tie_break=(route.distance,),
                reason="COUNTER_SIEGE_ADVANCE",
                metadata=(("target_id", str(target_id)),),
            )
        ]

    def _clear_counter_siege(self) -> None:
        self.memory.counter_siege_target_id = None
        self.memory.counter_siege_last_seen_tick = None
        self.memory.counter_siege_last_position = None
        self.memory.counter_siege_member_ids = ()
        self.memory.counter_siege_reserve_ids = ()
        self.memory.counter_siege_phase = "IDLE"

    def _choose_target(
        self,
        world: WorldModel,
        projection: TacticalMap,
        home_threat: bool,
        containment: bool,
    ) -> EnemyCoreIntel | None:
        if home_threat or world.core is None:
            return None
        visible_ids = {
            core.enemy_id for core in projection.enemy_cores if core.visible_now
        }
        candidates = [
            intel
            for intel in self.memory.enemy_core_intel.values()
            if intel.id in visible_ids
            and (
                manhattan(intel.position, world.core.position)
                <= self.config.raid_start_radius
                or (
                    manhattan(intel.position, world.core.position)
                    <= self.config.raid_confirmed_start_radius
                    and intel.sighting_count >= self.config.raid_confirmed_sightings
                )
                or (
                    containment
                    and manhattan(intel.position, world.core.position)
                    <= self.config.raid_containment_radius
                    and intel.sighting_count >= self.config.raid_confirmed_sightings
                )
            )
            and (
                self.memory.raid_interrupted_tick is None
                or intel.last_seen_tick > self.memory.raid_interrupted_tick
            )
        ]
        return min(
            candidates,
            key=lambda intel: (
                self._visible_guards(world, intel.position),
                intel.hp + intel.shield,
                manhattan(intel.position, world.core.position),
                intel.id.bytes,
            ),
            default=None,
        )

    def _confirmation_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        home_threat: bool,
    ) -> list[ActionIntent]:
        """Hold a safe empty Worker for one distant-Core confirmation Tick."""

        if home_threat or world.core is None:
            return []
        containment = self._containment_active(world)
        confirmation_radius = (
            self.config.raid_containment_radius
            if containment
            else self.config.raid_confirmed_start_radius
        )
        visible_ids = {
            core.enemy_id for core in projection.enemy_cores if core.visible_now
        }
        candidate = min(
            (
                intel
                for intel in self.memory.enemy_core_intel.values()
                if intel.id in visible_ids
                and self.config.raid_start_radius
                < manhattan(intel.position, world.core.position)
                <= confirmation_radius
                and intel.sighting_count < self.config.raid_confirmed_sightings
                and self._visible_guards(world, intel.position) == 0
                and self._select_members(world, intel, containment=containment)
            ),
            key=lambda intel: (
                manhattan(intel.position, world.core.position),
                intel.id.bytes,
            ),
            default=None,
        )
        if candidate is None:
            return []
        observer = min(
            (
                unit
                for unit in world.friendlies
                if unit.unit_type is UnitType.WORKER
                and unit.cargo == 0
                and unit.hp >= UNIT_MAX_HP[UnitType.WORKER]
                and manhattan(unit.position, candidate.position) <= 3
                and projection.immediate_attackers(unit.position) == 0
                and projection.future_attackers(unit.position) < unit.hp
            ),
            key=lambda unit: (
                manhattan(unit.position, candidate.position),
                unit.id.bytes,
            ),
            default=None,
        )
        if observer is None:
            return []
        return [
            ActionIntent.simple(
                observer.id,
                IntentAction.WAIT,
                UnitMission.RAID,
                45,
                target_id=candidate.id,
                target_position=candidate.position,
                reason="RAID_TARGET_CONFIRMATION",
            )
        ]

    def _select_members(
        self,
        world: WorldModel,
        target: EnemyCoreIntel,
        *,
        containment: bool = False,
    ) -> tuple[EntitySnapshot, ...]:
        healthy = tuple(
            unit
            for unit in world.friendlies
            if unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and unit.hp * 2 > UNIT_MAX_HP[unit.unit_type]
            and unit.id != world.beacon.carrier_id
            and world.core is not None
            and manhattan(unit.position, world.core.position)
            <= self.config.home_pursuit_radius
        )
        home_target = max(self.config.home_force_floor, self.memory.home_force_high_water)
        home_reserve = (
            min(home_target, self.config.raid_peace_home_reserve)
            if containment
            else home_target
        )
        surplus = max(0, len(healthy) - home_reserve)
        guards = self._visible_guards(world, target.position)
        required = max(
            self.config.raid_min_siege_members if containment else 2,
            guards + self.config.raid_force_margin,
        )
        if surplus < required:
            return ()
        vanguards = sorted(
            (unit for unit in healthy if unit.unit_type is UnitType.VANGUARD),
            key=lambda unit: (manhattan(unit.position, target.position), unit.id.bytes),
        )
        rangers = sorted(
            (unit for unit in healthy if unit.unit_type is UnitType.RANGER),
            key=lambda unit: (manhattan(unit.position, target.position), unit.id.bytes),
        )
        if not vanguards or not rangers:
            return ()
        selected = [vanguards.pop(0), rangers.pop(0)]
        selected_vanguards = 1
        selected_rangers = 1
        while len(selected) < required and (vanguards or rangers):
            if not vanguards:
                chosen = rangers.pop(0)
                selected_rangers += 1
            elif not rangers:
                chosen = vanguards.pop(0)
                selected_vanguards += 1
            elif selected_vanguards < selected_rangers:
                chosen = vanguards.pop(0)
                selected_vanguards += 1
            elif selected_rangers < selected_vanguards:
                chosen = rangers.pop(0)
                selected_rangers += 1
            else:
                vanguard_key = (
                    manhattan(vanguards[0].position, target.position),
                    vanguards[0].id.bytes,
                )
                ranger_key = (
                    manhattan(rangers[0].position, target.position),
                    rangers[0].id.bytes,
                )
                if vanguard_key <= ranger_key:
                    chosen = vanguards.pop(0)
                    selected_vanguards += 1
                else:
                    chosen = rangers.pop(0)
                    selected_rangers += 1
            selected.append(chosen)
        return tuple(selected)

    def _interruption_reason(self, world, members, target, home_threat):
        if home_threat:
            return "HOME_THREAT"
        if target is None:
            return "TARGET_INTEL_EXPIRED"
        continue_radius = (
            self.config.raid_containment_continue_radius
            if self.memory.raid_containment_mode
            else self.config.raid_continue_radius
        )
        if world.core is None or manhattan(target.position, world.core.position) > continue_radius:
            return "TARGET_TOO_FAR"
        if len(members) != len(self.memory.raid_member_ids):
            return "MEMBER_LOST"
        if any(unit.hp * 2 <= UNIT_MAX_HP[unit.unit_type] for unit in members):
            return "MEMBER_LOW_HP"
        visible = any(core.id == target.id for core in world.enemy_cores)
        if visible and self._visible_guards(world, target.position) + self.config.raid_force_margin > len(members):
            return "ENEMY_REINFORCED"
        return None

    def _containment_active(self, world: WorldModel) -> bool:
        if world.core is None:
            return False
        confirmed = sum(
            world.tick - intel.last_seen_tick <= self.config.raid_intel_ttl
            and intel.sighting_count >= self.config.raid_confirmed_sightings
            and manhattan(intel.position, world.core.position)
            <= self.config.raid_containment_radius
            for intel in self.memory.enemy_core_intel.values()
        )
        return confirmed >= self.config.raid_containment_core_count

    def _active_target(self, world: WorldModel) -> EnemyCoreIntel | None:
        if self.memory.raid_target_id is None:
            return None
        return self.memory.enemy_core_intel.get(self.memory.raid_target_id)

    @staticmethod
    def _assembled(members: tuple[EntitySnapshot, ...]) -> bool:
        return all(
            manhattan(left.position, right.position) <= 4
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        )

    def _assemble_intents(self, world, projection, members, protected):
        if not members:
            return []
        rendezvous = min(
            (unit.position for unit in members),
            key=lambda cell: sum(manhattan(cell, unit.position) for unit in members),
        )
        intents: list[ActionIntent] = []
        for unit in members:
            if unit.position == rendezvous or manhattan(unit.position, rendezvous) <= 1:
                intents.append(self._wait(unit, "RAID_ASSEMBLY_HOLD"))
            else:
                intents.extend(self._move(world, projection, unit, rendezvous, protected, "RAID_ASSEMBLE"))
        return intents

    def _advance_intents(self, world, projection, members, target, protected):
        intents: list[ActionIntent] = []
        ordered = sorted(members, key=lambda unit: (-manhattan(unit.position, target), unit.id.bytes))
        for unit in ordered:
            if any(
                manhattan(unit.position, other.position) > self.config.squad_max_separation
                and manhattan(unit.position, target) < manhattan(other.position, target)
                for other in members
            ):
                intents.append(self._wait(unit, "RAID_FORMATION_HOLD"))
                continue
            intents.extend(self._move(world, projection, unit, target, protected, "RAID_ADVANCE"))
        return intents

    def _siege_intents(self, world, projection, members, target, target_id, protected):
        intents: list[ActionIntent] = []
        for unit in members:
            if unit.unit_type is UnitType.RANGER and ranger_line_is_clear(
                unit.position, target, world.known_obstacles
            ):
                intents.append(
                    ActionIntent.simple(
                        unit.id,
                        IntentAction.SHOOT,
                        UnitMission.RAID,
                        36,
                        target_id=target_id,
                        expected_cell=target,
                        target_position=target,
                        reason="RAID_CORE_FIRE",
                    )
                )
                continue
            if unit.unit_type is UnitType.VANGUARD and manhattan(unit.position, target) == 1:
                direction = direction_between(unit.position, target)
                if direction is not None:
                    intents.append(
                        ActionIntent(
                            actor_id=unit.id,
                            action=IntentAction.SWEEP,
                            mission=UnitMission.RAID,
                            priority=36,
                            direction=direction,
                            target_id=target_id,
                            target_position=target,
                            reason="RAID_CORE_SWEEP",
                        )
                    )
                    continue
            destinations = (
                tuple(
                    cell
                    for cell in ranger_firing_positions(target)
                    if ranger_line_is_clear(cell, target, world.known_obstacles)
                )
                if unit.unit_type is UnitType.RANGER
                else tuple(cell for _, cell in cardinal_neighbors(target))
            )
            destination = min(
                (
                    cell
                    for cell in destinations
                    if cell in world.known_passable
                    and cell not in world.known_obstacles
                    and cell not in projection.hostile_occupied
                    and cell not in protected
                ),
                key=lambda cell: (manhattan(unit.position, cell), cell),
                default=None,
            )
            if destination is not None:
                intents.extend(self._move(world, projection, unit, destination, protected, "RAID_SIEGE_POSITION"))
        return intents

    def _search_intents(self, world, projection, members, age, protected):
        assert self.memory.raid_last_position is not None
        radius = min(8, max(1, ceil(age / 4)))
        ring = tuple(
            cell
            for cell in manhattan_ring(self.memory.raid_last_position, radius)
            if cell in world.known_passable
            and cell not in world.known_obstacles
            and cell not in projection.hostile_occupied
        )
        intents: list[ActionIntent] = []
        visible_ticks = dict(world.cell_last_visible)
        ordered_members = tuple(sorted(members, key=lambda item: item.id.bytes))
        for index, unit in enumerate(ordered_members):
            if not ring:
                intents.append(self._wait(unit, "RAID_SEARCH_NO_CELL"))
                continue
            start = index * len(ring) // len(ordered_members)
            end = (index + 1) * len(ring) // len(ordered_members)
            sector = ring[start:end] or ring
            target = min(
                sector,
                key=lambda cell: (
                    visible_ticks.get(cell, -1),
                    self.memory.visit_counts.get(cell, 0),
                    manhattan(unit.position, cell),
                    cell,
                ),
            )
            intents.extend(self._move(world, projection, unit, target, protected, "RAID_SEARCH"))
        return intents

    def _return_intents(self, world, projection, members, protected):
        assert world.core is not None
        intents: list[ActionIntent] = []
        returned = 0
        for unit in members:
            if manhattan(unit.position, world.core.position) <= self.config.home_engage_radius:
                returned += 1
                intents.append(self._wait(unit, "RAID_RETURN_HOLD"))
                continue
            intents.extend(self._move(world, projection, unit, world.core.position, protected, "RAID_RETURN"))
        if returned == len(members):
            self._clear()
        return intents

    def _move(self, world, projection, unit, target, protected, reason):
        blocked = (projection.hostile_occupied | protected) - {unit.position, target}
        route = route_to(
            world,
            unit.position,
            target,
            node_limit=self.config.path_node_limit,
            blocked=frozenset(blocked),
        )
        if route is None or route.first_direction is None:
            return [self._wait(unit, f"{reason}_BLOCKED")]
        immediate, future, remembered = projection.exposure(route.first_position)
        return [
            ActionIntent.move(
                unit.id,
                UnitMission.RAID,
                60,
                route.first_direction,
                route.first_position,
                risk=immediate * 100 + future * 10 + remembered,
                exclusive_destination=True,
                tie_break=(route.distance,),
                reason=reason,
            ),
            self._wait(unit, f"{reason}_BLOCKED_THIS_TICK"),
        ]

    @staticmethod
    def _wait(unit: EntitySnapshot, reason: str) -> ActionIntent:
        return ActionIntent.simple(
            unit.id,
            IntentAction.WAIT,
            UnitMission.RAID,
            61,
            reason=reason,
        )

    @staticmethod
    def _visible_guards(world: WorldModel, position: Position) -> int:
        return sum(
            enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.position, position) <= 8
            for enemy in world.enemies
        )

    def _clear(self) -> None:
        self.memory.raid_target_id = None
        self.memory.raid_last_seen_tick = None
        self.memory.raid_last_position = None
        self.memory.raid_member_ids = ()
        self.memory.raid_phase = "IDLE"
        self.memory.raid_containment_mode = False
