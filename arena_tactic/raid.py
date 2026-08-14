from __future__ import annotations

from dataclasses import replace
from math import ceil
from uuid import UUID
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
    RaidAttemptMemory,
    UnitMission,
    WorldModel,
    LongRangeRaidCampaign,
    RaidConfirmationLease,
    RaidDistanceBand,
    RaidReconMission,
    SiegeApproachPlan,
)
from .planning import move_viability, route_to, siege_approach_plan
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
        if target is not None and self.memory.raid_phase not in {"IDLE", "RETURNING"}:
            self.memory.raid_distance_band = self._distance_band(world, target)
            if (
                self.memory.raid_siege_approach is None
                or self.memory.raid_siege_approach.target_id != target.id
                or self.memory.raid_siege_approach.target_position != target.position
            ):
                self.memory.raid_siege_approach = self._siege_approach(world, target)

        if self.memory.raid_phase == "RECON":
            recon_result = self._recon_active_intents(
                world,
                projection,
                members,
                target,
                home_threat,
                protected,
            )
            if recon_result is not None:
                return recon_result
            # A current two-Tick confirmation promotes the reconnaissance
            # contact into the normal 2V+2R launch path below.
            members = ()
            target = None

        if self.memory.raid_phase not in {"IDLE", "RETURNING"}:
            self._update_long_range_progress(world, members)
            interruption = self._interruption_reason(world, members, target, home_threat)
            if interruption is not None:
                if target is not None and interruption in {
                    "MEMBER_LOW_HP",
                    "MEMBER_LOST",
                    "ENEMY_REINFORCED",
                    "LONG_RANGE_NO_PROGRESS",
                    "LONG_RANGE_CAMPAIGN_EXPIRED",
                }:
                    self._record_failed_attempt(target, interruption, world.tick)
                self.memory.raid_phase = "RETURNING"
                self.memory.raid_interrupted_tick = world.tick
                self.memory.raid_return_reason = interruption
        if self.memory.raid_phase == "RETURNING":
            return self._return_intents(world, projection, members, protected)

        if self.memory.raid_phase == "IDLE":
            containment = self._containment_active(world)
            target = self._choose_target(world, projection, home_threat, containment)
            if target is None:
                confirmation = self._confirmation_intents(
                    world,
                    projection,
                    home_threat,
                )
                if confirmation:
                    return confirmation
                return self._start_recon_intents(
                    world,
                    projection,
                    home_threat,
                    protected,
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
            self.memory.raid_distance_band = self._distance_band(world, target)
            self.memory.raid_siege_approach = self._siege_approach(world, target)
            self.memory.raid_return_reason = None
            self.memory.raid_confirmation_lease = None
            self.memory.raid_recon_mission = None
            self.memory.raid_handoff_targets.clear()
            if self.memory.raid_distance_band is RaidDistanceBand.LONG_RANGE:
                route_eta = self._known_route_eta(world, target)
                if route_eta is None:
                    self._clear()
                    return []
                duration = min(
                    self.config.raid_long_range_max_campaign_ticks,
                    max(
                        64,
                        route_eta + self.config.raid_long_range_search_reserve_ticks,
                    ),
                )
                self.memory.raid_long_range_campaign = LongRangeRaidCampaign(
                    target_id=target.id,
                    member_ids=tuple(unit.id for unit in selected),
                    phase="ASSEMBLING",
                    started_tick=world.tick,
                    route_eta=route_eta,
                    search_deadline_tick=world.tick + duration,
                    last_position=target.position,
                    last_group_distance=sum(
                        manhattan(unit.position, target.position) for unit in selected
                    ),
                )
            members = selected

        assert target is not None
        visible = next((core for core in world.enemy_cores if core.id == target.id), None)
        if visible is not None:
            self.memory.raid_last_seen_tick = world.tick
            self.memory.raid_last_position = visible.position
            target = self.memory.enemy_core_intel[target.id]
            self.memory.raid_siege_approach = self._siege_approach(world, target)
            if self.memory.raid_long_range_campaign is not None:
                self.memory.raid_long_range_campaign = replace(
                    self.memory.raid_long_range_campaign,
                    last_position=visible.position,
                )

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
                or (
                    self.config.raid_confirmed_start_radius
                    < manhattan(intel.position, world.core.position)
                    <= self.config.raid_containment_radius
                    and intel.sighting_count >= self.config.raid_confirmed_sightings
                    and self._known_route_eta(world, intel) is not None
                )
                or (
                    self.config.raid_containment_radius
                    < manhattan(intel.position, world.core.position)
                    <= self.config.raid_long_range_start_radius
                    and intel.sighting_count >= self.config.raid_confirmed_sightings
                    and self._known_route_eta(world, intel) is not None
                )
            )
            and (
                self.memory.raid_interrupted_tick is None
                or intel.last_seen_tick > self.memory.raid_interrupted_tick
            )
            and (
                (attempt := self.memory.raid_attempts.get(intel.id)) is None
                or attempt.last_failure_sighting_tick is None
                or intel.last_seen_tick > attempt.last_failure_sighting_tick
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
        confirmation_radius = self.config.raid_long_range_start_radius
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
                and self._select_members(
                    world,
                    intel,
                    containment=containment,
                    long_range=self._is_long_range_target(world, intel),
                )
                and (
                    not self._is_long_range_target(world, intel)
                    or self._known_route_eta(world, intel) is not None
                )
            ),
            key=lambda intel: (
                manhattan(intel.position, world.core.position),
                intel.id.bytes,
            ),
            default=None,
        )
        if candidate is None:
            lease = self.memory.raid_confirmation_lease
            if lease is not None and lease.expires_tick < world.tick:
                self.memory.raid_confirmation_lease = None
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
        self.memory.raid_confirmation_lease = RaidConfirmationLease(
            target_id=candidate.id,
            observer_id=observer.id,
            first_seen_tick=(
                candidate.confirmation_window_start_tick
                or candidate.last_seen_tick
            ),
            expires_tick=(
                (candidate.confirmation_window_start_tick or candidate.last_seen_tick)
                + self.config.raid_confirmation_window_ticks
            ),
        )
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
        long_range: bool | None = None,
    ) -> tuple[EntitySnapshot, ...]:
        if long_range is None:
            long_range = self._is_long_range_target(world, target)
        distance = (
            0
            if world.core is None
            else manhattan(target.position, world.core.position)
        )
        remote = distance > self.config.raid_confirmed_start_radius
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
            home_target
            if remote
            else min(home_target, self.config.raid_peace_home_reserve)
        )
        surplus = max(0, len(healthy) - home_reserve)
        guards = self._visible_guards(world, target.position)
        attempt = self.memory.raid_attempts.get(target.id)
        required_pairs = (
            self.config.raid_initial_pair_count
            + (0 if attempt is None else attempt.failed_attempts)
            * self.config.raid_escalation_pair_step
        )
        required = max(
            required_pairs * 2,
            self.config.raid_long_range_min_members if long_range else 0,
            self.config.raid_min_siege_members if containment else 0,
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
        if len(vanguards) < required_pairs or len(rangers) < required_pairs:
            return ()
        selected = [
            *(vanguards.pop(0) for _ in range(required_pairs)),
            *(rangers.pop(0) for _ in range(required_pairs)),
        ]
        selected_vanguards = required_pairs
        selected_rangers = required_pairs
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
        if remote:
            if selected_vanguards < required_pairs or selected_rangers < required_pairs:
                return ()
            remaining_vanguards = sum(
                unit.unit_type is UnitType.VANGUARD for unit in healthy
            ) - selected_vanguards
            remaining_rangers = sum(
                unit.unit_type is UnitType.RANGER for unit in healthy
            ) - selected_rangers
            if (
                remaining_vanguards < self.config.minimum_vanguards
                or remaining_rangers < self.config.minimum_rangers
                or len(healthy) - len(selected) < home_target
            ):
                return ()
        return tuple(selected)

    def _start_recon_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        home_threat: bool,
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        """Send only one mixed pair to refresh stale strategic Core intel."""

        if home_threat or world.core is None:
            return []
        visible_ids = {core.id for core in world.enemy_cores}
        candidates = []
        for intel in self.memory.enemy_core_intel.values():
            distance = manhattan(intel.position, world.core.position)
            age = world.tick - intel.last_seen_tick
            if (
                not self.config.raid_start_radius < distance
                <= self.config.raid_long_range_start_radius
                or age > self.config.enemy_core_control_ttl
                or (
                    intel.id in visible_ids
                    and intel.confirmation_sightings
                    >= self.config.raid_confirmed_sightings
                )
            ):
                continue
            approach = self._siege_approach(world, intel)
            if approach is None:
                continue
            candidates.append((age, distance, intel.id.bytes, intel, approach))
        if not candidates:
            return []
        _, _, _, target, approach = min(candidates)
        members = self._select_recon_members(world, target)
        if not members:
            return []
        self.memory.raid_target_id = target.id
        self.memory.raid_last_seen_tick = target.last_seen_tick
        self.memory.raid_last_position = target.position
        self.memory.raid_member_ids = tuple(unit.id for unit in members)
        self.memory.raid_phase = "RECON"
        self.memory.raid_distance_band = approach.distance_band
        self.memory.raid_siege_approach = approach
        self.memory.raid_recon_mission = RaidReconMission(
            target_id=target.id,
            member_ids=tuple(unit.id for unit in members),
            last_position=target.position,
            started_tick=world.tick,
            last_seen_tick=target.last_seen_tick,
            last_group_distance=sum(
                manhattan(unit.position, target.position) for unit in members
            ),
        )
        return self._recon_move_intents(
            world,
            projection,
            members,
            target.position,
            protected,
        )

    def _select_recon_members(
        self,
        world: WorldModel,
        target: EnemyCoreIntel,
    ) -> tuple[EntitySnapshot, ...]:
        assert world.core is not None
        healthy = tuple(
            unit
            for unit in world.friendlies
            if unit.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and unit.hp * 2 > UNIT_MAX_HP[unit.unit_type]
            and unit.id != world.beacon.carrier_id
            and manhattan(unit.position, world.core.position)
            <= self.config.home_pursuit_radius
        )
        home_target = max(self.config.home_force_floor, self.memory.home_force_high_water)
        vanguards = sorted(
            (unit for unit in healthy if unit.unit_type is UnitType.VANGUARD),
            key=lambda unit: (manhattan(unit.position, target.position), unit.id.bytes),
        )
        rangers = sorted(
            (unit for unit in healthy if unit.unit_type is UnitType.RANGER),
            key=lambda unit: (manhattan(unit.position, target.position), unit.id.bytes),
        )
        if (
            len(healthy) - 2 < home_target
            or len(vanguards) <= self.config.minimum_vanguards
            or len(rangers) <= self.config.minimum_rangers
        ):
            return ()
        return vanguards[0], rangers[0]

    def _recon_active_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        members: tuple[EntitySnapshot, ...],
        target: EnemyCoreIntel | None,
        home_threat: bool,
        protected: frozenset[Position],
    ) -> list[ActionIntent] | None:
        mission = self.memory.raid_recon_mission
        if mission is None:
            self.memory.raid_phase = "IDLE"
            self.memory.raid_member_ids = ()
            return None
        visible = target is not None and any(
            core.id == target.id for core in world.enemy_cores
        )
        if (
            visible
            and target is not None
            and target.confirmation_sightings >= self.config.raid_confirmed_sightings
        ):
            self.memory.raid_phase = "IDLE"
            self.memory.raid_member_ids = ()
            self.memory.raid_recon_mission = None
            return None
        if (
            home_threat
            or target is None
            or len(members) != len(mission.member_ids)
            or any(unit.hp * 2 <= UNIT_MAX_HP[unit.unit_type] for unit in members)
            or mission.no_progress_ticks >= 4
        ):
            self.memory.raid_phase = "RETURNING"
            self.memory.raid_return_reason = (
                "RECON_HOME_THREAT"
                if home_threat
                else "RECON_TARGET_CLEARED"
                if target is None
                else "RECON_MEMBER_UNAVAILABLE"
                if len(members) != len(mission.member_ids)
                else "RECON_MEMBER_LOW_HP"
                if any(unit.hp * 2 <= UNIT_MAX_HP[unit.unit_type] for unit in members)
                else "RECON_NO_PROGRESS"
            )
            return self._return_intents(world, projection, members, protected)
        group_distance = sum(
            manhattan(unit.position, target.position) for unit in members
        )
        no_progress = (
            0
            if mission.last_group_distance is None
            or group_distance < mission.last_group_distance
            else mission.no_progress_ticks + 1
        )
        self.memory.raid_recon_mission = replace(
            mission,
            last_position=target.position,
            last_seen_tick=target.last_seen_tick,
            no_progress_ticks=no_progress,
            last_group_distance=group_distance,
        )
        return self._recon_move_intents(
            world,
            projection,
            members,
            target.position,
            protected,
        )

    def _recon_move_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        members: tuple[EntitySnapshot, ...],
        target: Position,
        protected: frozenset[Position],
    ) -> list[ActionIntent]:
        intents: list[ActionIntent] = []
        blocked = frozenset(enemy.position for enemy in world.enemies) | {target}
        for unit in members:
            radius = 4 if unit.unit_type is UnitType.VANGUARD else 5
            rows = []
            for cell in manhattan_ring(target, radius):
                if (
                    cell not in world.known_passable
                    or cell in world.known_obstacles
                    or cell in blocked
                    or cell in protected
                ):
                    continue
                route = route_to(
                    world,
                    unit.position,
                    cell,
                    node_limit=self.config.path_node_limit,
                    blocked=blocked - {unit.position, cell},
                )
                if route is not None:
                    rows.append((route.distance, cell))
            destination = min(rows, default=(0, None))[1]
            if destination is None:
                intents.append(self._wait(unit, "RAID_RECON_NO_OBSERVATION_ROUTE"))
            elif destination == unit.position:
                intents.append(self._wait(unit, "RAID_RECON_OBSERVE"))
            else:
                intents.extend(
                    self._move(
                        world,
                        projection,
                        unit,
                        destination,
                        protected,
                        "RAID_RECON_ADVANCE",
                    )
                )
        return intents

    def _record_failed_attempt(
        self,
        target: EnemyCoreIntel,
        reason: str,
        tick: int,
    ) -> None:
        previous = self.memory.raid_attempts.get(target.id)
        self.memory.raid_attempts[target.id] = RaidAttemptMemory(
            core_id=target.id,
            failed_attempts=(0 if previous is None else previous.failed_attempts) + 1,
            last_failure_tick=tick,
            last_failure_reason=reason,
            last_failure_sighting_tick=target.last_seen_tick,
        )

    def _interruption_reason(self, world, members, target, home_threat):
        if home_threat:
            return "HOME_THREAT"
        if target is None:
            return "TARGET_INTEL_EXPIRED"
        campaign = self.memory.raid_long_range_campaign
        band = self.memory.raid_distance_band
        continue_radius = (
            self.config.raid_long_range_continue_radius
            if band is RaidDistanceBand.LONG_RANGE or campaign is not None
            else self.config.raid_containment_continue_radius
            if band is RaidDistanceBand.EXTENDED or self.memory.raid_containment_mode
            else self.config.raid_continue_radius
        )
        if world.core is None or manhattan(target.position, world.core.position) > continue_radius:
            return "TARGET_TOO_FAR"
        if len(members) != len(self.memory.raid_member_ids):
            return "MEMBER_LOST"
        if any(unit.hp * 2 <= UNIT_MAX_HP[unit.unit_type] for unit in members):
            return "MEMBER_LOW_HP"
        if campaign is not None:
            if world.tick > campaign.search_deadline_tick:
                return "LONG_RANGE_CAMPAIGN_EXPIRED"
            if campaign.no_progress_ticks >= 4:
                return "LONG_RANGE_NO_PROGRESS"
        visible = any(core.id == target.id for core in world.enemy_cores)
        if visible and self._visible_guards(world, target.position) + self.config.raid_force_margin > len(members):
            return "ENEMY_REINFORCED"
        return None

    def _is_long_range_target(
        self,
        world: WorldModel,
        target: EnemyCoreIntel,
    ) -> bool:
        return bool(
            world.core is not None
            and self.config.raid_containment_radius
            < manhattan(target.position, world.core.position)
            <= self.config.raid_long_range_start_radius
        )

    def _distance_band(
        self,
        world: WorldModel,
        target: EnemyCoreIntel,
    ) -> RaidDistanceBand:
        assert world.core is not None
        distance = manhattan(target.position, world.core.position)
        if distance <= self.config.raid_confirmed_start_radius:
            return RaidDistanceBand.NEAR
        if distance <= self.config.raid_containment_radius:
            return RaidDistanceBand.EXTENDED
        return RaidDistanceBand.LONG_RANGE

    def _siege_approach(
        self,
        world: WorldModel,
        target: EnemyCoreIntel,
    ) -> SiegeApproachPlan | None:
        return siege_approach_plan(
            world,
            target.id,
            target.position,
            band=self._distance_band(world, target),
            node_limit=self.config.path_node_limit,
            max_route=self.config.raid_long_range_max_route,
        )

    def _known_route_eta(
        self,
        world: WorldModel,
        target: EnemyCoreIntel,
    ) -> int | None:
        approach = self._siege_approach(world, target)
        return None if approach is None else approach.route_eta

    def _update_long_range_progress(
        self,
        world: WorldModel,
        members: tuple[EntitySnapshot, ...],
    ) -> None:
        campaign = self.memory.raid_long_range_campaign
        if campaign is None or not members:
            return
        target = self.memory.raid_last_position or campaign.last_position
        distance = sum(manhattan(unit.position, target) for unit in members)
        track_progress = self.memory.raid_phase in {"ADVANCING", "SEARCHING"}
        no_progress = (
            0
            if not track_progress
            or campaign.last_group_distance is None
            or distance < campaign.last_group_distance
            else campaign.no_progress_ticks + 1
        )
        self.memory.raid_long_range_campaign = replace(
            campaign,
            phase=self.memory.raid_phase,
            member_ids=tuple(unit.id for unit in members),
            last_position=target,
            last_group_distance=distance,
            no_progress_ticks=no_progress,
        )

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
                campaign = self.memory.raid_long_range_campaign
                if campaign is not None and campaign.no_progress_ticks >= 2:
                    laggard = max(
                        members,
                        key=lambda other: (
                            manhattan(other.position, target),
                            other.id.bytes,
                        ),
                    )
                    intents.extend(
                        self._move(
                            world,
                            projection,
                            unit,
                            laggard.position,
                            protected,
                            "RAID_FORMATION_REJOIN",
                        )
                    )
                else:
                    intents.append(self._wait(unit, "RAID_FORMATION_HOLD"))
                continue
            approach = self.memory.raid_siege_approach
            destinations = (
                ()
                if approach is None
                else approach.ranger_positions
                if unit.unit_type is UnitType.RANGER
                else approach.vanguard_positions
            )
            advance_target = min(
                destinations,
                key=lambda cell: (manhattan(unit.position, cell), cell),
                default=target,
            )
            intents.extend(
                self._move(
                    world,
                    projection,
                    unit,
                    advance_target,
                    protected,
                    "RAID_ADVANCE",
                )
            )
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
        continuing: list[UUID] = []
        living_ids = {unit.id for unit in members}
        for member_id in tuple(self.memory.raid_handoff_targets):
            if member_id not in living_ids:
                self.memory.raid_handoff_targets.pop(member_id, None)
        reserved_handoffs: set[Position] = set(
            self.memory.raid_handoff_targets.values()
        )
        for unit in sorted(members, key=lambda item: item.id.bytes):
            # A casualty is no longer formation-dependent.  The unified Core
            # service planner immediately owns its RECOVER trip, so a healthy
            # expedition partner can never pin it at the home boundary.
            if unit.hp < UNIT_MAX_HP[unit.unit_type]:
                self.memory.raid_handoff_targets.pop(unit.id, None)
                continue
            handoff = self.memory.raid_handoff_targets.get(unit.id)
            if handoff is None or not self._handoff_valid(
                world,
                projection,
                handoff,
                protected,
                reserved_handoffs - {handoff},
            ):
                if handoff is not None:
                    reserved_handoffs.discard(handoff)
                handoff = self._choose_handoff(
                    world,
                    projection,
                    unit,
                    protected,
                    reserved_handoffs,
                )
                if handoff is not None:
                    self.memory.raid_handoff_targets[unit.id] = handoff
                    reserved_handoffs.add(handoff)
            if handoff is not None and unit.position == handoff:
                self.memory.raid_handoff_targets.pop(unit.id, None)
                continue
            continuing.append(unit.id)
            if handoff is None:
                intents.append(self._wait(unit, "RAID_RETURN_NO_HANDOFF"))
            else:
                intents.extend(
                    self._move(
                        world,
                        projection,
                        unit,
                        handoff,
                        protected,
                        "RAID_RETURN_HANDOFF",
                    )
                )
        self.memory.raid_member_ids = tuple(continuing)
        if self.memory.raid_long_range_campaign is not None:
            self.memory.raid_long_range_campaign = replace(
                self.memory.raid_long_range_campaign,
                member_ids=tuple(continuing),
                phase="RETURNING",
            )
        if not continuing:
            self._clear()
        return intents

    def _handoff_valid(
        self,
        world: WorldModel,
        projection: TacticalMap,
        position: Position,
        protected: frozenset[Position],
        reserved: set[Position],
    ) -> bool:
        return bool(
            world.core is not None
            and manhattan(position, world.core.position)
            == self.config.home_return_handoff_radius
            and position in world.known_passable
            and position not in world.known_obstacles
            and position not in projection.hostile_occupied
            and position not in protected
            and position not in reserved
            and projection.immediate_attackers(position) == 0
        )

    def _choose_handoff(
        self,
        world: WorldModel,
        projection: TacticalMap,
        unit: EntitySnapshot,
        protected: frozenset[Position],
        reserved: set[Position],
    ) -> Position | None:
        assert world.core is not None
        blocked = frozenset((projection.hostile_occupied | protected | reserved) - {unit.position})
        rows = []
        for cell in manhattan_ring(
            world.core.position,
            self.config.home_return_handoff_radius,
        ):
            if not self._handoff_valid(
                world,
                projection,
                cell,
                protected,
                reserved,
            ):
                continue
            route = route_to(
                world,
                unit.position,
                cell,
                node_limit=self.config.path_node_limit,
                blocked=blocked - {cell},
            )
            if route is None:
                continue
            rows.append((route.distance, cell))
        return min(rows, default=(0, None))[1]

    def _handoff_intents(
        self,
        world: WorldModel,
        projection: TacticalMap,
        unit: EntitySnapshot,
        protected: frozenset[Position],
        reserved: set[Position],
    ) -> list[ActionIntent]:
        """Clear the return corridor once and release a healthy member home."""

        assert world.core is not None
        occupied = dict(world.occupied_cells)
        rows = []
        for index, (direction, destination) in enumerate(
            cardinal_neighbors(unit.position)
        ):
            if (
                destination in reserved
                or destination in protected
                or destination in world.known_obstacles
                or destination in projection.hostile_occupied
                or occupied.get(destination, 0) >= 2
                or projection.immediate_attackers(destination) >= unit.hp
                or projection.future_attackers(destination) >= unit.hp
            ):
                continue
            viability = move_viability(
                world,
                unit.position,
                destination,
                target=None,
                blocked=frozenset(protected - {unit.position, destination}),
                node_limit=min(self.config.path_node_limit, 128),
                require_open_area=True,
            )
            if not viability.viable:
                continue
            score = (
                int(
                    manhattan(destination, world.core.position)
                    > self.config.home_engage_radius
                ),
                occupied.get(destination, 0),
                projection.immediate_attackers(destination),
                projection.future_attackers(destination),
                -viability.forward_exits,
                manhattan(destination, world.core.position),
                index,
            )
            rows.append((score, direction, destination, viability))
        if not rows:
            return [self._wait(unit, "RAID_RETURN_HANDOFF_BLOCKED")]
        score, direction, destination, viability = min(rows, key=lambda row: row[0])
        reserved.add(destination)
        return [
            ActionIntent.move(
                unit.id,
                UnitMission.RAID,
                60,
                direction,
                destination,
                risk=score[2] * 100 + score[3] * 10,
                exclusive_destination=True,
                tie_break=score,
                reason="RAID_RETURN_HANDOFF",
                metadata=(
                    ("handoff_position", destination),
                    ("released_from_raid", True),
                )
                + viability.metadata,
            )
        ]

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
        terminal_exception = None
        if reason in {"RAID_ADVANCE", "RAID_SIEGE_POSITION"}:
            terminal_exception = "ATTACK" if route.first_position == target else None
        elif reason == "RAID_RETURN" and route.first_position == target:
            terminal_exception = "CORE_SERVICE"
        viability = move_viability(
            world,
            unit.position,
            route.first_position,
            target=target,
            blocked=frozenset(blocked),
            node_limit=min(self.config.path_node_limit, 512),
            require_continuation=(
                terminal_exception is None and route.first_position != target
            ),
            require_open_area=(
                terminal_exception is None and route.first_position == target
            ),
            terminal_exception=terminal_exception,
        )
        if not viability.viable:
            return [self._wait(unit, f"{reason}_NO_VIABLE_CONTINUATION")]
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
                metadata=viability.metadata,
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
        self.memory.raid_long_range_campaign = None
        self.memory.raid_confirmation_lease = None
        self.memory.raid_recon_mission = None
        self.memory.raid_distance_band = None
        self.memory.raid_siege_approach = None
        self.memory.raid_interrupted_tick = None
        self.memory.raid_return_reason = None
        self.memory.raid_handoff_targets.clear()
