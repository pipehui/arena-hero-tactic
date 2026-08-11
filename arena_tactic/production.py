from __future__ import annotations

from collections import Counter
from math import ceil

from arena_hero import CoreState, UnitType, unit_cost

from .combat import CombatPlanner
from .config import TacticConfig
from .models import ActionIntent, EntitySnapshot, IntentAction, UnitMission, WorldModel
from .geometry import manhattan
from .projection import TacticalMap
from .state import TacticMemory


class ProductionPlanner:
    """Dynamic-price production with explicit economy/defense modes."""

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
        *,
        reserved_resources: int = 0,
    ) -> tuple[list[ActionIntent], tuple[dict[str, object], ...]]:
        if world.core is None or world.core.state is CoreState.MOVING:
            return [], ()
        counts = Counter(unit.unit_type for unit in world.friendlies)
        workers = counts[UnitType.WORKER]
        vanguards = counts[UnitType.VANGUARD]
        rangers = counts[UnitType.RANGER]
        combat_enemies = tuple(
            enemy
            for enemy in world.enemies
            if enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
        )
        home_enemies = tuple(
            enemy
            for enemy in combat_enemies
            if manhattan(enemy.position, world.core.position)
            <= self.config.home_warning_radius
            or self.combat.target_is_urgent(world, projection, enemy)
        )
        urgent_defense = bool(home_enemies) and (
            len(home_enemies) >= vanguards + rangers
            or any(
                self.combat.target_is_urgent(world, projection, enemy)
                for enemy in home_enemies
            )
        )
        home_combat = any(
            self.combat.target_is_urgent(world, projection, enemy)
            for enemy in home_enemies
        )
        home_alert = self.memory.home_defense_alert_until >= world.tick
        home_target = max(self.config.home_force_floor, self.memory.home_force_high_water)
        combat_count = vanguards + rangers
        next_worker_target = ceil(
            (world.population + 1) * self.config.worker_ratio_percent / 100
        )

        chosen: UnitType | None = None
        reinforcement_order: tuple[UnitType, ...] = ()
        reason = "TARGETS_MET"
        if not self.memory.opening_complete:
            if urgent_defense:
                chosen = UnitType.VANGUARD
                reason = "OPENING_THREAT_INTERRUPT"
            elif workers < self.config.opening_worker_target:
                chosen = UnitType.WORKER
                reason = "OPENING_TO_FOUR_WORKERS"
            elif vanguards < 1:
                chosen = UnitType.VANGUARD
                reason = "OPENING_FIRST_VANGUARD"
            elif rangers < 1:
                chosen = UnitType.RANGER
                reason = "OPENING_FIRST_RANGER"
            else:
                self.memory.opening_complete = True
        if self.memory.opening_complete and chosen is None:
            if home_alert and combat_count < home_target:
                reinforcement_order = self._reinforcement_candidates(
                    vanguards, rangers, home_enemies, combat_count, home_target
                )
                chosen = reinforcement_order[0]
                reason = "HOME_ALERT_REINFORCEMENT"
            elif (home_combat or home_alert) and combat_count >= home_target:
                reason = "HOME_COMBAT_FREEZE"
            elif urgent_defense and combat_count < home_target:
                reinforcement_order = self._reinforcement_candidates(
                    vanguards, rangers, home_enemies, combat_count, home_target
                )
                chosen = reinforcement_order[0]
                reason = "EMERGENCY_HOME_FORCE"
            elif world.population >= self.config.population_stockpile_threshold:
                # At this population every extra Unit is in the fourth dynamic
                # price band.  Keep the Core liquid for healing, evacuation and
                # an actual home-defense emergency instead of growing the
                # already worker-heavy economy without a ceiling.
                reason = "POPULATION_STOCKPILE"
            elif world.population < self.config.worker_only_population_threshold:
                if workers < next_worker_target:
                    chosen = UnitType.WORKER
                    reason = "PRE25_WORKER_RATIO"
                elif (
                    vanguards < self.config.minimum_vanguards
                    or rangers < self.config.minimum_rangers
                ):
                    reinforcement_order = self._reinforcement_candidates(
                        vanguards, rangers, home_enemies, combat_count, home_target
                    )
                    chosen = reinforcement_order[0]
                    reason = "PRE25_COMBAT_FLOOR"
                elif combat_count < home_target:
                    reinforcement_order = self._reinforcement_candidates(
                        vanguards, rangers, home_enemies, combat_count, home_target
                    )
                    chosen = reinforcement_order[0]
                    reason = "PRE25_HOME_FORCE"
                else:
                    chosen = UnitType.WORKER
                    reason = "PRE25_ECONOMY"
            elif (
                vanguards < self.config.minimum_vanguards
                or rangers < self.config.minimum_rangers
                or combat_count < home_target
            ):
                reinforcement_order = self._reinforcement_candidates(
                    vanguards, rangers, home_enemies, combat_count, home_target
                )
                chosen = reinforcement_order[0]
                reason = "POST25_HOME_FORCE"
            else:
                chosen = UnitType.WORKER
                reason = "POST25_WORKER_ONLY"

        available = max(0, world.resources - reserved_resources)
        strategic_primary = chosen
        selection_order = reinforcement_order or (() if chosen is None else (chosen,))
        selected = next(
            (
                unit_type
                for unit_type in selection_order
                if unit_cost(unit_type, world.population) <= available
            ),
            None,
        )
        fallback_reason = (
            "PRIMARY_UNAFFORDABLE"
            if selected is not None
            and strategic_primary is not None
            and selected is not strategic_primary
            else None
        )
        candidates = tuple(
            {
                "unit_type": unit_type.value,
                "cost": unit_cost(unit_type, world.population),
                "chosen": unit_type is selected,
                "strategic_primary": unit_type is strategic_primary,
                "selected": unit_type is selected,
                "fallback_reason": fallback_reason if unit_type is selected else None,
                "affordable": unit_cost(unit_type, world.population) <= available,
                "reserved_for_recovery": reserved_resources,
                "reason": (
                    reason
                    if unit_type in selection_order or strategic_primary is None
                    else "LOWER_PRIORITY"
                ),
            }
            for unit_type in UnitType
        )
        if selected is None:
            return [], candidates
        cost = unit_cost(selected, world.population)
        return [
            ActionIntent.simple(
                None,
                IntentAction.SPAWN,
                UnitMission.PRODUCTION,
                80,
                unit_type=selected,
                resource_cost=cost,
                reason=reason,
            )
        ], candidates

    def _reinforcement_candidates(
        self,
        vanguards: int,
        rangers: int,
        enemies: tuple[EntitySnapshot, ...],
        combat_count: int,
        home_target: int,
    ) -> tuple[UnitType, ...]:
        vanguard_gap = max(0, self.config.minimum_vanguards - vanguards)
        ranger_gap = max(0, self.config.minimum_rangers - rangers)
        if vanguard_gap > ranger_gap:
            primary = UnitType.VANGUARD
        elif ranger_gap > vanguard_gap:
            primary = UnitType.RANGER
        else:
            enemy_rangers = sum(enemy.unit_type is UnitType.RANGER for enemy in enemies)
            enemy_vanguards = sum(enemy.unit_type is UnitType.VANGUARD for enemy in enemies)
            if enemy_rangers > enemy_vanguards:
                primary = UnitType.RANGER
            else:
                primary = (
                    UnitType.VANGUARD
                    if vanguards <= rangers
                    else UnitType.RANGER
                )
        valid: list[UnitType] = []
        if vanguard_gap > 0:
            valid.append(UnitType.VANGUARD)
        if ranger_gap > 0:
            valid.append(UnitType.RANGER)
        if not valid and combat_count < home_target:
            valid.extend((UnitType.VANGUARD, UnitType.RANGER))
        if primary not in valid:
            valid.insert(0, primary)
        ordered = [primary]
        ordered.extend(unit_type for unit_type in valid if unit_type is not primary)
        return tuple(ordered)

    def _reinforcement(
        self,
        vanguards: int,
        rangers: int,
        enemies: tuple[EntitySnapshot, ...],
    ) -> UnitType:
        """Compatibility helper retained for private diagnostic callers."""

        return self._reinforcement_candidates(
            vanguards,
            rangers,
            enemies,
            vanguards + rangers,
            max(self.config.home_force_floor, vanguards + rangers + 1),
        )[0]
