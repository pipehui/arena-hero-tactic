from __future__ import annotations

from collections import Counter

from arena_hero import UnitType

from .config import TacticConfig
from .models import EconomyPolicyDecision, WorldModel
from .state import TacticMemory


def decide_economy_policy(
    config: TacticConfig,
    memory: TacticMemory,
    world: WorldModel,
) -> EconomyPolicyDecision:
    """Derive the single per-Tick economy policy from authoritative state.

    Storage saturation is deliberately retained as a hysteresis signal.  It
    may stage loaded Workers whenever storage cannot accept cargo, but it only
    releases empty Workers into saturated patrol after the full mature force
    has been built.
    """

    counts = Counter(unit.unit_type for unit in world.friendlies)
    workers = counts[UnitType.WORKER]
    vanguards = counts[UnitType.VANGUARD]
    rangers = counts[UnitType.RANGER]
    combat_units = vanguards + rangers
    home_force_target = max(config.home_force_floor, memory.home_force_high_water)
    mature_worker_target = config.stockpile_worker_target
    mature_combat_target = max(config.stockpile_combat_target, home_force_target)
    population_ready = world.population >= config.population_stockpile_threshold
    worker_ready = workers >= mature_worker_target
    combat_ready = combat_units >= mature_combat_target
    species_ready = (
        vanguards >= config.minimum_vanguards
        and rangers >= config.minimum_rangers
    )
    mature_stockpile_ready = (
        population_ready and worker_ready and combat_ready and species_ready
    )
    storage_full_now = world.resources == world.resource_capacity
    normal_production_requires_full = (
        world.population >= config.worker_only_population_threshold
        and not mature_stockpile_ready
    )
    normal_production_allowed = (
        not mature_stockpile_ready
        and (not normal_production_requires_full or storage_full_now)
    )
    saturated_patrol_active = (
        mature_stockpile_ready and memory.storage_saturated
    )

    if mature_stockpile_ready:
        phase = "MATURE_STOCKPILE"
        production_gate_reason = "HIGH_POP_STOCKPILE"
    elif normal_production_requires_full:
        phase = "FULL_STORAGE_GATED"
        production_gate_reason = (
            "FULL_STORAGE_GATE_OPEN"
            if storage_full_now
            else "WAIT_FOR_FULL_STORAGE"
        )
    else:
        phase = "EARLY_GROWTH"
        production_gate_reason = "EARLY_PRODUCTION_ALLOWED"

    if saturated_patrol_active:
        patrol_gate_reason = "MATURE_STORAGE_SATURATED"
    elif not mature_stockpile_ready:
        patrol_gate_reason = "MATURE_FORCE_INCOMPLETE"
    else:
        patrol_gate_reason = "STORAGE_HYSTERESIS_INACTIVE"

    return EconomyPolicyDecision(
        phase=phase,
        population=world.population,
        workers=workers,
        vanguards=vanguards,
        rangers=rangers,
        combat_units=combat_units,
        home_force_target=home_force_target,
        mature_worker_target=mature_worker_target,
        mature_combat_target=mature_combat_target,
        worker_gap=max(0, mature_worker_target - workers),
        combat_gap=max(0, mature_combat_target - combat_units),
        vanguard_gap=max(0, config.minimum_vanguards - vanguards),
        ranger_gap=max(0, config.minimum_rangers - rangers),
        population_ready=population_ready,
        worker_ready=worker_ready,
        combat_ready=combat_ready,
        species_ready=species_ready,
        mature_stockpile_ready=mature_stockpile_ready,
        storage_full_now=storage_full_now,
        storage_saturated_hysteresis=memory.storage_saturated,
        normal_production_requires_full=normal_production_requires_full,
        normal_production_allowed=normal_production_allowed,
        saturated_patrol_active=saturated_patrol_active,
        production_gate_reason=production_gate_reason,
        patrol_gate_reason=patrol_gate_reason,
    )


__all__ = ("decide_economy_policy",)
