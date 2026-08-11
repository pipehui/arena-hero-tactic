from __future__ import annotations

from types import MappingProxyType

from arena_hero import UnitType


# Gameplay v0.14 values that the SDK does not currently expose as helpers.
# Prices and Core capacity deliberately continue to use arena_hero.unit_cost
# and Turn.resource_capacity rather than duplicating their formulas.
UNIT_MAX_HP = MappingProxyType(
    {
        UnitType.WORKER: 2,
        UnitType.VANGUARD: 4,
        UnitType.RANGER: 2,
    }
)
CORE_MAX_HP = 5
CORE_BASE_SHIELD_CAP = 5
CORE_BEACON_SHIELD_CAP = 10
CORE_VISION_RADIUS = 5
UNIT_VISION_RADIUS = MappingProxyType(
    {
        UnitType.WORKER: 3,
        UnitType.VANGUARD: 4,
        UnitType.RANGER: 5,
    }
)
