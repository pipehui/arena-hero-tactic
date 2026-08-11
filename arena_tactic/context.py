from __future__ import annotations

from dataclasses import dataclass

from arena_hero import Position

from .models import CoreServiceQueue, WorldModel
from .projection import TacticalMap


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """All controller-free values shared by planners during one Turn."""

    world: WorldModel
    tactical_map: TacticalMap
    service: CoreServiceQueue
    protected_positions: frozenset[Position]
    core_starting_move: bool
    combat_active: bool

    @property
    def projection(self) -> TacticalMap:
        """Compatibility view for planners migrated from the old name."""

        return self.tactical_map
