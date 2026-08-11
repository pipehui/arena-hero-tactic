from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from uuid import UUID

from arena_hero import Position

from .geometry import manhattan
from .models import (
    ActionIntent,
    IntentAction,
    IntentResolution,
    RejectedIntent,
    WorldModel,
)


def _actor_key(actor_id: UUID | None) -> tuple[int, bytes]:
    return (0, b"") if actor_id is None else (1, actor_id.bytes)


class IntentResolver:
    """Resolve pure intents into a complete, deterministic and capacity-safe plan."""

    def __init__(
        self,
        *,
        decision_node_limit: int = 4_096,
        combat_exclusive: bool = False,
        combat_exclusive_center: Position | None = None,
        combat_exclusive_radius: int | None = None,
        protected_positions: frozenset[Position] = frozenset(),
        actor_priority_ceilings: Mapping[UUID, int] | None = None,
        actor_move_blocks: Mapping[UUID, frozenset[Position]] | None = None,
    ) -> None:
        if decision_node_limit <= 0:
            raise ValueError("decision_node_limit must be positive")
        self.decision_node_limit = decision_node_limit
        self.combat_exclusive = combat_exclusive
        self.combat_exclusive_center = combat_exclusive_center
        self.combat_exclusive_radius = combat_exclusive_radius
        if combat_exclusive_radius is not None and combat_exclusive_radius < 0:
            raise ValueError("combat_exclusive_radius must be non-negative")
        self.protected_positions = protected_positions
        # A pre-planned actor may still take a more urgent action (attack,
        # escape or emergency recovery), but ordinary patrol/exploration must
        # not replace its service choreography.  The mapping contains only
        # value types and never retains SDK controllers from a Turn.
        self.actor_priority_ceilings = dict(actor_priority_ceilings or {})
        self.actor_move_blocks = dict(actor_move_blocks or {})

    def resolve(
        self,
        world: WorldModel,
        intents: tuple[ActionIntent, ...] | list[ActionIntent],
    ) -> IntentResolution:
        groups: dict[UUID | None, list[ActionIntent]] = defaultdict(list)
        original = tuple(intents)
        for intent in original:
            groups[intent.actor_id].append(intent)
        for candidates in groups.values():
            candidates.sort(key=ActionIntent.sort_key)

        forbidden: dict[ActionIntent, str] = {}
        for intent in original:
            if intent.actor_id is None:
                continue
            ceiling = self.actor_priority_ceilings.get(intent.actor_id)
            if ceiling is not None and intent.priority > ceiling:
                forbidden[intent] = "ACTOR_PREPLANNED"
            if (
                intent.action is IntentAction.MOVE
                and intent.target_position
                in self.actor_move_blocks.get(intent.actor_id, frozenset())
            ):
                forbidden[intent] = "RECENT_MOVE_FAILURE"
        budget_rejections: set[ActionIntent] = set()
        selected: dict[UUID | None, ActionIntent] = {}

        # Every conflicting pass permanently rejects at least one candidate,
        # so ``len(original) + 1`` is a strict convergence bound.  The
        # configurable ceiling guards malformed integrations that emit an
        # unbounded candidate set without weakening normal safety guarantees.
        iteration_limit = min(self.decision_node_limit, len(original) + 1)
        converged = False
        for _ in range(iteration_limit):
            selected, budget_rejections = self._choose_with_budget(
                world,
                groups,
                forbidden,
            )
            invalid = self._movement_conflicts(world, selected)
            new_invalid = {
                intent: reason
                for intent, reason in invalid.items()
                if intent not in forbidden
            }
            if not new_invalid:
                converged = True
                break
            forbidden.update(new_invalid)
        if not converged:
            # Refuse to return a potentially over-capacity plan.  This should
            # be unreachable with the default budget, but failing closed is
            # safer than submitting unresolved movement dependencies.
            raise RuntimeError("intent resolution exceeded its node limit")

        selected_values = tuple(
            sorted(selected.values(), key=lambda intent: _actor_key(intent.actor_id))
        )
        selected_set = set(selected_values)
        rejected: list[RejectedIntent] = []
        for intent in sorted(original, key=ActionIntent.sort_key):
            if intent in selected_set:
                continue
            reason = forbidden.get(intent)
            if reason is None and intent in budget_rejections:
                reason = "RESOURCE_BUDGET"
            if reason is None:
                reason = "ACTOR_ALREADY_ASSIGNED"
            rejected.append(RejectedIntent(intent=intent, reason=reason))

        reservations = tuple(
            sorted(
                {
                    position
                    for intent in selected_values
                    for position in intent.reserve_positions
                }
            )
        )
        return IntentResolution(
            selected=selected_values,
            rejected=tuple(rejected),
            reserved_positions=reservations,
            resource_spent=sum(intent.resource_cost for intent in selected_values),
            resource_gained=sum(intent.resource_gain for intent in selected_values),
        )

    def _choose_with_budget(
        self,
        world: WorldModel,
        groups: dict[UUID | None, list[ActionIntent]],
        forbidden: dict[ActionIntent, str],
    ) -> tuple[dict[UUID | None, ActionIntent], set[ActionIntent]]:
        selected: dict[UUID | None, ActionIntent] = {}
        budget_rejections: set[ActionIntent] = set()
        available = world.resources

        def actor_order(actor: UUID | None):
            candidates = [
                intent for intent in groups[actor] if intent not in forbidden
            ]
            leading = candidates[0] if candidates else None
            # Worker deposits resolve before combat, Unit healing and the Core
            # action.  Evaluate a Worker's selected gain before resource
            # consumers even when a survival Core intent has a numerically
            # higher tactical priority; otherwise a zero-balance Core would
            # incorrectly reject a heal/spawn that the same-Tick deposit can
            # legally fund.
            return (
                leading is None or leading.resource_gain <= 0,
                leading.sort_key() if leading is not None else (10_000,),
                _actor_key(actor),
            )

        order = sorted(
            groups,
            key=actor_order,
        )
        for actor in order:
            for intent in groups[actor]:
                if intent in forbidden:
                    continue
                if intent.resource_cost > available:
                    budget_rejections.add(intent)
                    continue
                if not self._static_valid(world, intent):
                    forbidden[intent] = "STATIC_CONFLICT"
                    continue
                selected[actor] = intent
                available += intent.resource_gain - intent.resource_cost
                break
        return selected, budget_rejections

    def _static_valid(self, world: WorldModel, intent: ActionIntent) -> bool:
        if intent.actor_id is not None and world.friendly(intent.actor_id) is None:
            return False
        if intent.actor_id is None and world.core is None:
            return False
        if intent.action is IntentAction.MOVE:
            if intent.target_position is None or intent.direction is None:
                return False
            if intent.target_position in world.known_obstacles:
                return False
            if (
                intent.target_position in self.protected_positions
                and not dict(intent.metadata).get("allow_protected", False)
            ):
                return False
        if intent.action is IntentAction.START_MOVE:
            if intent.target_position is None or intent.direction is None:
                return False
            if intent.target_position in world.known_obstacles:
                return False
            if intent.target_position in world.visible_resources:
                return False
        return True

    def _movement_conflicts(
        self,
        world: WorldModel,
        selected: dict[UUID | None, ActionIntent],
    ) -> dict[ActionIntent, str]:
        enemy_positions = {enemy.position for enemy in world.enemies}
        enemy_positions.update(core.position for core in world.enemy_cores)
        final_units, moves_by_destination, invalid = self._project_unit_moves(
            world,
            selected,
            enemy_positions,
        )
        counts = Counter(final_units.values())
        projected_core_position = self._projected_core_position(world)
        invalid.update(
            self._capacity_conflicts(
                world,
                selected,
                final_units,
                counts,
                projected_core_position,
            )
        )
        invalid.update(
            self._reservation_conflicts(
                moves_by_destination,
                projected_core_position,
            )
        )
        invalid.update(self._swap_conflicts(world, selected, final_units))
        invalid.update(
            self._core_conflicts(
                world,
                selected,
                counts,
                enemy_positions,
            )
        )
        return invalid

    @staticmethod
    def _project_unit_moves(
        world: WorldModel,
        selected: dict[UUID | None, ActionIntent],
        enemy_positions: set[Position],
    ) -> tuple[
        dict[UUID, Position],
        dict[Position, list[ActionIntent]],
        dict[ActionIntent, str],
    ]:
        final_units: dict[UUID, Position] = {}
        moves_by_destination: dict[Position, list[ActionIntent]] = defaultdict(list)
        invalid: dict[ActionIntent, str] = {}
        for unit in world.friendlies:
            intent = selected.get(unit.id)
            destination = unit.position
            if intent is not None and intent.action is IntentAction.MOVE:
                assert intent.target_position is not None
                destination = intent.target_position
                moves_by_destination[destination].append(intent)
                if destination in enemy_positions:
                    invalid[intent] = "ENEMY_OCCUPIED_DESTINATION"
            final_units[unit.id] = destination
        return final_units, moves_by_destination, invalid

    @staticmethod
    def _projected_core_position(world: WorldModel) -> Position | None:
        projected_core_position = None
        if world.core is not None:
            projected_core_position = world.core.position
            if (
                world.core.state.value == "MOVING"
                and world.core.destination is not None
                and world.core.move_progress is not None
                and world.core.move_required_ticks is not None
                and world.core.move_progress >= world.core.move_required_ticks - 1
            ):
                projected_core_position = world.core.destination
        return projected_core_position

    def _capacity_conflicts(
        self,
        world: WorldModel,
        selected: dict[UUID | None, ActionIntent],
        final_units: dict[UUID, Position],
        counts: Counter[Position],
        projected_core_position: Position | None,
    ) -> dict[ActionIntent, str]:
        invalid: dict[ActionIntent, str] = {}
        for destination, count in counts.items():
            unit_capacity = 1 if destination == projected_core_position else 2
            contenders = [
                selected[unit.id]
                for unit in world.friendlies
                if final_units[unit.id] == destination
                and unit.id in selected
                and selected[unit.id].action is IntentAction.MOVE
            ]
            service_overlap = any(
                dict(intent.metadata).get("allow_service_overlap", False)
                for intent in contenders
            )
            if not service_overlap and self._combat_cell_is_exclusive(
                destination,
                projected_core_position,
            ):
                unit_capacity = 1
            if not service_overlap and any(
                intent.exclusive_destination for intent in contenders
            ):
                unit_capacity = min(unit_capacity, 1)
            if count <= unit_capacity:
                continue
            stationary = sum(
                1
                for unit in world.friendlies
                if unit.position == destination
                and not (
                    unit.id in selected
                    and selected[unit.id].action is IntentAction.MOVE
                )
            )
            slots = max(0, unit_capacity - stationary)
            contenders.sort(key=ActionIntent.sort_key)
            for loser in contenders[slots:]:
                invalid[loser] = "CELL_CAPACITY"
        return invalid

    def _reservation_conflicts(
        self,
        moves_by_destination: dict[Position, list[ActionIntent]],
        projected_core_position: Position | None,
    ) -> dict[ActionIntent, str]:
        # An explicit exclusive reservation cannot be shared even if the game
        # would technically allow two friendly entities there.
        invalid: dict[ActionIntent, str] = {}
        for destination, contenders in moves_by_destination.items():
            if len(contenders) <= 1:
                continue
            if self._combat_cell_is_exclusive(
                destination,
                projected_core_position,
            ) or any(item.exclusive_destination for item in contenders):
                contenders.sort(key=ActionIntent.sort_key)
                for loser in contenders[1:]:
                    invalid[loser] = "RESERVATION_CONFLICT"
        return invalid

    @staticmethod
    def _swap_conflicts(
        world: WorldModel,
        selected: dict[UUID | None, ActionIntent],
        final_units: dict[UUID, Position],
    ) -> dict[ActionIntent, str]:
        # Arbitrary head-on swaps are legal-looking in a final occupancy
        # count, but they are a common source of two-Tick oscillation: a Core
        # egress action and a patrol action can exchange cells and then reverse
        # on the next Turn.  A narrow one-cell Core corridor is the deliberate
        # exception: an admitted carrier and the Core occupant may perform one
        # explicitly marked service handoff.  Keep the higher-priority move in
        # every unmarked pair; the next conflict pass validates whether it can
        # still enter beside the now-stationary unit.
        moving = {
            unit.id: (unit.position, final_units[unit.id], selected[unit.id])
            for unit in world.friendlies
            if unit.id in selected
            and selected[unit.id].action is IntentAction.MOVE
        }
        invalid: dict[ActionIntent, str] = {}
        seen_pairs: set[frozenset[UUID]] = set()
        for actor_id, (origin, destination, intent) in moving.items():
            for other_id, (other_origin, other_destination, other_intent) in moving.items():
                if actor_id == other_id:
                    continue
                pair = frozenset((actor_id, other_id))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if destination != other_origin or other_destination != origin:
                    continue
                if (
                    dict(intent.metadata).get("allow_head_on_swap", False)
                    and dict(other_intent.metadata).get(
                        "allow_head_on_swap", False
                    )
                ):
                    continue
                loser = max((intent, other_intent), key=ActionIntent.sort_key)
                invalid[loser] = "HEAD_ON_SWAP"
        return invalid

    @staticmethod
    def _core_conflicts(
        world: WorldModel,
        selected: dict[UUID | None, ActionIntent],
        counts: Counter[Position],
        enemy_positions: set[Position],
    ) -> dict[ActionIntent, str]:
        invalid: dict[ActionIntent, str] = {}
        core_intent = selected.get(None)
        if core_intent is not None and core_intent.action is IntentAction.SPAWN:
            assert world.core is not None
            if counts.get(world.core.position, 0) > 0:
                invalid[core_intent] = "CORE_CELL_OCCUPIED"
        if core_intent is not None and core_intent.action is IntentAction.START_MOVE:
            destination = core_intent.target_position
            if destination in enemy_positions:
                invalid[core_intent] = "ENEMY_OCCUPIED_DESTINATION"
            elif destination is not None:
                stationary = [
                    unit
                    for unit in world.friendlies
                    if unit.position == destination
                    and not (
                        unit.id in selected
                        and selected[unit.id].action is IntentAction.MOVE
                    )
                ]
                incoming = sorted(
                    (
                        selected[unit.id]
                        for unit in world.friendlies
                        if unit.id in selected
                        and selected[unit.id].action is IntentAction.MOVE
                        and selected[unit.id].target_position == destination
                    ),
                    key=ActionIntent.sort_key,
                )
                if len(stationary) > 1:
                    invalid[core_intent] = "CELL_CAPACITY"
                else:
                    # A new migration may start beside one friendly Unit.  A
                    # lower-priority formation/logistics move must not defeat
                    # the priority-0 Core escape reservation by becoming a
                    # second occupant during the same movement phase.
                    slots = 1 - len(stationary)
                    for loser in incoming[slots:]:
                        invalid[loser] = "CORE_START_DESTINATION_RESERVED"
        return invalid

    def _combat_cell_is_exclusive(
        self,
        cell: Position,
        projected_core_position: Position | None,
    ) -> bool:
        if not self.combat_exclusive or cell == projected_core_position:
            return False
        if (
            self.combat_exclusive_center is None
            or self.combat_exclusive_radius is None
        ):
            # Retain the small pure-kernel API used by embedders: enabling
            # exclusivity without geometry deliberately means global.
            return True
        return (
            manhattan(cell, self.combat_exclusive_center)
            <= self.combat_exclusive_radius
        )
