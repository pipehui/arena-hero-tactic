from __future__ import annotations

import unittest

from arena_hero import Direction, UnitType

from arena_tactic import ActionIntent, IntentAction, IntentResolver, UnitMission
from arena_tactic.world import build_world_model
from tests.helpers import friendly_core, make_turn, unit


class IntentKernelTests(unittest.TestCase):
    def test_higher_priority_intent_wins_for_one_actor(self) -> None:
        worker = unit(1, UnitType.WORKER, (1, 0), cargo=1)
        world = build_world_model(make_turn(units=(worker,)))
        intents = (
            ActionIntent.move(worker.id, UnitMission.EXPLORE, 70, Direction.RIGHT, (2, 0), reason="explore"),
            ActionIntent.simple(worker.id, IntentAction.DEPOSIT, UnitMission.DEPOSIT, 10, reason="deliver"),
        )

        result = IntentResolver().resolve(world, intents)

        self.assertEqual(result.for_actor(worker.id).action, IntentAction.DEPOSIT)
        self.assertEqual(result.rejected[0].reason, "ACTOR_ALREADY_ASSIGNED")

    def test_conflicting_moves_are_deterministic_and_capacity_safe(self) -> None:
        first = unit(1, UnitType.WORKER, (-1, 0))
        second = unit(2, UnitType.WORKER, (1, 0))
        core = friendly_core(position=(10, 10))
        world = build_world_model(make_turn(core=core, units=(first, second)))
        intents = (
            ActionIntent.move(first.id, UnitMission.EXPLORE, 70, Direction.RIGHT, (0, 0), tie_break=(1,)),
            ActionIntent.move(second.id, UnitMission.EXPLORE, 70, Direction.LEFT, (0, 0), tie_break=(2,)),
            ActionIntent.simple(first.id, IntentAction.WAIT, UnitMission.WAIT, 99),
            ActionIntent.simple(second.id, IntentAction.WAIT, UnitMission.WAIT, 99),
        )

        result = IntentResolver(combat_exclusive=True).resolve(world, intents)

        movers = [intent for intent in result.selected if intent.action is IntentAction.MOVE]
        self.assertEqual([intent.actor_id for intent in movers], [first.id])
        self.assertEqual(result.for_actor(second.id).action, IntentAction.WAIT)

    def test_combat_exclusivity_is_bounded_to_the_home_radius(self) -> None:
        first = unit(1, UnitType.WORKER, (19, 0))
        second = unit(2, UnitType.WORKER, (21, 0))
        world = build_world_model(
            make_turn(core=friendly_core(position=(0, 0)), units=(first, second))
        )
        intents = (
            ActionIntent.move(
                first.id,
                UnitMission.EXPLORE,
                70,
                Direction.RIGHT,
                (20, 0),
            ),
            ActionIntent.move(
                second.id,
                UnitMission.EXPLORE,
                70,
                Direction.LEFT,
                (20, 0),
            ),
        )

        result = IntentResolver(
            combat_exclusive=True,
            combat_exclusive_center=(0, 0),
            combat_exclusive_radius=13,
        ).resolve(world, intents)

        self.assertEqual(
            {intent.actor_id for intent in result.selected},
            {first.id, second.id},
        )

    def test_recent_failed_destination_falls_through_to_an_alternative(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0))
        world = build_world_model(make_turn(units=(worker,), resources=0))
        intents = (
            ActionIntent.move(
                worker.id,
                UnitMission.EXPLORE,
                70,
                Direction.RIGHT,
                (1, 0),
                reason="first",
            ),
            ActionIntent.move(
                worker.id,
                UnitMission.EXPLORE,
                70,
                Direction.DOWN,
                (0, 1),
                reason="alternate",
            ),
        )

        result = IntentResolver(
            actor_move_blocks={worker.id: frozenset(((1, 0),))}
        ).resolve(world, intents)

        self.assertEqual(result.for_actor(worker.id).target_position, (0, 1))
        self.assertTrue(
            any(item.reason == "RECENT_MOVE_FAILURE" for item in result.rejected)
        )

    def test_departure_and_entry_can_share_one_tick(self) -> None:
        first = unit(1, UnitType.WORKER, (0, 0))
        second = unit(2, UnitType.WORKER, (1, 0))
        core = friendly_core(position=(10, 10))
        world = build_world_model(make_turn(core=core, units=(first, second)))
        intents = (
            ActionIntent.move(first.id, UnitMission.RETURN_CARGO, 40, Direction.RIGHT, (1, 0)),
            ActionIntent.move(second.id, UnitMission.EXPLORE, 70, Direction.RIGHT, (2, 0)),
        )

        result = IntentResolver().resolve(world, intents)

        self.assertEqual({intent.action for intent in result.selected}, {IntentAction.MOVE})

    def test_preplanned_actor_cannot_fall_through_to_patrol(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        world = build_world_model(make_turn(units=(vanguard,)))
        service_wait = ActionIntent.simple(
            vanguard.id,
            IntentAction.WAIT,
            UnitMission.CLEAR_CORE,
            46,
            reason="CORE_EXIT_BLOCKED_THIS_TICK",
        )
        patrol = ActionIntent.move(
            vanguard.id,
            UnitMission.PATROL,
            70,
            Direction.RIGHT,
            (1, 0),
            reason="PATROL",
        )

        result = IntentResolver(
            actor_priority_ceilings={vanguard.id: 46}
        ).resolve(world, (service_wait, patrol))

        self.assertEqual(result.for_actor(vanguard.id), service_wait)
        patrol_rejection = next(
            item for item in result.rejected if item.intent == patrol
        )
        self.assertEqual(patrol_rejection.reason, "ACTOR_PREPLANNED")

    def test_unplanned_head_on_swap_is_rejected(self) -> None:
        first = unit(1, UnitType.WORKER, (0, 0))
        second = unit(2, UnitType.WORKER, (1, 0))
        world = build_world_model(
            make_turn(core=friendly_core(position=(10, 10)), units=(first, second))
        )
        intents = (
            ActionIntent.move(
                first.id,
                UnitMission.CLEAR_CORE,
                45,
                Direction.RIGHT,
                second.position,
            ),
            ActionIntent.move(
                second.id,
                UnitMission.EXPLORE,
                70,
                Direction.LEFT,
                first.position,
            ),
            ActionIntent.simple(first.id, IntentAction.WAIT, UnitMission.WAIT, 99),
            ActionIntent.simple(second.id, IntentAction.WAIT, UnitMission.WAIT, 99),
        )

        result = IntentResolver().resolve(world, intents)

        selected_moves = [
            intent for intent in result.selected if intent.action is IntentAction.MOVE
        ]
        self.assertLessEqual(len(selected_moves), 1)
        self.assertTrue(
            any(item.reason == "HEAD_ON_SWAP" for item in result.rejected)
        )

    def test_resource_budget_rejects_lower_priority_spend(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0), hp=1)
        ranger = unit(2, UnitType.RANGER, (0, 0), hp=1)
        world = build_world_model(make_turn(units=(vanguard, ranger), resources=2))
        intents = (
            ActionIntent.simple(vanguard.id, IntentAction.HEAL, UnitMission.RECOVER, 30, resource_cost=3),
            ActionIntent.simple(ranger.id, IntentAction.HEAL, UnitMission.RECOVER, 31, resource_cost=1),
            ActionIntent.simple(vanguard.id, IntentAction.WAIT, UnitMission.WAIT, 99),
            ActionIntent.simple(ranger.id, IntentAction.WAIT, UnitMission.WAIT, 99),
        )

        result = IntentResolver().resolve(world, intents)

        self.assertEqual(result.for_actor(vanguard.id).action, IntentAction.WAIT)
        self.assertEqual(result.for_actor(ranger.id).action, IntentAction.HEAL)

    def test_earlier_deposit_can_fund_same_tick_healing(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        ranger = unit(2, UnitType.RANGER, (0, 0), hp=1)
        world = build_world_model(make_turn(units=(worker, ranger), resources=0))
        intents = (
            ActionIntent.simple(worker.id, IntentAction.DEPOSIT, UnitMission.DEPOSIT, 10, resource_gain=1),
            ActionIntent.simple(ranger.id, IntentAction.HEAL, UnitMission.RECOVER, 40, resource_cost=1),
        )

        result = IntentResolver().resolve(world, intents)

        self.assertEqual(
            {intent.action for intent in result.selected},
            {IntentAction.DEPOSIT, IntentAction.HEAL},
        )

    def test_deposit_can_fund_same_tick_core_action(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        world = build_world_model(make_turn(units=(worker,), resources=0))
        intents = (
            ActionIntent.simple(
                worker.id,
                IntentAction.DEPOSIT,
                UnitMission.DEPOSIT,
                10,
                resource_gain=1,
            ),
            ActionIntent.simple(
                None,
                IntentAction.HEAL,
                UnitMission.CORE_SURVIVAL,
                0,
                resource_cost=1,
            ),
        )

        result = IntentResolver().resolve(world, intents)

        self.assertEqual(
            {intent.action for intent in result.selected},
            {IntentAction.DEPOSIT, IntentAction.HEAL},
        )

    def test_core_start_reservation_rejects_lower_priority_second_occupant(self) -> None:
        stationary = unit(1, UnitType.VANGUARD, (1, 0))
        incoming = unit(2, UnitType.RANGER, (1, 1))
        world = build_world_model(make_turn(units=(stationary, incoming)))
        start = ActionIntent(
            actor_id=None,
            action=IntentAction.START_MOVE,
            mission=UnitMission.CORE_SURVIVAL,
            priority=0,
            direction=Direction.RIGHT,
            target_position=(1, 0),
        )
        advance = ActionIntent.move(
            incoming.id,
            UnitMission.HOME_DEFENSE,
            55,
            Direction.UP,
            (1, 0),
        )
        wait = ActionIntent.simple(
            incoming.id,
            IntentAction.WAIT,
            UnitMission.HOME_DEFENSE,
            59,
        )

        result = IntentResolver().resolve(world, (start, advance, wait))

        self.assertEqual(result.for_actor(None).action, IntentAction.START_MOVE)
        self.assertEqual(result.for_actor(incoming.id).action, IntentAction.WAIT)
        rejection = next(item for item in result.rejected if item.intent == advance)
        self.assertEqual(rejection.reason, "CORE_START_DESTINATION_RESERVED")


if __name__ == "__main__":
    unittest.main()
