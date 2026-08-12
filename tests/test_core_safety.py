from __future__ import annotations

import unittest

from arena_hero import (
    BeaconStatus,
    CancelMoveAction,
    ChampionBeacon,
    DepositAction,
    Direction,
    HealAction,
    MoveAction,
    ResolutionEvent,
    RepairShieldAction,
    SpawnAction,
    StartMoveAction,
    UnitType,
)

from arena_tactic import BalancedTactic, TacticMemory
from tests.helpers import friendly_core, make_turn, uid, unit


class CoreSafetyTests(unittest.TestCase):
    def test_core_evacuation_rejects_a_three_wall_pocket(self) -> None:
        enemies = tuple(
            unit(100 + index, UnitType.RANGER, (0, 3 + index), controlled=False)
            for index in range(4)
        )
        turn = make_turn(
            enemies=enemies,
            obstacle_cells=((-1, -1), (1, -1), (0, -2)),
            resources=20,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, StartMoveAction)
        self.assertNotEqual(turn.plan.core_action.direction, Direction.UP)

    def test_moving_core_cancels_a_newly_revealed_dead_end_destination(self) -> None:
        core = friendly_core(moving=True, direction=Direction.RIGHT)
        turn = make_turn(
            core=core,
            obstacle_cells=((1, -1), (1, 1), (2, 0)),
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, CancelMoveAction)

    def test_core_may_pass_a_narrow_cell_that_opens_within_two_steps(self) -> None:
        enemies = tuple(
            unit(100 + index, UnitType.RANGER, (0, 3 + index), controlled=False)
            for index in range(4)
        )
        turn = make_turn(
            enemies=enemies,
            obstacle_cells=((-1, 0), (1, 0), (-1, -1), (1, -1)),
            resources=20,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, StartMoveAction)
        self.assertEqual(turn.plan.core_action.direction, Direction.UP)

    def test_core_rejects_a_narrow_corridor_that_closes_ahead(self) -> None:
        enemies = tuple(
            unit(100 + index, UnitType.RANGER, (0, 3 + index), controlled=False)
            for index in range(4)
        )
        turn = make_turn(
            enemies=enemies,
            obstacle_cells=(
                (-1, -1),
                (1, -1),
                (-1, -2),
                (1, -2),
                (0, -3),
            ),
            resources=20,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, StartMoveAction)
        self.assertNotEqual(turn.plan.core_action.direction, Direction.UP)

    def test_core_waits_when_every_escape_direction_is_a_dead_end(self) -> None:
        enemies = self._overwhelming_enemy()
        turn = make_turn(
            enemies=enemies,
            obstacle_cells=(
                (-1, -1),
                (-2, 0),
                (-1, 1),
                (1, -1),
                (2, 0),
                (1, 1),
                (0, -2),
                (0, 2),
            ),
            resources=0,
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertNotIsInstance(turn.plan.core_action, StartMoveAction)
        safety = tactic.last_decision_trace["core_safety"]
        self.assertTrue(safety["no_escape_route"])
        self.assertTrue(
            all(
                not candidate["viable"]
                and candidate["rejection_reason"]
                == "CORE_DEAD_END_DESTINATION"
                for candidate in safety["move_candidates"]
            )
        )


    def test_core_does_not_build_beacon_shield_after_certain_carrier_death(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, 2), hp=1)
        attacker = unit(100, UnitType.RANGER, (0, 5), controlled=False)
        beacon = ChampionBeacon(
            position=carrier.position,
            status=BeaconStatus.CARRIED,
            carrier_id=carrier.id,
        )
        turn = make_turn(
            units=(carrier,),
            enemies=(attacker,),
            resources=1,
            beacon=beacon,
        )

        BalancedTactic().choose_actions(turn)

        self.assertNotIsInstance(turn.plan.core_action, RepairShieldAction)

    def _overwhelming_enemy(self) -> tuple:
        return tuple(
            unit(100 + index, UnitType.RANGER, (0, 3 + index), controlled=False)
            for index in range(4)
        )

    def test_overwhelmed_core_starts_moving(self) -> None:
        turn = make_turn(enemies=self._overwhelming_enemy(), resources=20)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, StartMoveAction)

    def test_evacuation_continues_after_more_than_three_successes(self) -> None:
        tactic = BalancedTactic()
        enemies = self._overwhelming_enemy()
        for tick in range(1, 6):
            turn = make_turn(tick=tick, core=friendly_core(position=(tick - 1, 0)), enemies=enemies, resources=20)
            tactic.choose_actions(turn)
            self.assertIsInstance(turn.plan.core_action, StartMoveAction)

    def test_moving_core_does_not_deposit_or_spawn(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        turn = make_turn(core=friendly_core(moving=True), units=(carrier,), resources=20)

        BalancedTactic().choose_actions(turn)

        self.assertNotIsInstance(turn.plan.unit_actions[carrier.id], DepositAction)
        self.assertNotIsInstance(turn.plan.core_action, SpawnAction)

    def test_starting_migration_blocks_same_tick_deposit(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        turn = make_turn(
            units=(carrier,),
            enemies=self._overwhelming_enemy(),
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, StartMoveAction)
        self.assertNotIsInstance(turn.plan.unit_actions[carrier.id], DepositAction)

    def test_starting_migration_blocks_same_tick_unit_heal(self) -> None:
        ranger = unit(1, UnitType.RANGER, (0, 0), hp=1)
        turn = make_turn(
            units=(ranger,),
            enemies=self._overwhelming_enemy(),
            resources=1,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, StartMoveAction)
        self.assertNotIsInstance(turn.plan.unit_actions[ranger.id], HealAction)

    def test_carrier_follows_the_projected_core_lane_during_migration(self) -> None:
        carrier = unit(1, UnitType.WORKER, (-3, 0), cargo=1)
        turn = make_turn(
            core=friendly_core(moving=True, direction=Direction.RIGHT),
            units=(carrier,),
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[carrier.id], MoveAction)

    def test_core_survival_budget_preempts_production(self) -> None:
        turn = make_turn(core=friendly_core(hp=2), enemies=self._overwhelming_enemy(), resources=12)

        BalancedTactic().choose_actions(turn)

        self.assertNotIsInstance(turn.plan.core_action, SpawnAction)

    def test_peaceful_relocation_uses_a_worker_verified_defensible_site(self) -> None:
        passable = {(x, 0) for x in range(0, 13)}
        passable.update({(12, -1), (11, -1), (10, -1), (12, -2), (12, 2), (10, 0)})
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            known_obstacles={(12, 1)},
            known_passable=passable,
            worker_cell_last_visible={(12, 0): 1},
            strategic_relocation_pending=True,
            strategic_relocation_safe_ticks=8,
            opening_complete=True,
        )
        turn = make_turn(tick=1, resources=0)

        BalancedTactic(memory=memory).choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, StartMoveAction)

    def test_moving_core_cancels_when_destination_becomes_a_resource(self) -> None:
        core = friendly_core(moving=True)
        turn = make_turn(core=core, resource_cells=(core.destination,), resources=0)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, CancelMoveAction)

    def test_moving_core_cancels_before_finishing_on_two_units(self) -> None:
        core = friendly_core(moving=True, progress=3)
        units = (
            unit(1, UnitType.WORKER, core.destination),
            unit(2, UnitType.WORKER, core.destination),
        )
        turn = make_turn(core=core, units=units, resources=0)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, CancelMoveAction)

    def test_two_recent_guard_losses_start_an_evacuation_campaign(self) -> None:
        tactic = BalancedTactic()
        guards = (
            unit(1, UnitType.VANGUARD, (1, 0)),
            unit(2, UnitType.RANGER, (-1, 0)),
        )
        tactic.choose_actions(make_turn(tick=1, units=guards, resources=0))
        turn = make_turn(tick=2, units=(), resources=0)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, StartMoveAction)

    def test_damage_event_for_previous_core_does_not_trigger_relocation(self) -> None:
        memory = TacticMemory(core_id=uid(10_000), core_position=(0, 0))
        event = ResolutionEvent(
            event_id=uid(90_000),
            tick=1,
            event_type="CORE_DAMAGED",
            target_id=uid(9_999),
            values={"hp_damage": 1},
        )
        turn = make_turn(tick=1, events=(event,), resources=0)

        BalancedTactic(memory=memory).choose_actions(turn)

        self.assertFalse(memory.strategic_relocation_pending)

    def test_failed_core_start_backs_off_the_attempted_destination_not_origin(self) -> None:
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            last_core_move_destination=(1, 0),
        )
        event = ResolutionEvent(
            event_id=uid(90_001),
            tick=1,
            event_type="CORE_MOVE_START_FAILED",
            actor_id=uid(10_000),
            position=(0, 0),
            reason_code="CORE_DESTINATION_OCCUPIED",
        )

        BalancedTactic(memory=memory).choose_actions(
            make_turn(tick=2, events=(event,), resources=0)
        )

        self.assertIn((1, 0), memory.failed_core_destinations)
        self.assertNotIn((0, 0), memory.failed_core_destinations)


if __name__ == "__main__":
    unittest.main()
