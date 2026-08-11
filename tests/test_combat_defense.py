from __future__ import annotations

import unittest
from collections import Counter

from arena_hero import (
    BeaconStatus,
    ChampionBeacon,
    MoveAction,
    ShootAction,
    SpawnAction,
    SweepAction,
    UnitType,
    WaitAction,
)

from arena_tactic import BalancedTactic, TacticMemory
from arena_tactic.geometry import manhattan_ring, ranger_firing_positions, ranger_line_is_clear
from tests.helpers import enemy_core, friendly_core, make_turn, unit


class CombatDefenseTests(unittest.TestCase):
    def test_distant_healthy_rangers_advance_to_real_enemy_firing_lines(self) -> None:
        rangers = (
            unit(1, UnitType.RANGER, (-4, -3)),
            unit(2, UnitType.RANGER, (0, -4)),
            unit(3, UnitType.RANGER, (4, -3)),
        )
        enemy = unit(100, UnitType.VANGUARD, (0, 6), controlled=False)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-13, 14)
            for y in range(-13, 14)
            if abs(x) + abs(y) <= 13
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(units=rangers, enemies=(enemy,), resources=0)

        tactic.choose_actions(turn)

        tasks = {
            task["actor_id"]: task
            for task in tactic.last_decision_trace["tasks"]
            if task["actor_id"] in {str(ranger.id) for ranger in rangers}
        }
        advancing = [
            task
            for task in tasks.values()
            if task["reason"] == "ADVANCE_TO_DYNAMIC_FIRE_LINE"
        ]
        self.assertGreaterEqual(len(advancing), 2)
        self.assertTrue(all(task["action"] == "MOVE" for task in advancing))

    def test_ranger_prioritizes_visible_enemy_beacon_carrier(self) -> None:
        ranger = unit(1, UnitType.RANGER, (0, 0))
        ordinary = unit(100, UnitType.WORKER, (2, 0), controlled=False)
        carrier = unit(101, UnitType.WORKER, (0, 2), controlled=False)
        beacon = ChampionBeacon(
            position=carrier.position,
            status=BeaconStatus.CARRIED,
            carrier_id=carrier.id,
        )
        turn = make_turn(
            units=(ranger,),
            enemies=(ordinary, carrier),
            resources=0,
            beacon=beacon,
        )

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[ranger.id]
        self.assertIsInstance(action, ShootAction)
        self.assertEqual(action.expected_cell, carrier.position)

    def test_recent_home_contact_holds_formation_through_a_vision_gap(self) -> None:
        defenders = (
            unit(1, UnitType.VANGUARD, (1, 0)),
            unit(2, UnitType.RANGER, (-1, 0)),
        )
        enemy = unit(100, UnitType.RANGER, (20, 0), controlled=False)
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(tick=1, units=defenders, enemies=(enemy,), resources=0)
        )

        tactic.choose_actions(make_turn(tick=2, units=defenders, resources=0))

        missions = {
            item["mission"]
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] in {str(unit.id) for unit in defenders}
        }
        self.assertIn("HOME_DEFENSE", missions)
        self.assertNotIn("PATROL", missions)

        tactic.choose_actions(make_turn(tick=6, units=defenders, resources=0))
        missions = {
            item["mission"]
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] in {str(unit.id) for unit in defenders}
        }
        self.assertIn("PATROL", missions)

    def test_home_alert_freezes_worker_production_after_contact_vanishes(self) -> None:
        units = tuple(
            unit(index, UnitType.WORKER, (index + 30, 0))
            for index in range(1, 14)
        ) + tuple(
            unit(100 + index, UnitType.VANGUARD, (index - 3, 1))
            for index in range(1, 7)
        ) + tuple(
            unit(200 + index, UnitType.RANGER, (index - 3, -1))
            for index in range(1, 7)
        )
        memory = TacticMemory(opening_complete=True)
        tactic = BalancedTactic(memory=memory)
        enemy = unit(500, UnitType.RANGER, (20, 0), controlled=False)
        tactic.choose_actions(
            make_turn(tick=1, units=units, enemies=(enemy,), resources=100)
        )
        turn = make_turn(tick=2, units=units, resources=100)

        tactic.choose_actions(turn)

        self.assertNotIsInstance(turn.plan.core_action, SpawnAction)
        reasons = {
            item["reason"]
            for item in tactic.last_decision_trace["economy"]["production_candidates"]
        }
        self.assertIn("HOME_COMBAT_FREEZE", reasons)

    def test_ranger_line_supports_axis_and_exact_diagonal_only(self) -> None:
        self.assertTrue(ranger_line_is_clear((0, 0), (3, 0), frozenset()))
        self.assertTrue(ranger_line_is_clear((0, 0), (3, 3), frozenset()))
        self.assertFalse(ranger_line_is_clear((0, 0), (2, 1), frozenset()))
        self.assertFalse(ranger_line_is_clear((0, 0), (3, 3), frozenset({(1, 1)})))

    def test_ranger_formation_geometry_keeps_long_diagonal_lines(self) -> None:
        positions = ranger_firing_positions((0, 0))

        self.assertIn((2, 2), positions)
        self.assertIn((-3, 3), positions)
        self.assertNotIn((2, 1), positions)

    def test_urgent_legal_ranger_shot_is_not_overridden_by_patrol(self) -> None:
        ranger = unit(1, UnitType.RANGER, (0, 2))
        enemy = unit(100, UnitType.VANGUARD, (0, 1), controlled=False)
        turn = make_turn(units=(ranger,), enemies=(enemy,))

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[ranger.id], ShootAction)
        self.assertEqual(turn.plan.unit_actions[ranger.id].expected_cell, enemy.position)

    def test_target_firing_stance_precedes_generic_sector_formation(self) -> None:
        ranger = unit(1, UnitType.RANGER, (2, 2))
        enemy = unit(100, UnitType.RANGER, (6, 0), controlled=False)
        turn = make_turn(units=(ranger,), enemies=(enemy,), resources=0)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[ranger.id]
        self.assertIsInstance(action, MoveAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(ranger.id)
        )
        self.assertEqual(task["reason"], "ADVANCE_TO_DYNAMIC_FIRE_LINE")

    def test_high_confidence_two_hp_target_gets_lethal_volley(self) -> None:
        tactic = BalancedTactic()
        rangers = (
            unit(1, UnitType.RANGER, (-2, 0)),
            unit(2, UnitType.RANGER, (2, 0)),
        )
        enemy = unit(100, UnitType.RANGER, (0, 0), hp=2, controlled=False)
        tactic.choose_actions(make_turn(tick=1, units=rangers, enemies=(enemy,)))
        second = make_turn(tick=2, units=rangers, enemies=(enemy,))

        tactic.choose_actions(second)

        shots = [second.plan.unit_actions[ranger.id] for ranger in rangers]
        self.assertTrue(all(isinstance(action, ShootAction) for action in shots))
        self.assertEqual({action.expected_cell for action in shots}, {(0, 0)})

    def test_moving_ranger_uses_current_and_firing_position_split(self) -> None:
        rangers = (
            unit(1, UnitType.RANGER, (2, -2)),
            unit(2, UnitType.RANGER, (2, 2)),
            unit(3, UnitType.RANGER, (1, -2)),
        )
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(
                tick=1,
                units=rangers,
                enemies=(
                    unit(100, UnitType.RANGER, (0, 0), controlled=False),
                ),
                resources=0,
            )
        )
        turn = make_turn(
            tick=2,
            units=rangers,
            enemies=(unit(100, UnitType.RANGER, (1, 0), controlled=False),),
            resources=0,
        )

        tactic.choose_actions(turn)

        shots = [turn.plan.unit_actions[ranger.id] for ranger in rangers]
        self.assertTrue(all(isinstance(action, ShootAction) for action in shots))
        cells = [action.expected_cell for action in shots]
        self.assertEqual(len(set(cells)), 3)
        mission = tactic.last_decision_trace["combat"]["fire_missions"][0]
        self.assertTrue(mission["split_fire"])
        self.assertIn("CURRENT", mission["candidate_roles"])
        self.assertIn("NEXT_FIRE_POSITION", mission["candidate_roles"])

    def test_uncertain_target_candidates_include_wait_and_legal_steps(self) -> None:
        tactic = BalancedTactic()
        ranger = unit(1, UnitType.RANGER, (-3, 0))
        enemy1 = unit(100, UnitType.WORKER, (0, 0), controlled=False)
        tactic.choose_actions(make_turn(tick=1, units=(ranger,), enemies=(enemy1,), obstacle_cells=((0, 1),)))
        enemy2 = unit(100, UnitType.WORKER, (1, 0), controlled=False)
        tactic.choose_actions(make_turn(tick=2, units=(ranger,), enemies=(enemy2,), obstacle_cells=((1, 1),)))

        mission = tactic.last_decision_trace["combat"]["fire_missions"][0]

        self.assertIn([1, 0], mission["candidate_cells"])
        self.assertIn([2, 0], mission["candidate_cells"])
        self.assertNotIn([1, 1], mission["candidate_cells"])
        self.assertGreater(len(mission["candidate_cells"]), 1)
        self.assertLessEqual(len(mission["candidate_cells"]), 5)

    def test_uncertain_vanguard_splits_two_rangers_across_angles(self) -> None:
        rangers = (
            unit(1, UnitType.RANGER, (8, -2)),
            unit(2, UnitType.RANGER, (8, 2)),
        )
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(
                tick=1,
                units=rangers,
                enemies=(unit(100, UnitType.VANGUARD, (10, 0), controlled=False),),
                resources=0,
            )
        )
        turn = make_turn(
            tick=2,
            units=rangers,
            enemies=(unit(100, UnitType.VANGUARD, (11, 0), controlled=False),),
            resources=0,
        )

        tactic.choose_actions(turn)

        shots = [turn.plan.unit_actions[ranger.id] for ranger in rangers]
        self.assertTrue(all(isinstance(action, ShootAction) for action in shots))
        self.assertEqual(len({action.expected_cell for action in shots}), 2)
        mission = tactic.last_decision_trace["combat"]["fire_missions"][0]
        self.assertEqual(mission["prediction_mode"], "UNCERTAIN")
        self.assertTrue(mission["split_fire"])

    def test_retreating_vanguard_prefers_continuing_outward(self) -> None:
        rangers = (
            unit(1, UnitType.RANGER, (10, -3)),
            unit(2, UnitType.RANGER, (10, 3)),
        )
        tactic = BalancedTactic()
        for tick, position in ((1, (10, 0)), (2, (11, 0)), (3, (12, 0))):
            tactic.choose_actions(
                make_turn(
                    tick=tick,
                    units=rangers,
                    enemies=(unit(100, UnitType.VANGUARD, position, controlled=False),),
                    resources=0,
                )
            )

        mission = tactic.last_decision_trace["combat"]["fire_missions"][0]
        self.assertEqual(mission["prediction_mode"], "RETREATING")
        self.assertEqual(mission["candidate_cells"][0], [13, 0])

    def test_outer_contact_forms_two_by_two_screen_and_keeps_four_by_four_home(self) -> None:
        units = list(
            [unit(10 + index, UnitType.VANGUARD, (index - 4, 5)) for index in range(6)]
            + [unit(30 + index, UnitType.RANGER, (index - 4, -5)) for index in range(6)]
        )
        units[0] = unit(10, UnitType.VANGUARD, (12, 0))
        units[1] = unit(11, UnitType.VANGUARD, (11, 1))
        units[6] = unit(30, UnitType.RANGER, (12, 2))
        units[7] = unit(31, UnitType.RANGER, (11, -1))
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-35, 36)
            for y in range(-35, 36)
            if abs(x) + abs(y) <= 35
        )
        tactic = BalancedTactic(memory=memory)
        target = unit(200, UnitType.VANGUARD, (17, 0), controlled=False)

        tactic.choose_actions(
            make_turn(tick=1, units=tuple(units), enemies=(target,), resources=0)
        )

        group = tactic.memory.screening_groups[target.id]
        self.assertEqual(len(group.vanguard_ids), 2)
        self.assertEqual(len(group.ranger_ids), 2)
        all_vanguards = {item.id for item in units if item.unit_type is UnitType.VANGUARD}
        all_rangers = {item.id for item in units if item.unit_type is UnitType.RANGER}
        self.assertEqual(len(all_vanguards - set(group.vanguard_ids)), 4)
        self.assertEqual(len(all_rangers - set(group.ranger_ids)), 4)
        group_ids = {str(item) for item in (*group.vanguard_ids, *group.ranger_ids)}
        group_tasks = [
            task
            for task in tactic.last_decision_trace["tasks"]
            if task["actor_id"] in group_ids
        ]
        self.assertTrue(group_tasks)
        self.assertTrue(
            all(
                task["reason"].startswith("OUTER_SCREEN_")
                or task["reason"]
                in {
                    "INTENT_SPLIT_COVERAGE",
                    "LETHAL_FIRE_PACKAGE",
                    "URGENT_REMAINDER",
                    "URGENT_CROSS_COVERAGE",
                }
                for task in group_tasks
            )
        )

    def test_two_rangers_cross_cover_a_moving_worker(self) -> None:
        rangers = (
            unit(1, UnitType.RANGER, (-1, 0)),
            unit(2, UnitType.RANGER, (4, 0)),
        )
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=friendly_core(position=(10, 10)),
                units=rangers,
                enemies=(unit(100, UnitType.WORKER, (0, 0), controlled=False),),
                resources=0,
            )
        )
        turn = make_turn(
            tick=2,
            core=friendly_core(position=(10, 10)),
            units=rangers,
            enemies=(unit(100, UnitType.WORKER, (1, 0), controlled=False),),
            resources=0,
        )

        tactic.choose_actions(turn)

        shots = [turn.plan.unit_actions[ranger.id] for ranger in rangers]
        self.assertTrue(all(isinstance(action, ShootAction) for action in shots))
        self.assertEqual(
            Counter(action.expected_cell for action in shots),
            Counter({(2, 0): 1, (1, 0): 1}),
        )

    def test_mobile_vanguard_prediction_prefers_profitable_advance_over_current_cell(self) -> None:
        ranger = unit(1, UnitType.RANGER, (3, 0))
        friendly_worker = unit(2, UnitType.WORKER, (3, 1))
        tactic = BalancedTactic()
        for tick, position in ((1, (0, 0)), (2, (0, -1))):
            tactic.choose_actions(
                make_turn(
                    tick=tick,
                    core=friendly_core(position=(5, 5)),
                    units=(ranger, friendly_worker),
                    enemies=(
                        unit(100, UnitType.VANGUARD, position, controlled=False),
                    ),
                    resources=0,
                )
            )

        third = make_turn(
            tick=3,
            core=friendly_core(position=(5, 5)),
            units=(ranger, friendly_worker),
            enemies=(unit(100, UnitType.VANGUARD, (0, 0), controlled=False),),
            resources=0,
        )
        tactic.choose_actions(third)
        self.assertEqual(third.plan.unit_actions[ranger.id].expected_cell, (1, 0))

        fourth = make_turn(
            tick=4,
            core=friendly_core(position=(5, 5)),
            units=(ranger, friendly_worker),
            enemies=(unit(100, UnitType.VANGUARD, (1, 0), controlled=False),),
            resources=0,
        )
        tactic.choose_actions(fourth)
        self.assertEqual(fourth.plan.unit_actions[ranger.id].expected_cell, (2, 0))

    def test_full_health_vanguard_accepts_one_nonfatal_hit_to_improve_intercept(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (2, 0))
        enemy = unit(100, UnitType.VANGUARD, (0, 0), controlled=False)
        tactic = BalancedTactic()
        turn = make_turn(
            tick=1,
            core=friendly_core(position=(0, 5)),
            units=(vanguard,),
            enemies=(enemy,),
            resources=0,
        )

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[vanguard.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction.value, "LEFT")
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(vanguard.id)
        )
        self.assertEqual(task["reason"], "ROUTE_INTERCEPT_ADVANCE")
        self.assertTrue(task["metadata"]["intercept_improved"])
        self.assertEqual(task["metadata"]["immediate_attackers"], 1)

    def test_adjacent_vanguard_sweeps(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        enemy = unit(100, UnitType.RANGER, (1, 0), controlled=False)
        turn = make_turn(units=(vanguard,), enemies=(enemy,))

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[vanguard.id], SweepAction)

    def test_enemy_worker_inside_home_ring_is_intercepted(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        intruder = unit(100, UnitType.WORKER, (2, 0), controlled=False)
        turn = make_turn(units=(vanguard,), enemies=(intruder,), resources=0)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[vanguard.id], MoveAction)

    def test_vanguard_near_urgent_enemy_does_not_patrol_or_plain_wait(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        enemy = unit(100, UnitType.RANGER, (3, 0), controlled=False)
        turn = make_turn(units=(vanguard,), enemies=(enemy,))

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[vanguard.id]
        self.assertIsInstance(action, MoveAction)
        self.assertNotIsInstance(action, WaitAction)

    def test_vanguard_intercepts_core_attacker_before_a_nearer_worker(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (1, 0))
        worker = unit(100, UnitType.WORKER, (3, 0), controlled=False)
        ranger = unit(101, UnitType.RANGER, (0, 3), controlled=False)
        tactic = BalancedTactic()

        tactic.choose_actions(
            make_turn(units=(vanguard,), enemies=(worker, ranger), resources=0)
        )

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(vanguard.id)
        )
        self.assertEqual(task["mission"], "ATTACK")
        self.assertEqual(task["metadata"].get("target_id"), str(ranger.id))

    def test_core_occupant_intercepts_an_attacker_instead_of_generic_egress(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        ranger = unit(101, UnitType.RANGER, (0, 3), controlled=False)
        tactic = BalancedTactic()

        tactic.choose_actions(
            make_turn(units=(vanguard,), enemies=(ranger,), resources=0)
        )

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(vanguard.id)
        )
        self.assertEqual(task["mission"], "ATTACK")
        self.assertNotEqual(task["reason"], "CORE_SERVICE_EXIT")

    def test_remote_worker_attacker_does_not_pull_home_vanguard_across_map(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        remote_worker = unit(2, UnitType.WORKER, (40, 0))
        enemy = unit(100, UnitType.RANGER, (40, 3), controlled=False)
        tactic = BalancedTactic()

        tactic.choose_actions(
            make_turn(units=(vanguard, remote_worker), enemies=(enemy,), resources=0)
        )

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(vanguard.id)
        )
        self.assertNotEqual(task["mission"], "ATTACK")

    def test_single_mobile_ranger_does_not_pull_every_vanguard(self) -> None:
        vanguards = tuple(
            unit(index, UnitType.VANGUARD, (index - 3, 0))
            for index in range(1, 6)
        )
        enemy = unit(100, UnitType.RANGER, (0, 3), controlled=False)
        tactic = BalancedTactic()

        tactic.choose_actions(
            make_turn(units=vanguards, enemies=(enemy,), resources=0)
        )

        attacking = [
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] in {str(unit.id) for unit in vanguards}
            and item["mission"] == "ATTACK"
        ]
        self.assertLessEqual(len(attacking), 2)

    def test_urgent_legal_shot_utilization_is_100_percent(self) -> None:
        rangers = (
            unit(1, UnitType.RANGER, (-2, 0)),
            unit(2, UnitType.RANGER, (2, 0)),
        )
        enemy = unit(100, UnitType.VANGUARD, (0, 0), controlled=False)
        tactic = BalancedTactic()

        tactic.choose_actions(make_turn(units=rangers, enemies=(enemy,)))

        combat = tactic.last_decision_trace["combat"]
        self.assertEqual(combat["legal_attack_opportunities"], 2)
        self.assertEqual(combat["utilization_percent"], 100)

    def test_half_health_unit_with_a_safer_cell_withdraws_before_attacking(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0), hp=2)
        enemy = unit(100, UnitType.RANGER, (0, 3), controlled=False)
        turn = make_turn(units=(vanguard,), enemies=(enemy,))

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[vanguard.id]
        self.assertIsInstance(action, MoveAction)
        self.assertIn(action.direction.value, {"UP", "LEFT", "RIGHT"})

    def test_low_health_ranger_withdraws_instead_of_taking_a_normal_shot(self) -> None:
        ranger = unit(1, UnitType.RANGER, (0, 0), hp=1)
        enemy = unit(100, UnitType.RANGER, (0, 3), controlled=False)
        turn = make_turn(units=(ranger,), enemies=(enemy,), resources=0)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[ranger.id], MoveAction)

    def test_trapped_low_health_ranger_uses_last_stand_fire(self) -> None:
        ranger = unit(1, UnitType.RANGER, (0, 0), hp=1)
        enemy = unit(100, UnitType.RANGER, (0, 3), controlled=False)
        turn = make_turn(
            units=(ranger,),
            enemies=(enemy,),
            obstacle_cells=((-1, 0), (1, 0), (0, -1)),
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[ranger.id], ShootAction)

    def test_exact_enemy_core_fire_precedes_nonurgent_worker_shot(self) -> None:
        ranger = unit(1, UnitType.RANGER, (20, 0))
        core_target = enemy_core(200, (20, 3), hp=1, shield=0)
        worker_target = unit(100, UnitType.WORKER, (21, 0), controlled=False)
        turn = make_turn(
            units=(ranger,),
            enemies=(worker_target, core_target),
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[ranger.id]
        self.assertIsInstance(action, ShootAction)
        self.assertEqual(action.target_id, core_target.id)

    def test_engaged_target_remains_urgent_inside_pursuit_radius(self) -> None:
        ranger = unit(1, UnitType.RANGER, (11, 0))
        tactic = BalancedTactic()
        first_enemy = unit(100, UnitType.RANGER, (13, 0), controlled=False)
        tactic.choose_actions(make_turn(tick=1, units=(ranger,), enemies=(first_enemy,)))
        second_enemy = unit(100, UnitType.RANGER, (14, 0), controlled=False)
        second = make_turn(tick=2, units=(ranger,), enemies=(second_enemy,))

        tactic.choose_actions(second)

        self.assertIsInstance(second.plan.unit_actions[ranger.id], ShootAction)

    def test_seven_peaceful_squads_use_two_two_three_layers(self) -> None:
        units = tuple(
            [unit(10 + index, UnitType.VANGUARD, (index - 3, 1)) for index in range(7)]
            + [unit(30 + index, UnitType.RANGER, (index - 3, -1)) for index in range(7)]
        )
        tactic = BalancedTactic()

        tactic.choose_actions(make_turn(units=units, resources=0))

        self.assertEqual(
            Counter(squad.radius for squad in tactic.memory.squad_states.values()),
            Counter({5: 2, 10: 2, 15: 3}),
        )
        combat_ids = {str(item.id) for item in units}
        self.assertFalse(
            any(
                task["actor_id"] in combat_ids
                and task["reason"] == "NO_LEGAL_TASK"
                for task in tactic.last_decision_trace["tasks"]
            )
        )

    def test_healthy_partner_patrols_while_fixed_partner_recovers(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (1, 0))
        wounded = unit(2, UnitType.RANGER, (1, 1), hp=1)
        second_vanguard = unit(3, UnitType.VANGUARD, (-1, 0))
        second_ranger = unit(4, UnitType.RANGER, (-1, 1))
        tactic = BalancedTactic()

        tactic.choose_actions(
            make_turn(
                units=(vanguard, wounded, second_vanguard, second_ranger),
                resources=0,
            )
        )

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(vanguard.id)
        )
        self.assertEqual(task["mission"], "PATROL")
        self.assertNotEqual(task["reason"], "NO_LEGAL_TASK")

    def test_crowded_home_patrol_offers_alternate_first_steps(self) -> None:
        core = friendly_core(position=(406, -157))
        vanguards = (
            unit(1, UnitType.VANGUARD, (404, -159)),
            unit(2, UnitType.VANGUARD, (405, -160)),
            unit(3, UnitType.VANGUARD, (404, -160)),
        )
        ranger = unit(4, UnitType.RANGER, (408, -159))
        obstacles = (
            (400, -159),
            (403, -160),
            (404, -154),
            (405, -161),
            (405, -158),
            (406, -160),
            (407, -160),
            (408, -161),
            (408, -156),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(
            make_turn(
                core=core,
                units=(*vanguards, ranger),
                obstacle_cells=obstacles,
                resources=0,
            )
        )

        tasks = {
            item["actor_id"]: item
            for item in tactic.last_decision_trace["tasks"]
        }
        self.assertTrue(
            all(tasks[str(unit_view.id)]["reason"] != "NO_LEGAL_TASK" for unit_view in vanguards)
        )

    def test_reassembling_squad_advances_ranger_toward_outer_vanguard(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (5, 0))
        ranger = unit(2, UnitType.RANGER, (0, 1))
        tactic = BalancedTactic()

        turn = make_turn(units=(vanguard, ranger), resources=0)
        tactic.choose_actions(turn)

        ranger_task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(ranger.id)
        )
        vanguard_task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(vanguard.id)
        )
        self.assertIsInstance(turn.plan.unit_actions[ranger.id], MoveAction)
        self.assertEqual(ranger_task["reason"], "SQUAD_REASSEMBLE")
        self.assertEqual(vanguard_task["reason"], "SQUAD_REASSEMBLE_PARTNER_HOLD")

    def test_peaceful_ranger_support_keeps_advancing_with_its_patrol(self) -> None:
        memory = TacticMemory()
        ring = manhattan_ring((0, 0), 5)
        memory.visit_counts.update({cell: 100 for cell in ring})
        memory.visit_counts[(2, -3)] = 0
        memory.visit_counts[(-3, -2)] = 0
        tactic = BalancedTactic(memory=memory)
        vanguard_position = (0, -2)
        ranger_position = (-1, 0)
        ranger_actions = []

        for tick in range(1, 9):
            vanguard = unit(1, UnitType.VANGUARD, vanguard_position)
            ranger = unit(2, UnitType.RANGER, ranger_position)
            turn = make_turn(
                tick=tick,
                units=(vanguard, ranger),
                resources=0,
            )
            tactic.choose_actions(turn)
            vanguard_action = turn.plan.unit_actions[vanguard.id]
            ranger_action = turn.plan.unit_actions[ranger.id]
            ranger_actions.append(ranger_action)
            if isinstance(vanguard_action, MoveAction):
                dx, dy = vanguard_action.direction.delta
                vanguard_position = (
                    vanguard_position[0] + dx,
                    vanguard_position[1] + dy,
                )
            if isinstance(ranger_action, MoveAction):
                dx, dy = ranger_action.direction.delta
                ranger_position = (
                    ranger_position[0] + dx,
                    ranger_position[1] + dy,
                )

        self.assertTrue(
            any(isinstance(action, MoveAction) for action in ranger_actions[2:]),
            "a healthy Ranger must not become a permanent inner support turret",
        )

    def test_distant_core_is_confirmed_then_raid_returns_for_home_threat(self) -> None:
        units = [
            unit(1, UnitType.VANGUARD, (17, 0)),
            unit(2, UnitType.RANGER, (17, 1)),
        ]
        units.extend(
            unit(10 + index, UnitType.VANGUARD, (index - 3, 2))
            for index in range(6)
        )
        units.extend(
            unit(30 + index, UnitType.RANGER, (index - 3, -2))
            for index in range(6)
        )
        observer = unit(60, UnitType.WORKER, (24, 0))
        target = enemy_core(200, (25, 0))
        tactic = BalancedTactic()
        first = make_turn(
            tick=1,
            units=tuple((*units, observer)),
            enemies=(target,),
            resources=0,
        )

        tactic.choose_actions(first)

        first_task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(observer.id)
        )
        self.assertEqual(first_task["reason"], "RAID_TARGET_CONFIRMATION")

        tactic.choose_actions(
            make_turn(
                tick=2,
                units=tuple((*units, observer)),
                enemies=(target,),
                resources=0,
            )
        )
        self.assertNotEqual(tactic.memory.raid_phase, "IDLE")

        home_enemy = unit(300, UnitType.RANGER, (0, 10), controlled=False)
        tactic.choose_actions(
            make_turn(
                tick=3,
                units=tuple((*units, observer)),
                enemies=(target, home_enemy),
                resources=0,
            )
        )
        self.assertEqual(tactic.memory.raid_phase, "RETURNING")

    def test_core_density_launches_four_member_containment_raid(self) -> None:
        units = tuple(
            [
                unit(10 + index, UnitType.VANGUARD, (index - 3, 2))
                for index in range(6)
            ]
            + [
                unit(30 + index, UnitType.RANGER, (index - 3, -2))
                for index in range(6)
            ]
        )
        targets = (enemy_core(200, (27, 0)), enemy_core(201, (31, 0)))
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(tick=1, units=units, enemies=targets, resources=0)
        )

        tactic.choose_actions(
            make_turn(tick=2, units=units, enemies=targets, resources=0)
        )

        self.assertNotEqual(tactic.memory.raid_phase, "IDLE")
        self.assertTrue(tactic.memory.raid_containment_mode)
        self.assertEqual(len(tactic.memory.raid_member_ids), 4)


if __name__ == "__main__":
    unittest.main()
