from __future__ import annotations

import unittest
from collections import Counter
from itertools import groupby

from arena_hero import (
    BeaconStatus,
    ChampionBeacon,
    Direction,
    MoveAction,
    ResolutionEvent,
    ShootAction,
    SpawnAction,
    SweepAction,
    UnitType,
    WaitAction,
)

from arena_tactic import (
    ActionIntent,
    BalancedTactic,
    FormationMoveFeedback,
    IntentResolver,
    ScreeningGroupState,
    TacticMemory,
    UnitMission,
    build_tactical_map,
)
from arena_tactic.geometry import manhattan_ring, ranger_firing_positions, ranger_line_is_clear
from arena_tactic.models import ShotFeedback, SquadState
from arena_tactic.world import build_world_model
from tests.helpers import enemy_core, friendly_core, make_turn, uid, unit


class CombatDefenseTests(unittest.TestCase):
    def test_returning_raid_member_handoffs_instead_of_waiting_at_radius(self) -> None:
        core = friendly_core(position=(0, 0))
        member = unit(1, UnitType.RANGER, (13, 0))
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            opening_complete=True,
            raid_member_ids=(member.id,),
            raid_phase="RETURNING",
        )
        memory.known_passable.update(
            (x, y) for x in range(-15, 16) for y in range(-3, 4)
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(core=core, units=(member,), resources=0)

        tactic.choose_actions(turn)

        self.assertNotIsInstance(turn.plan.unit_actions[member.id], WaitAction)
        self.assertNotIn(member.id, tactic.memory.raid_member_ids)

    def test_single_ranger_does_not_blind_fire_at_moving_nonurgent_worker(self) -> None:
        core = friendly_core(position=(20, 20))
        ranger = unit(1, UnitType.RANGER, (0, 0))
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=core,
                units=(ranger,),
                enemies=(unit(100, UnitType.WORKER, (3, 0), controlled=False),),
                resources=0,
            )
        )
        turn = make_turn(
            tick=2,
            core=core,
            units=(ranger,),
            enemies=(unit(100, UnitType.WORKER, (2, 0), controlled=False),),
            resources=0,
        )

        tactic.choose_actions(turn)

        self.assertNotIsInstance(turn.plan.unit_actions[ranger.id], ShootAction)
        mission = next(
            row
            for row in tactic.last_decision_trace["combat"]["fire_missions"]
            if row["target_id"] == str(uid(100))
        )
        self.assertFalse(mission["urgent"])
        self.assertEqual(
            tactic.last_decision_trace["combat"]["low_value_shots_deferred"][0][
                "target_id"
            ],
            str(uid(100)),
        )

    def test_confirmed_distance_fifty_core_launches_surplus_two_by_two_raid(self) -> None:
        core = friendly_core(position=(0, 0))
        vanguards = tuple(
            unit(100 + index, UnitType.VANGUARD, (-1 - index, 0))
            for index in range(15)
        )
        rangers = tuple(
            unit(200 + index, UnitType.RANGER, (index % 7, 1 + index // 7))
            for index in range(14)
        )
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            opening_complete=True,
            home_force_high_water=22,
        )
        memory.known_passable.update(
            (x, y) for x in range(-18, 51) for y in range(-4, 5)
        )
        tactic = BalancedTactic(memory=memory)

        for tick in (1, 2):
            tactic.choose_actions(
                make_turn(
                    tick=tick,
                    core=core,
                    units=(*vanguards, *rangers),
                    enemies=(enemy_core(999, (50, 0)),),
                    resources=0,
                )
            )

        campaign = tactic.memory.raid_long_range_campaign
        self.assertIsNotNone(campaign)
        assert campaign is not None
        self.assertEqual(len(campaign.member_ids), 4)
        member_types = Counter(
            next(
                unit_view.unit_type
                for unit_view in (*vanguards, *rangers)
                if unit_view.id == member_id
            )
            for member_id in campaign.member_ids
        )
        self.assertEqual(member_types[UnitType.VANGUARD], 2)
        self.assertEqual(member_types[UnitType.RANGER], 2)
        self.assertGreaterEqual(29 - len(campaign.member_ids), 22)
        raid_trace = tactic.last_decision_trace["combat"]["raid"]
        self.assertEqual(raid_trace["long_range"]["route_eta"], 50)

    def test_distance_fifty_core_does_not_launch_without_full_home_reserve(self) -> None:
        core = friendly_core(position=(0, 0))
        combatants = tuple(
            unit(index + 1, unit_type, (-index, 0))
            for index, unit_type in enumerate(
                (UnitType.VANGUARD,) * 6 + (UnitType.RANGER,) * 6
            )
        )
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            opening_complete=True,
            home_force_high_water=12,
        )
        memory.known_passable.update(
            (x, y) for x in range(-12, 51) for y in range(-2, 3)
        )
        tactic = BalancedTactic(memory=memory)

        for tick in (1, 2):
            tactic.choose_actions(
                make_turn(
                    tick=tick,
                    core=core,
                    units=combatants,
                    enemies=(enemy_core(999, (50, 0)),),
                    resources=0,
                )
            )

        self.assertIsNone(tactic.memory.raid_long_range_campaign)
        self.assertEqual(tactic.memory.raid_phase, "IDLE")

    def test_two_cell_reversal_prefers_the_stable_next_turning_point(self) -> None:
        ranger = unit(1, UnitType.RANGER, (3, 0))
        tactic = BalancedTactic()
        for tick, position in ((1, (0, 0)), (2, (1, 0))):
            tactic.choose_actions(
                make_turn(
                    tick=tick,
                    core=friendly_core(position=(10, 10)),
                    units=(ranger,),
                    enemies=(unit(100, UnitType.WORKER, position, controlled=False),),
                    resources=0,
                )
            )
        turn = make_turn(
            tick=3,
            core=friendly_core(position=(10, 10)),
            units=(ranger,),
            enemies=(unit(100, UnitType.WORKER, (0, 0), controlled=False),),
            resources=0,
        )

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[ranger.id]
        self.assertIsInstance(action, ShootAction)
        self.assertEqual(action.expected_cell, (1, 0))

    def test_nonurgent_low_rank_shot_yields_to_top_candidate_stance(self) -> None:
        ranger = unit(1, UnitType.RANGER, (11, -3))
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=friendly_core(position=(0, 0)),
                units=(ranger,),
                enemies=(unit(100, UnitType.WORKER, (14, 0), controlled=False),),
                resources=0,
            )
        )
        turn = make_turn(
            tick=2,
            core=friendly_core(position=(0, 0)),
            units=(ranger,),
            enemies=(unit(100, UnitType.WORKER, (15, 0), controlled=False),),
            resources=0,
        )

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[ranger.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.RIGHT)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(ranger.id)
        )
        self.assertEqual(task["reason"], "LOW_VALUE_FIRE_REPOSITION")
        self.assertGreater(
            task["metadata"]["first_step_high_value_coverage"],
            0,
        )

    def test_high_confidence_prediction_does_not_bypass_miss_suppression(self) -> None:
        ranger = unit(1, UnitType.RANGER, (3, 0))
        target_id = uid(100)
        core = friendly_core(position=(10, 10))
        tactic = BalancedTactic(
            memory=TacticMemory(core_id=core.id, core_position=core.position)
        )
        for tick, position in ((1, (0, 0)), (2, (1, 0))):
            tactic.choose_actions(
                make_turn(
                    tick=tick,
                    core=friendly_core(position=(10, 10)),
                    units=(ranger,),
                    enemies=(unit(100, UnitType.WORKER, position, controlled=False),),
                    resources=0,
                )
            )
        tactic.memory.ranger_shot_feedback[(target_id, (1, 0))] = ShotFeedback(
            target_id=target_id,
            expected_cell=(1, 0),
            misses=2,
            suppressed_until=3,
            last_evidence_tick=2,
        )
        turn = make_turn(
            tick=3,
            core=core,
            units=(ranger,),
            enemies=(unit(100, UnitType.WORKER, (0, 0), controlled=False),),
            resources=0,
        )

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[ranger.id]
        self.assertIsInstance(action, ShootAction)
        self.assertNotEqual(action.expected_cell, (1, 0))

    def test_single_current_observation_does_not_release_miss_suppression(self) -> None:
        ranger = unit(1, UnitType.RANGER, (3, 0))
        target_id = uid(100)
        core = friendly_core(position=(10, 10))
        tactic = BalancedTactic(
            memory=TacticMemory(core_id=core.id, core_position=core.position)
        )
        tactic.memory.ranger_shot_feedback[(target_id, (0, 0))] = ShotFeedback(
            target_id=target_id,
            expected_cell=(0, 0),
            misses=2,
            suppressed_until=4,
            last_evidence_tick=1,
        )
        turn = make_turn(
            tick=2,
            core=core,
            units=(ranger,),
            enemies=(unit(100, UnitType.WORKER, (0, 0), controlled=False),),
            resources=0,
        )

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[ranger.id]
        self.assertFalse(
            isinstance(action, ShootAction) and action.expected_cell == (0, 0)
        )
        feedback = tactic.memory.ranger_shot_feedback[(target_id, (0, 0))]
        self.assertEqual(feedback.misses, 2)
        self.assertIsNone(feedback.release_reason)

    def test_stationary_vanguard_promotes_current_cell_and_releases_suppression(self) -> None:
        core = friendly_core(position=(10, 10))
        rangers = (
            unit(1, UnitType.RANGER, (3, 0)),
            unit(2, UnitType.RANGER, (0, 3)),
        )
        target = unit(100, UnitType.VANGUARD, (0, 0), controlled=False)
        tactic = BalancedTactic(
            memory=TacticMemory(core_id=core.id, core_position=core.position)
        )
        tactic.memory.ranger_shot_feedback[(target.id, target.position)] = ShotFeedback(
            target_id=target.id,
            expected_cell=target.position,
            misses=2,
            suppressed_until=4,
            last_evidence_tick=1,
        )

        tactic.choose_actions(
            make_turn(tick=1, core=core, units=rangers, enemies=(target,), resources=0)
        )
        tactic.choose_actions(
            make_turn(tick=2, core=core, units=rangers, enemies=(target,), resources=0)
        )

        mission = tactic.last_decision_trace["combat"]["fire_missions"][0]
        self.assertEqual(mission["prediction_mode"], "STATIONARY")
        self.assertEqual(mission["candidate_cells"][0], [0, 0])
        self.assertIn("TWO_TICK_STATIONARY", mission["evidence"])
        feedback = tactic.memory.ranger_shot_feedback[(target.id, target.position)]
        self.assertEqual(feedback.misses, 0)
        self.assertEqual(feedback.release_reason, "STATIONARY_CONFIRMATION")

    def test_three_tick_stationary_vanguard_concentrates_current_cell_fire(self) -> None:
        core = friendly_core(position=(10, 10))
        rangers = tuple(
            unit(index, UnitType.RANGER, position)
            for index, position in enumerate(
                ((3, 0), (-3, 0), (0, 3), (0, -3)),
                start=1,
            )
        )
        target = unit(100, UnitType.VANGUARD, (0, 0), controlled=False)
        tactic = BalancedTactic(
            memory=TacticMemory(core_id=core.id, core_position=core.position)
        )

        for tick in range(1, 4):
            turn = make_turn(
                tick=tick,
                core=core,
                units=rangers,
                enemies=(target,),
                resources=0,
            )
            tactic.choose_actions(turn)

        mission = tactic.last_decision_trace["combat"]["fire_missions"][0]
        self.assertEqual(mission["prediction_mode"], "STATIONARY")
        self.assertEqual(mission["confidence"], "HIGH")
        shots = [
            action
            for action in turn.plan.unit_actions.values()
            if isinstance(action, ShootAction)
        ]
        self.assertGreaterEqual(len(shots), target.hp)
        self.assertTrue(
            all(
                action.expected_cell == target.position
                for action in shots[: target.hp]
            )
        )

    def test_peaceful_squads_globally_reserve_distinct_support_slots(self) -> None:
        vanguards = (
            unit(1, UnitType.VANGUARD, (0, -2)),
            unit(3, UnitType.VANGUARD, (0, 2)),
        )
        rangers = (
            unit(2, UnitType.RANGER, (-1, 0)),
            unit(4, UnitType.RANGER, (1, 0)),
        )
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))

        tactic.choose_actions(
            make_turn(units=(*vanguards, *rangers), resources=0)
        )

        supports = [
            squad.support_target
            for squad in tactic.memory.squad_states.values()
            if squad.support_target is not None
        ]
        self.assertEqual(len(supports), len(set(supports)))
        self.assertEqual(tactic.last_decision_trace["schema_version"], 41)
        formation = tactic.last_decision_trace["combat"]["formation"]
        assigned_supports = [
            tuple(bundle["support"])
            for bundle in formation["assignment"]["bundles"]
        ]
        self.assertEqual(len(assigned_supports), len(set(assigned_supports)))
        self.assertIn("blocked_or_idle_percent", formation["waits"])

    def test_stacked_noncombat_occupant_keeps_formation_cell_blocked(self) -> None:
        worker = unit(9, UnitType.WORKER, (0, -5))
        vanguard = unit(1, UnitType.VANGUARD, (0, -5))
        ranger = unit(2, UnitType.RANGER, (0, -3))
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))

        tactic.choose_actions(
            make_turn(
                units=(worker, vanguard, ranger),
                resources=0,
            )
        )

        assignment = tactic.last_decision_trace["combat"]["formation"]["assignment"]
        self.assertIsNotNone(assignment)
        claimed = {
            tuple(bundle[key])
            for bundle in assignment["bundles"]
            for key in ("anchor", "support")
        }
        self.assertNotIn(worker.position, claimed)

    def test_arrived_patrol_member_does_not_wait_beyond_partner_progress_limit(self) -> None:
        core = friendly_core()
        vanguard = unit(1, UnitType.VANGUARD, (0, -5))
        ranger = unit(2, UnitType.RANGER, (0, -2))
        memory = TacticMemory(
            opening_complete=True,
            core_id=core.id,
            core_position=core.position,
        )
        key = (vanguard.id, ranger.id)
        memory.squad_states[key] = SquadState(
            vanguard_id=vanguard.id,
            ranger_id=ranger.id,
            radius=5,
            sector_index=0,
            patrol_anchor=vanguard.position,
            support_target=(0, -3),
            target_assigned_tick=1,
        )
        tactic = BalancedTactic(memory=memory)

        reasons = []
        for tick in range(1, 5):
            tactic.choose_actions(
                make_turn(
                    tick=tick,
                    core=core,
                    units=(vanguard, ranger),
                    resources=0,
                )
            )
            reasons.append(
                next(
                    task["reason"]
                    for task in tactic.last_decision_trace["tasks"]
                    if task["actor_id"] == str(vanguard.id)
                )
            )

        self.assertLessEqual(
            max(
                (
                    len(list(run))
                    for reason, run in groupby(reasons)
                    if reason in {
                        "VANGUARD_ANCHOR_HOLD",
                        "WAIT_FOR_PARTNER_PROGRESS",
                    }
                ),
                default=0,
            ),
            2,
        )
        self.assertNotEqual(memory.squad_states[key].patrol_anchor, vanguard.position)

    def test_broken_pair_is_not_recreated_during_pairing_cooldown(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (5, 0))
        ranger = unit(2, UnitType.RANGER, (0, 1))
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))
        key = (vanguard.id, ranger.id)

        for tick in range(1, 7):
            tactic.choose_actions(
                make_turn(
                    tick=tick,
                    units=(vanguard, ranger),
                    resources=0,
                )
            )

        self.assertNotIn(key, tactic.memory.squad_states)
        cooldowns = getattr(tactic.memory, "squad_pairing_cooldowns", {})
        self.assertGreaterEqual(cooldowns[key].expires_tick, 8)

    def test_home_defense_assigns_targets_instead_of_generic_pool_waits(self) -> None:
        combatants = tuple(
            unit(index, UnitType.VANGUARD, (index - 4, 0))
            for index in range(1, 7)
        ) + tuple(
            unit(index + 20, UnitType.RANGER, (index - 4, 1))
            for index in range(1, 7)
        )
        enemy = unit(100, UnitType.VANGUARD, (0, -10), controlled=False)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-12, 13)
            for y in range(-12, 13)
            if abs(x) + abs(y) <= 12
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(units=combatants, enemies=(enemy,), resources=0)
        world = build_world_model(turn, memory, tactic.config)
        projection = build_tactical_map(world, tactic.config)

        intents = tactic._kernel.defense.intents(
            world,
            projection,
            frozenset(manhattan_ring((0, 0), 5)),
        )

        self.assertFalse(
            any(intent.reason == "DEFENSE_POOL_RESERVE" for intent in intents)
        )
        assigned = {intent.actor_id for intent in intents if intent.actor_id is not None}
        self.assertEqual(assigned, {unit_view.id for unit_view in combatants})

    def test_outer_screen_ranger_does_not_take_the_zero_risk_contact_losing_step(self) -> None:
        core = friendly_core(position=(405, -156))
        vanguards = (
            unit(1, UnitType.VANGUARD, (400, -146)),
            unit(2, UnitType.VANGUARD, (402, -146)),
        )
        rangers = (
            unit(3, UnitType.RANGER, (401, -147)),
            unit(4, UnitType.RANGER, (404, -142)),
        )
        enemy = unit(100, UnitType.RANGER, (400, -143), controlled=False)
        memory = TacticMemory(
            opening_complete=True,
            core_id=core.id,
            core_position=core.position,
        )
        memory.screening_groups[enemy.id] = ScreeningGroupState(
            target_id=enemy.id,
            vanguard_ids=(vanguards[0].id, vanguards[1].id),
            ranger_ids=(rangers[0].id, rangers[1].id),
            started_tick=1,
            last_seen_tick=1,
            last_distance=18,
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(
            tick=2,
            core=core,
            units=(*vanguards, *rangers),
            enemies=(enemy,),
            obstacle_cells=(
                (400, -145),
                (402, -145),
                (402, -138),
                (404, -138),
                (405, -139),
                (406, -143),
                (406, -142),
                (407, -140),
                (412, -141),
                (412, -139),
                (414, -140),
            ),
            resources=0,
        )

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[rangers[1].id]
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(rangers[1].id)
        )
        self.assertFalse(
            isinstance(action, MoveAction) and action.direction is Direction.RIGHT,
            "the zero-risk move leaves the target-facing side of the screen",
        )
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.LEFT)
        self.assertNotEqual(task["reason"], "OUTER_SCREEN_FIRE_SUPPORT")
        contact = tactic.last_decision_trace["combat"]["screening_contact"][0]
        self.assertGreaterEqual(contact["visible_after"], contact["visible_before"])
        self.assertEqual(
            contact["options"][0]["first_position"],
            [403, -142],
        )

        advanced_rangers = (
            rangers[0],
            unit(4, UnitType.RANGER, (403, -142)),
        )
        follow_up = make_turn(
            tick=3,
            core=core,
            units=(*vanguards, *advanced_rangers),
            enemies=(enemy,),
            obstacle_cells=(
                (400, -145),
                (402, -145),
                (402, -138),
                (404, -138),
                (405, -139),
                (406, -143),
                (406, -142),
                (407, -140),
                (412, -141),
                (412, -139),
                (414, -140),
            ),
            resources=0,
        )
        tactic.choose_actions(follow_up)
        follow_up_action = follow_up.plan.unit_actions[advanced_rangers[1].id]
        self.assertFalse(
            isinstance(follow_up_action, MoveAction)
            and follow_up_action.direction is Direction.RIGHT,
            "the contact lease must not immediately reverse a useful advance",
        )

    def test_stalled_reassembly_moves_the_holder_after_two_ticks(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (5, 0))
        ranger = unit(2, UnitType.RANGER, (0, 1))
        tactic = BalancedTactic()

        for tick in range(1, 4):
            turn = make_turn(
                tick=tick,
                units=(vanguard, ranger),
                resources=0,
            )
            tactic.choose_actions(turn)

        vanguard_task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(vanguard.id)
        )
        self.assertIsInstance(turn.plan.unit_actions[vanguard.id], MoveAction)
        self.assertEqual(vanguard_task["reason"], "SQUAD_REASSEMBLE_RENDEZVOUS")

    def test_ranger_support_conflict_does_not_fall_back_into_a_dead_end(self) -> None:
        ranger = unit(1, UnitType.RANGER, (0, 0))
        blocker = unit(2, UnitType.VANGUARD, (-1, -1))
        core = friendly_core(position=(5, 5))
        memory = TacticMemory(
            opening_complete=True,
            core_id=core.id,
            core_position=core.position,
        )
        memory.squad_states[(blocker.id, ranger.id)] = SquadState(
            vanguard_id=blocker.id,
            ranger_id=ranger.id,
            radius=5,
            sector_index=0,
        )
        turn = make_turn(
            core=core,
            units=(ranger, blocker),
            obstacle_cells=((-1, 1), (1, 1), (0, 2)),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)
        world = build_world_model(turn, memory, tactic.config)
        projection = build_tactical_map(world, tactic.config)
        support = tactic._kernel.defense._move_or_wait(
            world,
            projection,
            world.friendly(ranger.id),
            (-2, 0),
            frozenset({(0, -1), (1, 0)}),
            "RANGER_SUPPORT",
        )
        claim = ActionIntent.move(
            blocker.id,
            UnitMission.PATROL,
            1,
            Direction.DOWN,
            (-1, 0),
            exclusive_destination=True,
            reason="TEST_RESERVATION",
        )

        resolution = IntentResolver().resolve(world, [claim, *support])
        tactic._kernel.defense.observe_resolution(world, resolution)

        selected = resolution.for_actor(ranger.id)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.action.value, "WAIT")
        self.assertNotEqual(selected.direction, Direction.DOWN)
        self.assertIn(
            tactic.memory.formation_move_feedback[ranger.id].rejection_reason,
            {"PHYSICAL_CELL_CAPACITY", "COMBAT_UNIT_EXCLUSIVE"},
        )

    def test_repeated_formation_conflict_uses_safe_non_reversing_yield(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        blocker = unit(2, UnitType.RANGER, (1, 0))
        core = friendly_core(position=(5, 5))
        memory = TacticMemory(
            opening_complete=True,
            core_id=core.id,
            core_position=core.position,
        )
        memory.known_passable.update(
            (x, y)
            for x in range(-2, 7)
            for y in range(-3, 7)
        )
        memory.position_history[vanguard.id] = ((0, -1), vanguard.position)
        memory.formation_move_feedback[vanguard.id] = FormationMoveFeedback(
            actor_id=vanguard.id,
            tick=1,
            action="WAIT",
            reason="VANGUARD_ANCHOR_ROUTE_BLOCKED_THIS_TICK",
            target_position=(2, 0),
            rejection_reason="CELL_CAPACITY",
            consecutive_blocked_ticks=2,
        )
        tactic = BalancedTactic(memory=memory)
        world = build_world_model(
            make_turn(
                tick=2,
                core=core,
                units=(vanguard, blocker),
                resources=0,
            ),
            memory,
            tactic.config,
        )
        projection = build_tactical_map(world, tactic.config)

        candidates = tactic._kernel.defense._move_or_wait(
            world,
            projection,
            world.friendly(vanguard.id),
            (2, 0),
            frozenset(),
            "VANGUARD_ANCHOR",
        )
        resolution = IntentResolver().resolve(world, candidates)

        selected = resolution.for_actor(vanguard.id)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.reason, "FORMATION_YIELD")
        self.assertEqual(selected.target_position, (0, 1))
        self.assertNotEqual(selected.target_position, (0, -1))
        self.assertGreaterEqual(dict(selected.metadata)["forward_exits"], 2)

    def test_home_vanguards_receive_unique_intercept_cells(self) -> None:
        vanguards = (
            unit(1, UnitType.VANGUARD, (-2, -2)),
            unit(2, UnitType.VANGUARD, (2, -2)),
            unit(3, UnitType.VANGUARD, (-3, 0)),
            unit(4, UnitType.VANGUARD, (3, 0)),
            unit(5, UnitType.VANGUARD, (0, -2)),
            unit(6, UnitType.VANGUARD, (1, -2)),
        )
        enemies = (
            unit(900, UnitType.VANGUARD, (1, -5), controlled=False),
            unit(901, UnitType.VANGUARD, (2, -5), controlled=False),
            unit(902, UnitType.VANGUARD, (0, -6), controlled=False),
        )
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-8, 9)
            for y in range(-8, 9)
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(units=vanguards, enemies=enemies, resources=0)
        )

        tasks = tactic.last_decision_trace["combat"]["home_vanguard_assignment"]["tasks"]
        intercepts = [tuple(row["intercept_cell"]) for row in tasks]
        self.assertEqual(len(intercepts), len(set(intercepts)))
        self.assertLessEqual(len(tasks), len(enemies) * 2)

    def test_vanguard_intercept_line_is_leased_across_prediction_noise(self) -> None:
        vanguards = (
            unit(1, UnitType.VANGUARD, (-2, -2)),
            unit(2, UnitType.VANGUARD, (2, -2)),
        )
        target = unit(900, UnitType.VANGUARD, (0, -6), controlled=False)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y) for x in range(-9, 10) for y in range(-9, 10)
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=1, units=vanguards, enemies=(target,), resources=0)
        )
        first = {
            row["vanguard_id"]: tuple(row["intercept_cell"])
            for row in tactic.last_decision_trace["combat"][
                "home_vanguard_assignment"
            ]["tasks"]
        }
        tactic.choose_actions(
            make_turn(tick=2, units=vanguards, enemies=(target,), resources=0)
        )
        second_rows = tactic.last_decision_trace["combat"][
            "home_vanguard_assignment"
        ]["tasks"]
        second = {
            row["vanguard_id"]: tuple(row["intercept_cell"])
            for row in second_rows
        }

        self.assertEqual(second, first)
        self.assertTrue(second_rows)
        self.assertTrue(
            all(row["lease"]["assigned_tick"] == 1 for row in second_rows)
        )

    def test_urgent_combat_target_suppresses_nonurgent_worker_fire(self) -> None:
        rangers = (
            unit(1, UnitType.RANGER, (13, 0)),
            unit(2, UnitType.RANGER, (10, 3)),
            unit(3, UnitType.RANGER, (10, -3)),
        )
        enemies = (
            unit(100, UnitType.VANGUARD, (10, 0), controlled=False),
            unit(101, UnitType.WORKER, (16, 0), controlled=False),
        )
        turn = make_turn(
            core=friendly_core(position=(0, 0)),
            units=rangers,
            enemies=enemies,
            resources=0,
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        worker_cell = enemies[1].position
        self.assertFalse(
            any(
                isinstance(action, ShootAction)
                and action.expected_cell == worker_cell
                for action in turn.plan.unit_actions.values()
            )
        )

    def test_remote_ranger_may_fire_at_worker_when_it_cannot_reinforce(self) -> None:
        ranger = unit(1, UnitType.RANGER, (27, 0))
        enemies = (
            unit(100, UnitType.VANGUARD, (10, 0), controlled=False),
            unit(101, UnitType.WORKER, (30, 0), controlled=False),
        )
        turn = make_turn(
            core=friendly_core(position=(0, 0)),
            units=(ranger,),
            enemies=enemies,
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[ranger.id]
        self.assertIsInstance(action, ShootAction)
        self.assertEqual(action.expected_cell, enemies[1].position)

    def test_global_vanguard_assignment_uses_nearest_defender_before_uuid_order(self) -> None:
        vanguards = (
            unit(1, UnitType.VANGUARD, (1, -5)),
            unit(2, UnitType.VANGUARD, (6, 0)),
            unit(3, UnitType.VANGUARD, (-6, 0)),
            unit(4, UnitType.VANGUARD, (0, -9)),
        )
        enemy = unit(900, UnitType.VANGUARD, (0, -13), controlled=False)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-16, 17)
            for y in range(-16, 17)
            if abs(x) + abs(y) <= 16
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(units=vanguards, enemies=(enemy,), resources=0)
        )

        nearest_task = next(
            task
            for task in tactic.last_decision_trace["tasks"]
            if task["actor_id"] == str(vanguards[-1].id)
        )
        self.assertEqual(nearest_task["mission"], "ATTACK")
        self.assertNotIn("SECTOR_RESERVE", nearest_task["reason"])

    def test_vanguard_sweeps_a_multi_enemy_candidate_convergence_cell(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        enemies = (
            unit(100, UnitType.VANGUARD, (-1, -1), controlled=False),
            unit(101, UnitType.VANGUARD, (-1, 1), controlled=False),
        )

        turn = make_turn(units=(vanguard,), enemies=enemies, resources=0)
        tactic = BalancedTactic()
        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[vanguard.id]
        self.assertIsInstance(action, SweepAction)
        self.assertEqual(action.direction, Direction.LEFT)

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
        stances = [tuple(task["metadata"]["firing_stance"]) for task in advancing]
        self.assertEqual(len(stances), len(set(stances)))

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
        enemy = unit(100, UnitType.RANGER, (17, 0), controlled=False)
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

    def test_home_defenders_counter_siege_exposed_enemy_core_after_units_clear(self) -> None:
        defenders = (
            unit(1, UnitType.VANGUARD, (4, 0)),
            unit(2, UnitType.RANGER, (3, 0)),
            unit(3, UnitType.VANGUARD, (0, 1)),
            unit(4, UnitType.RANGER, (0, -1)),
        )
        hostile_core = enemy_core(500, (5, 0), shield=0)
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))

        tactic.choose_actions(
            make_turn(
                tick=1,
                units=defenders,
                enemies=(
                    hostile_core,
                    unit(600, UnitType.VANGUARD, (5, 1), controlled=False),
                ),
                resources=0,
            )
        )
        turn = make_turn(
            tick=2,
            units=defenders,
            enemies=(hostile_core,),
            resources=0,
        )

        tactic.choose_actions(turn)

        attacks = tuple(
            task
            for task in tactic.last_decision_trace["tasks"]
            if task["target_id"] == str(hostile_core.id)
        )
        self.assertTrue(attacks)
        self.assertTrue(
            any(task["reason"].startswith("COUNTER_SIEGE") for task in attacks)
        )
        counter = tactic.last_decision_trace["combat"]["counter_siege"]
        self.assertEqual(counter["phase"], "PRESSING")
        self.assertEqual(counter["target_id"], str(hostile_core.id))

    def test_counter_siege_releases_destroyed_enemy_core_immediately(self) -> None:
        defenders = (
            unit(1, UnitType.VANGUARD, (4, 0)),
            unit(2, UnitType.RANGER, (3, 0)),
            unit(3, UnitType.VANGUARD, (0, 1)),
            unit(4, UnitType.RANGER, (0, -1)),
        )
        hostile_core = enemy_core(500, (5, 0), shield=0)
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))
        tactic.choose_actions(
            make_turn(
                tick=1,
                units=defenders,
                enemies=(
                    hostile_core,
                    unit(600, UnitType.VANGUARD, (5, 1), controlled=False),
                ),
                resources=0,
            )
        )
        destroyed = ResolutionEvent(
            event_id=uid(901),
            tick=2,
            event_type="DESTRUCTION_PARTICIPATION",
            target_id=hostile_core.id,
            reason_code="CORE",
        )

        tactic.choose_actions(
            make_turn(tick=2, units=defenders, events=(destroyed,), resources=0)
        )

        self.assertEqual(
            tactic.last_decision_trace["combat"]["counter_siege"]["phase"],
            "IDLE",
        )

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
        units[0] = unit(10, UnitType.VANGUARD, (13, 0))
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
        target = unit(200, UnitType.VANGUARD, (18, 0), controlled=False)

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
                    "ADVANCE_TO_DYNAMIC_FIRE_LINE",
                    "HOLD_CONTACT",
                    "REACQUIRE_CONTACT",
                    "CONTACT_REPOSITION_BLOCKED",
                }
                for task in group_tasks
            )
        )
        home_tasks = [
            task
            for task in tactic.last_decision_trace["tasks"]
            if task["actor_id"] not in group_ids
            and task["actor_id"]
            in {str(item.id) for item in units}
        ]
        self.assertTrue(home_tasks)
        self.assertTrue(all(task["mission"] == "PATROL" for task in home_tasks))
        trigger = tactic.last_decision_trace["combat"]["defense_trigger"]
        self.assertFalse(trigger["global_home_pool"])
        self.assertEqual(
            trigger["outer_screen_only"][0]["reason"],
            "REMOTE_LOCAL_SCREEN",
        )

    def test_peaceful_formation_rejects_cross_map_member_routes(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (30, 0))
        ranger = unit(2, UnitType.RANGER, (30, 2))
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-20, 35)
            for y in range(-20, 21)
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(units=(vanguard, ranger), resources=0)
        )

        bundles = tactic.last_decision_trace["combat"]["formation"]["assignment"]["bundles"]
        self.assertTrue(
            all(
                bundle["vanguard_route_distance"] <= 12
                and bundle["ranger_route_distance"] <= 12
                for bundle in bundles
            )
        )
        tasks = [
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] in {str(vanguard.id), str(ranger.id)}
        ]
        self.assertTrue(all(item["reason"] != "WAIT_FOR_PARTNER_PROGRESS" for item in tasks))

    def test_outer_screen_members_survive_home_handoff_at_radius_thirteen(self) -> None:
        unit_rows = list(
            [unit(10 + index, UnitType.VANGUARD, (index - 3, 5)) for index in range(6)]
            + [unit(30 + index, UnitType.RANGER, (index - 3, -5)) for index in range(6)]
        )
        unit_rows[0] = unit(10, UnitType.VANGUARD, (12, 0))
        unit_rows[1] = unit(11, UnitType.VANGUARD, (11, 1))
        unit_rows[6] = unit(30, UnitType.RANGER, (12, 2))
        unit_rows[7] = unit(31, UnitType.RANGER, (11, -1))
        units = tuple(unit_rows)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-35, 36)
            for y in range(-35, 36)
            if abs(x) + abs(y) <= 35
        )
        tactic = BalancedTactic(memory=memory)
        target_id = 200
        tactic.choose_actions(
            make_turn(
                tick=1,
                units=units,
                enemies=(unit(target_id, UnitType.VANGUARD, (17, 0), controlled=False),),
                resources=0,
            )
        )
        original = tactic.memory.screening_groups[
            unit(target_id, UnitType.VANGUARD, (17, 0), controlled=False).id
        ]

        tactic.choose_actions(
            make_turn(
                tick=2,
                units=units,
                enemies=(unit(target_id, UnitType.VANGUARD, (13, 0), controlled=False),),
                resources=0,
            )
        )

        retained = tactic.memory.screening_groups[original.target_id]
        self.assertEqual(retained.phase, "HOME_HANDOFF")
        self.assertEqual(retained.vanguard_ids, original.vanguard_ids)
        assignments = tactic.last_decision_trace["combat"]["home_vanguard_assignment"]["tasks"]
        handoff_ids = {
            row["vanguard_id"]
            for row in assignments
            if row["phase"] == "HOME_HANDOFF"
        }
        self.assertEqual(handoff_ids, {str(item) for item in original.vanguard_ids})

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

    def test_vanguard_prediction_uses_the_opening_in_a_blocking_wall(self) -> None:
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-8, 9)
            for y in range(-8, 9)
        )
        wall = tuple((x, -3) for x in range(-3, 2))
        memory.known_obstacles.update(wall)
        ranger = unit(1, UnitType.RANGER, (3, -2))
        enemy = unit(100, UnitType.VANGUARD, (0, -4), controlled=False)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(
                units=(ranger,),
                enemies=(enemy,),
                obstacle_cells=wall,
                resources=0,
            )
        )

        estimate = tactic.last_decision_trace["combat"]["enemy_action_candidates"][0]
        self.assertEqual(estimate["ranked_cells"][0], [1, -4])
        self.assertIn("OBSTACLE_AWARE_APPROACH", estimate["evidence"])

    def test_full_health_vanguard_accepts_one_nonfatal_hit_to_improve_intercept(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (2, 0))
        enemy = unit(100, UnitType.VANGUARD, (0, 0), controlled=False)
        tactic = BalancedTactic()
        turn = make_turn(
            tick=1,
            core=friendly_core(position=(0, 5)),
            units=(vanguard,),
            enemies=(enemy,),
            obstacle_cells=((2, -1), (2, 1), (3, 0)),
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
        self.assertEqual(ranger_task["reason"], "SQUAD_REASSEMBLE_RENDEZVOUS")
        self.assertEqual(vanguard_task["reason"], "SQUAD_REASSEMBLE_RENDEZVOUS")
        self.assertIsInstance(turn.plan.unit_actions[vanguard.id], MoveAction)

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
        ranger_positions = []
        vanguard_positions = []
        patrol_leases = []

        for tick in range(1, 15):
            vanguard = unit(1, UnitType.VANGUARD, vanguard_position)
            ranger = unit(2, UnitType.RANGER, ranger_position)
            vanguard_positions.append(vanguard_position)
            ranger_positions.append(ranger_position)
            turn = make_turn(
                tick=tick,
                units=(vanguard, ranger),
                resources=0,
            )
            tactic.choose_actions(turn)
            squad = tactic.memory.squad_states[(vanguard.id, ranger.id)]
            patrol_leases.append((squad.patrol_anchor, squad.support_target))
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
        self.assertGreater(len(set(patrol_leases)), 1)
        self.assertFalse(
            any(
                ranger_positions[index] == ranger_positions[index - 2]
                == ranger_positions[index - 4]
                and ranger_positions[index - 1] == ranger_positions[index - 3]
                and ranger_positions[index] != ranger_positions[index - 1]
                for index in range(4, len(ranger_positions))
            ),
            "a peaceful Ranger must not remain in a two-cell support loop",
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
