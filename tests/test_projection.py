from __future__ import annotations

import unittest

from arena_hero import Direction, MoveAction, UnitType

from arena_tactic import BalancedTactic, TacticMemory, build_projected_turn
from arena_tactic.world import build_world_model
from tests.helpers import friendly_core, make_turn, unit


class TacticalMapTests(unittest.TestCase):
    def test_global_vision_records_core_and_every_unit_contributor(self) -> None:
        worker = unit(1, UnitType.WORKER, (8, 0))
        vanguard = unit(2, UnitType.VANGUARD, (16, 0))
        ranger = unit(3, UnitType.RANGER, (24, 0))
        world = build_world_model(
            make_turn(
                core=friendly_core(position=(0, 0)),
                units=(worker, vanguard, ranger),
            ),
            TacticMemory(),
        )

        projection = build_projected_turn(world)

        self.assertEqual(
            {source.actor_kind for source in projection.vision_sources},
            {"CORE", "WORKER", "VANGUARD", "RANGER"},
        )
        for source in projection.vision_sources:
            self.assertIn(source.actor_id, projection.observers(source.position))
        self.assertEqual(projection.visible_cells, world.visible_cells)
        self.assertEqual(
            projection.last_visible_ticks,
            dict(world.cell_last_visible),
        )

    def test_each_contributor_respects_shared_permanent_occlusion(self) -> None:
        ranger = unit(3, UnitType.RANGER, (0, 0))
        world = build_world_model(
            make_turn(
                core=friendly_core(position=(20, 20)),
                units=(ranger,),
                obstacle_cells=((1, 0),),
            ),
            TacticMemory(),
        )

        projection = build_projected_turn(world)
        ranger_source = next(
            source for source in projection.vision_sources if source.actor_id == ranger.id
        )

        self.assertIn((1, 0), ranger_source.visible_cells)
        self.assertNotIn((2, 0), ranger_source.visible_cells)

    def test_resource_seen_only_by_ranger_is_assigned_to_worker(self) -> None:
        worker = unit(1, UnitType.WORKER, (1, 0))
        ranger = unit(2, UnitType.RANGER, (6, 0))
        turn = make_turn(
            core=friendly_core(position=(0, 0)),
            units=(worker, ranger),
            resources=0,
            resource_cells=((10, 0),),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        world = tactic._last_world
        assert world is not None
        self.assertNotIn(worker.id, world.observers((10, 0)))
        self.assertIn(ranger.id, world.observers((10, 0)))
        self.assertEqual(tactic.memory.unit_missions[worker.id].target, (10, 0))
        tactical_map = tactic.last_tactical_map
        assert tactical_map is not None
        resource = tactical_map.resource((10, 0))
        assert resource is not None
        self.assertEqual(resource.assigned_worker_ids, (worker.id,))
        self.assertEqual(tactical_map.planned_positions[worker.id], (2, 0))
        self.assertIn((2, 0), tactical_map.reserved_positions)
        self.assertTrue(tactical_map.service_positions)
        selected = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertEqual(selected["mission"], "HARVEST")
        self.assertEqual(selected["reason"], "GLOBAL_RESOURCE_MATCH")

    def test_worker_uses_shared_threat_as_risk_without_unrelated_escape(self) -> None:
        worker = unit(1, UnitType.WORKER, (8, 0))
        ranger = unit(2, UnitType.RANGER, (4, 0))
        enemy = unit(100, UnitType.RANGER, (4, 4), controlled=False)
        turn = make_turn(
            core=friendly_core(position=(20, 0)),
            units=(worker, ranger),
            enemies=(enemy,),
            resources=0,
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        world = tactic._last_world
        assert world is not None
        self.assertNotIn(worker.id, world.observers(enemy.position))
        self.assertIn(ranger.id, world.observers(enemy.position))
        self.assertNotIn(worker.id, tactic.memory.worker_escape_states)
        tactical_map = tactic.last_tactical_map
        assert tactical_map is not None
        self.assertIsNotNone(tactical_map.enemy(enemy.id))
        selected = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertNotEqual(selected["mission"], "ESCAPE")

    def test_worker_escapes_a_shared_threat_within_two_worker_vision_radii(self) -> None:
        worker = unit(1, UnitType.WORKER, (8, 0))
        ranger = unit(2, UnitType.RANGER, (3, 0))
        enemy = unit(100, UnitType.RANGER, (3, 1), controlled=False)
        turn = make_turn(
            core=friendly_core(position=(20, 0)),
            units=(worker, ranger),
            enemies=(enemy,),
            resources=0,
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        world = tactic._last_world
        assert world is not None
        self.assertNotIn(worker.id, world.observers(enemy.position))
        self.assertIn(ranger.id, world.observers(enemy.position))
        escape = tactic.memory.worker_escape_states[worker.id]
        self.assertEqual(escape.phase, "GLOBAL_ALERT_RETREAT")
        selected = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertEqual(selected["mission"], "ESCAPE")

    def test_remote_enemy_does_not_reverse_an_unrelated_cargo_return(self) -> None:
        worker = unit(1, UnitType.WORKER, (10, 0), cargo=1)
        observer = unit(2, UnitType.RANGER, (-17, 0))
        enemy = unit(100, UnitType.VANGUARD, (-20, 0), controlled=False)
        turn = make_turn(
            core=friendly_core(position=(0, 0)),
            units=(worker, observer),
            enemies=(enemy,),
            resources=0,
        )
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-24, 13)
            for y in range(-4, 5)
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.LEFT)
        queue = tactic.last_decision_trace["economy"]["service_queue"]
        reservation = next(
            row
            for row in queue["return_reservations"]
            if row["worker_id"] == str(worker.id)
        )
        self.assertEqual(tuple(reservation["first_position"]), (9, 0))

    def test_worker_discovery_triggers_combat_response(self) -> None:
        worker = unit(1, UnitType.WORKER, (10, 0))
        vanguard = unit(2, UnitType.VANGUARD, (4, 0))
        ranger = unit(3, UnitType.RANGER, (5, 1))
        enemy = unit(100, UnitType.VANGUARD, (12, 0), controlled=False)
        turn = make_turn(
            units=(worker, vanguard, ranger),
            enemies=(enemy,),
            resources=0,
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        world = tactic._last_world
        assert world is not None
        self.assertIn(worker.id, world.observers(enemy.position))
        self.assertNotIn(vanguard.id, world.observers(enemy.position))
        selected = {
            item["actor_id"]: item
            for item in tactic.last_decision_trace["tasks"]
        }
        self.assertIn(
            selected[str(vanguard.id)]["mission"],
            {"ATTACK", "HOME_DEFENSE"},
        )
        self.assertEqual(selected[str(ranger.id)]["mission"], "HOME_DEFENSE")

    def test_core_pressure_consumes_enemies_discovered_outside_core_vision(self) -> None:
        scout = unit(1, UnitType.WORKER, (9, 0))
        enemies = (
            unit(100, UnitType.VANGUARD, (10, 0), controlled=False),
            unit(101, UnitType.RANGER, (10, 1), controlled=False),
        )
        turn = make_turn(units=(scout,), enemies=enemies, resources=0)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        world = tactic._last_world
        assert world is not None and world.core is not None
        self.assertNotIn(world.core.id, world.observers((10, 0)))
        self.assertIn(scout.id, world.observers((10, 0)))
        core_task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] is None
        )
        self.assertEqual(core_task["mission"], "CORE_SURVIVAL")
        self.assertEqual(core_task["action"], "START_MOVE")

    def test_fogged_enemy_remains_uncertain_global_intelligence(self) -> None:
        memory = TacticMemory()
        ranger = unit(2, UnitType.RANGER, (4, 0))
        enemy = unit(100, UnitType.VANGUARD, (4, 4), controlled=False)
        build_world_model(
            make_turn(tick=1, units=(ranger,), enemies=(enemy,)),
            memory,
        )
        world = build_world_model(
            make_turn(tick=2, units=(ranger,)),
            memory,
        )

        projection = build_projected_turn(world)
        intel = projection.enemy(enemy.id)

        assert intel is not None
        self.assertFalse(intel.visible_now)
        self.assertEqual(intel.last_seen_tick, 1)
        self.assertEqual(intel.age, 1)
        self.assertGreater(len(intel.movement_corridor), 1)
        self.assertNotIn(enemy.position, projection.hostile_occupied)

    def test_visible_enemy_attack_horizons_do_not_assume_move_and_attack(self) -> None:
        enemy = unit(100, UnitType.VANGUARD, (2, 0), controlled=False)
        world = build_world_model(make_turn(enemies=(enemy,)), TacticMemory())

        projection = build_projected_turn(world)

        self.assertEqual(projection.immediate_attackers((0, 0)), 0)
        self.assertEqual(projection.future_attackers((0, 0)), 1)

    def test_future_ranger_risk_includes_terrain_legal_one_step_origins(self) -> None:
        enemy = unit(100, UnitType.RANGER, (0, 3), controlled=False)
        world = build_world_model(
            make_turn(enemies=(enemy,), obstacle_cells=((0, 2),)),
            TacticMemory(),
        )

        projection = build_projected_turn(world)

        self.assertEqual(projection.immediate_attackers((1, 0)), 0)
        self.assertEqual(projection.future_attackers((1, 0)), 1)
        enemy_projection = projection.enemy(enemy.id)
        assert enemy_projection is not None
        self.assertNotIn((0, 2), enemy_projection.possible_positions)

    def test_enemy_prediction_excludes_a_currently_full_destination(self) -> None:
        blockers = (
            unit(1, UnitType.WORKER, (1, 0)),
            unit(2, UnitType.WORKER, (1, 0)),
        )
        enemy = unit(100, UnitType.RANGER, (0, 0), controlled=False)
        world = build_world_model(
            make_turn(
                core=friendly_core(position=(10, 10)),
                units=blockers,
                enemies=(enemy,),
            ),
            TacticMemory(),
        )

        projection = build_projected_turn(world)

        enemy_projection = projection.enemy(enemy.id)
        assert enemy_projection is not None
        self.assertNotIn((1, 0), enemy_projection.possible_positions)

    def test_enemy_worker_projection_includes_wait_and_legal_steps(self) -> None:
        enemy = unit(100, UnitType.WORKER, (0, 0), controlled=False)
        world = build_world_model(
            make_turn(
                core=friendly_core(position=(10, 10)),
                enemies=(enemy,),
                obstacle_cells=((0, 1),),
            ),
            TacticMemory(),
        )

        projection = build_projected_turn(world)

        enemy_projection = projection.enemy(enemy.id)
        assert enemy_projection is not None
        self.assertEqual(
            set(enemy_projection.possible_positions),
            {(0, 0), (-1, 0), (1, 0), (0, -1)},
        )
        self.assertEqual(enemy_projection.immediate_attack_cells, frozenset())
        self.assertEqual(enemy_projection.future_attack_cells, frozenset())

    def test_finishing_core_migration_has_one_shared_projected_position(self) -> None:
        core = friendly_core(
            moving=True,
            direction=Direction.RIGHT,
            progress=3,
        )
        world = build_world_model(make_turn(core=core), TacticMemory())

        projection = build_projected_turn(world)

        self.assertTrue(projection.core_completes_move)
        self.assertEqual(projection.projected_core_position, (1, 0))


if __name__ == "__main__":
    unittest.main()
