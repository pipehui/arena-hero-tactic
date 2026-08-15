from __future__ import annotations

from collections import Counter
import math
import unittest

from arena_hero import (
    DepositAction,
    Direction,
    HealAction,
    MoveAction,
    ResolutionEvent,
    SpawnAction,
    UnitType,
    WaitAction,
)

from arena_tactic import (
    BalancedTactic,
    CrisisForceBaseline,
    PatientAdmissionProgress,
    ResourceWorkOrder,
    ScoutReturnRouteLease,
    TacticConfig,
    TacticMemory,
    ThreatHeatCell,
    UnitMission,
    WorkerEconomyMode,
    WorkerScoutPhase,
    WorkerScoutState,
)
from arena_tactic.models import MissionState, WorkerEscapeState
from arena_tactic.geometry import add_direction, manhattan
from arena_tactic.planning import weighted_route_to
from arena_tactic.resource_allocator import minimum_cost_matching
from arena_tactic.world import build_world_model
from tests.helpers import enemy_core, friendly_core, make_turn, uid, unit


class EconomyAndLogisticsTests(unittest.TestCase):
    def test_wounded_ranger_uses_full_health_ranger_cell_on_service_route(self) -> None:
        core = friendly_core(position=(3, 0))
        patient = unit(1, UnitType.RANGER, (0, 0), hp=1)
        guard = unit(2, UnitType.RANGER, (1, 0))
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            opening_complete=True,
        )
        memory.known_passable.update({(0, 0), (1, 0), (2, 0), (3, 0)})
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(
            core=core,
            units=(patient, guard),
            obstacle_cells=((0, -1), (0, 1), (1, -1), (1, 1)),
            resources=2,
        )

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[patient.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.RIGHT)
        decision = next(
            row
            for row in tactic.last_decision_trace["decisions"]
            if row["actor_id"] == str(patient.id)
        )
        self.assertEqual(
            decision["final"]["destination_exclusivity"],
            "SERVICE_TRANSIT",
        )
        self.assertEqual(
            decision["service"]["transit"]["shared_with_id"],
            str(guard.id),
        )
        self.assertTrue(decision["service"]["transit_route"]["options"])

    def test_worker_disengages_from_enemy_core_zone_and_ignores_nearby_resource(self) -> None:
        core = friendly_core(position=(20, 0))
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        memory.known_passable.update(
            (x, y) for x in range(-3, 22) for y in range(-4, 5)
        )
        tactic = BalancedTactic(memory=memory)
        position = (1, 0)

        for tick in range(1, 5):
            worker = unit(1, UnitType.WORKER, position)
            turn = make_turn(
                tick=tick,
                core=core,
                units=(worker,),
                enemies=(enemy_core(99, (0, 0)),),
                resource_cells=((1, 0),),
                resources=0,
            )
            tactic.choose_actions(turn)
            action = turn.plan.unit_actions[worker.id]
            self.assertIsInstance(action, MoveAction)
            task = next(
                row
                for row in tactic.last_decision_trace["tasks"]
                if row["actor_id"] == str(worker.id)
            )
            self.assertEqual(task["mission"], "ESCAPE")
            dx, dy = action.direction.delta
            destination = position[0] + dx, position[1] + dy
            self.assertGreater(
                manhattan(destination, (0, 0)),
                manhattan(position, (0, 0)),
            )
            position = destination

        mission = tactic.memory.unit_missions.get(uid(1))
        self.assertTrue(mission is None or mission.mission is not UnitMission.HARVEST)

    def test_stale_enemy_core_intel_is_not_a_hard_worker_control_zone(self) -> None:
        core = friendly_core(position=(20, 0))
        worker = unit(1, UnitType.WORKER, (1, 0))
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        memory.known_passable.update(
            (x, y) for x in range(-3, 22) for y in range(-4, 5)
        )
        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=core,
                units=(worker,),
                enemies=(enemy_core(99, (0, 0)),),
                resources=0,
            )
        )
        turn = make_turn(
            tick=100,
            core=core,
            units=(unit(1, UnitType.WORKER, (7, 0)),),
            resources=0,
        )

        tactic.choose_actions(turn)

        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertNotEqual(task["mission"], "ESCAPE")
        self.assertIn(uid(99), tactic.memory.enemy_core_control_zones)
        zone = tactic.memory.enemy_core_control_zones[uid(99)]
        self.assertEqual(zone.control_level, "STRATEGIC")

    def test_recent_fogged_enemy_core_is_soft_risk_after_hard_window(self) -> None:
        core = friendly_core(position=(20, 0))
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        memory.known_passable.update(
            (x, y) for x in range(-3, 22) for y in range(-4, 5)
        )
        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=core,
                units=(unit(1, UnitType.WORKER, (1, 0)),),
                enemies=(enemy_core(99, (0, 0)),),
                resources=0,
            )
        )
        turn = make_turn(
            tick=20,
            core=core,
            units=(unit(1, UnitType.WORKER, (7, 0)),),
            resources=0,
        )

        tactic.choose_actions(turn)

        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(uid(1))
        )
        self.assertNotEqual(task["mission"], "ESCAPE")
        self.assertEqual(
            tactic.memory.enemy_core_control_zones[uid(99)].control_level,
            "SOFT",
        )

    def test_enemy_core_clearing_keeps_zone_sticky_and_detours_toward_core(
        self,
    ) -> None:
        """A Worker just outside the clear radius must not step back inside.

        This is the minimal form of the live ``525edcc5`` two-cell cycle: the
        first Tick forces the Worker out through the only outward exit, while
        the next Tick offers a lateral route around the remembered enemy Core.
        """

        core = friendly_core(position=(0, -10))
        worker_id = uid(1)
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        memory.known_passable.update(
            (x, y) for x in range(-4, 5) for y in range(-12, 13)
        )
        tactic = BalancedTactic(memory=memory)
        first = make_turn(
            tick=1,
            core=core,
            units=(unit(1, UnitType.WORKER, (0, 8), cargo=1),),
            enemies=(enemy_core(99, (0, 0)),),
            obstacle_cells=((-1, 8), (1, 8)),
            resources=0,
        )

        tactic.choose_actions(first)

        first_action = first.plan.unit_actions[worker_id]
        self.assertIsInstance(first_action, MoveAction)
        outside = add_direction((0, 8), first_action.direction)
        self.assertEqual(outside, (0, 9))
        second = make_turn(
            tick=2,
            core=core,
            units=(unit(1, UnitType.WORKER, outside, cargo=1),),
            obstacle_cells=((-1, 8), (1, 8)),
            resources=0,
        )

        tactic.choose_actions(second)

        second_action = second.plan.unit_actions[worker_id]
        self.assertIsInstance(second_action, MoveAction)
        destination = add_direction(outside, second_action.direction)
        self.assertGreater(
            manhattan(destination, (0, 0)),
            tactic.config.enemy_core_worker_clear_radius,
        )
        lease = tactic.memory.worker_escape_states[worker_id]
        self.assertEqual(lease.control_core_ids, (uid(99),))
        self.assertEqual(lease.control_centers, ((0, 0),))

    def test_destroyed_enemy_core_releases_controlled_resource_backoff(self) -> None:
        core = friendly_core(position=(20, 0))
        hostile_core = enemy_core(99, (0, 0))
        resource = (1, 0)
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        memory.unit_missions[uid(1)] = MissionState(
            UnitMission.HARVEST,
            resource,
            0,
        )
        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=core,
                units=(unit(1, UnitType.WORKER, resource),),
                enemies=(hostile_core,),
                resource_cells=(resource,),
                resources=0,
            )
        )
        self.assertIn(resource, tactic.memory.target_backoff_until)
        destroyed = ResolutionEvent(
            event_id=uid(901),
            tick=2,
            event_type="DESTRUCTION_PARTICIPATION",
            target_id=hostile_core.id,
            reason_code="CORE",
        )

        tactic.choose_actions(
            make_turn(
                tick=2,
                core=core,
                units=(unit(1, UnitType.WORKER, (2, 0)),),
                resource_cells=(resource,),
                events=(destroyed,),
                resources=0,
            )
        )

        self.assertNotIn(resource, tactic.memory.target_backoff_until)
        self.assertNotIn(uid(99), tactic.memory.enemy_core_control_zones)
        self.assertNotIn(uid(1), tactic.memory.worker_disengage_leases)

    def test_fog_retreat_does_not_undo_a_safe_visible_escape_step(self) -> None:
        core = friendly_core(position=(10, 0))
        worker_id = uid(1)
        tactic = BalancedTactic()
        visible = make_turn(
            tick=1,
            core=core,
            units=(unit(1, UnitType.WORKER, (0, 0), cargo=1),),
            enemies=(unit(100, UnitType.VANGUARD, (2, 0), controlled=False),),
            resources=0,
        )
        tactic.choose_actions(visible)
        self.assertEqual(visible.plan.unit_actions[worker_id].direction, Direction.LEFT)

        fogged = make_turn(
            tick=2,
            core=core,
            units=(unit(1, UnitType.WORKER, (-1, 0), cargo=1),),
            enemies=(),
            resources=0,
        )
        tactic.choose_actions(fogged)

        action = fogged.plan.unit_actions[worker_id]
        self.assertIsInstance(action, MoveAction)
        dx, dy = action.direction.delta
        destination = (-1 + dx, dy)
        # At age one the conservative distance is d(last_seen)-1.  Returning
        # right would reduce it from two to one and re-enter the pursuit lane.
        before = max(0, manhattan((-1, 0), (2, 0)) - 1)
        after = max(0, manhattan(destination, (2, 0)) - 1)
        self.assertGreaterEqual(after, before)
        # The hidden Vanguard lies on the direct road to Core.  Retreat should
        # route around it laterally, not keep marching straight away from home.
        self.assertEqual(destination[0], -1)

    def test_fog_retreat_does_not_wait_when_only_survivable_steps_are_homeward(
        self,
    ) -> None:
        """Regression for live Tick 107396 -> 107397.

        The visible escape step increases Ranger distance.  Once the Ranger
        enters fog, its conservative envelope covers both homeward exits, but
        each exit still has a non-fatal four-Tick survival continuation.  The
        fog-home gate must not delete both and replace motion with WAIT.
        """

        core = friendly_core(position=(405, -156))
        worker_id = uid(1)
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        memory.known_passable.update(
            (x, y)
            for x in range(388, 410)
            for y in range(-222, -210)
        )
        tactic = BalancedTactic(memory=memory)
        visible = make_turn(
            tick=107396,
            core=core,
            units=(unit(1, UnitType.WORKER, (392, -217)),),
            enemies=(
                unit(100, UnitType.RANGER, (390, -218), controlled=False),
            ),
            obstacle_cells=((389, -217), (392, -216), (394, -218)),
            resources=0,
        )

        tactic.choose_actions(visible)

        first = visible.plan.unit_actions[worker_id]
        self.assertIsInstance(first, MoveAction)
        first_position = add_direction((392, -217), first.direction)
        self.assertEqual(first_position, (393, -217))

        fogged = make_turn(
            tick=107397,
            core=core,
            units=(unit(1, UnitType.WORKER, first_position),),
            obstacle_cells=(
                (392, -216),
                (393, -214),
                (394, -218),
                (394, -215),
                (396, -217),
            ),
            resources=0,
        )

        tactic.choose_actions(fogged)

        action = fogged.plan.unit_actions[worker_id]
        self.assertIsInstance(action, MoveAction)
        destination = add_direction(first_position, action.direction)
        before = max(0, manhattan(first_position, (390, -218)) - 1)
        after = max(0, manhattan(destination, (390, -218)) - 1)
        self.assertGreaterEqual(after, before)
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker_id)
        )
        self.assertEqual(task["mission"], "ESCAPE")
        self.assertNotEqual(task["reason"], "NO_SURVIVABLE_ROUTE")
        self.assertGreater(task["metadata"]["survival_terminals"], 0)
        self.assertEqual(
            tactic.memory.worker_escape_states[worker_id].last_threat_tick,
            107396,
        )

    def test_fog_retreat_prefers_core_when_last_seen_enemy_is_behind(self) -> None:
        core = friendly_core(position=(10, 0))
        worker_id = uid(1)
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        memory.known_passable.update(
            (x, y)
            for x in range(-8, 13)
            for y in range(-6, 7)
        )
        tactic = BalancedTactic(memory=memory)
        visible = make_turn(
            tick=1,
            core=core,
            units=(unit(1, UnitType.WORKER, (0, 0)),),
            enemies=(
                unit(100, UnitType.RANGER, (-3, 0), controlled=False),
            ),
            resources=0,
        )

        tactic.choose_actions(visible)

        first = visible.plan.unit_actions[worker_id]
        self.assertIsInstance(first, MoveAction)
        position = add_direction((0, 0), first.direction)
        self.assertEqual(position, (1, 0))
        fogged = make_turn(
            tick=2,
            core=core,
            units=(unit(1, UnitType.WORKER, position),),
            resources=0,
        )

        tactic.choose_actions(fogged)

        action = fogged.plan.unit_actions[worker_id]
        self.assertIsInstance(action, MoveAction)
        destination = add_direction(position, action.direction)
        self.assertLess(
            manhattan(destination, core.position),
            manhattan(position, core.position),
        )
        self.assertGreaterEqual(
            manhattan(destination, (-3, 0)),
            manhattan(position, (-3, 0)),
        )

    def test_escape_loop_keeps_a_valid_waypoint_lease(self) -> None:
        core = friendly_core(position=(10, 0))
        worker = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        threat = unit(100, UnitType.VANGUARD, (2, 0), controlled=False)
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=core,
                units=(worker,),
                enemies=(threat,),
                resources=0,
            )
        )
        tactic.memory.position_history[worker.id] = (
            (0, 0),
            (-1, 0),
            (0, 0),
            (-1, 0),
            (0, 0),
        )
        tactic.memory.worker_escape_states[worker.id] = WorkerEscapeState(
            phase="FLEEING",
            threat_ids=(threat.id,),
            last_threat_tick=1,
            waypoint=(-4, 0),
            last_min_enemy_distance=2,
            route_version=1,
            waypoint_assigned_tick=1,
            waypoint_expires_tick=5,
        )

        tactic.choose_actions(
            make_turn(
                tick=2,
                core=core,
                units=(worker,),
                enemies=(threat,),
                resources=0,
            )
        )

        state = tactic.memory.worker_escape_states[worker.id]
        self.assertEqual(state.waypoint, (-4, 0))
        self.assertEqual(state.route_version, 1)

    def test_clear_core_only_requires_a_locally_viable_exit(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0))
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            opening_complete=True,
            worker_scout_states={
                worker.id: WorkerScoutState(
                    worker_id=worker.id,
                    slot=0,
                    sector_index=0,
                    stage=0,
                    phase=WorkerScoutPhase.SECTOR_SCOUT,
                    target=(300, 0),
                    assigned_tick=1,
                )
            },
        )
        memory.known_passable.update((x, 0) for x in range(0, 301))
        memory.known_passable.update({(-1, 0), (0, 1), (0, -1), (1, 1), (1, -1)})
        turn = make_turn(tick=2, units=(worker,), resources=0)

        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[worker.id], MoveAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertEqual(task["mission"], "CLEAR_CORE")
        self.assertNotEqual(task["reason"], "CORE_EXIT_BLOCKED")

    def test_wartime_existing_worker_stack_is_actively_separated(self) -> None:
        core = friendly_core(position=(0, 0))
        first = unit(1, UnitType.WORKER, (3, 0))
        second = unit(2, UnitType.WORKER, (3, 0))
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            home_defense_alert_until=10,
        )
        memory.known_passable.update(
            (x, y) for x in range(-5, 6) for y in range(-5, 6)
        )
        turn = make_turn(
            tick=8,
            core=core,
            units=(first, second),
            resources=0,
        )

        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(turn)

        actions = [turn.plan.unit_actions[first.id], turn.plan.unit_actions[second.id]]
        self.assertTrue(any(isinstance(action, MoveAction) for action in actions))
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] in {str(first.id), str(second.id)}
            and row["reason"] == "WARTIME_WORKER_DECONFLICT"
        )
        self.assertEqual(task["mission"], "DECONFLICT_CELL")

    def test_remote_scout_does_not_enter_a_known_three_wall_pocket(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0))
        goal = (0, 40)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(5, 0),
            opening_complete=True,
            known_passable={
                (0, 0),
                (0, 1),
                (-1, 0),
                (1, 0),
                (0, -1),
                goal,
            },
            known_obstacles={(-1, 1), (1, 1), (0, 2)},
            unit_missions={
                worker.id: MissionState(UnitMission.EXPLORE, goal, 1)
            },
            worker_scout_states={
                worker.id: WorkerScoutState(
                    worker_id=worker.id,
                    slot=0,
                    sector_index=0,
                    stage=0,
                    phase=WorkerScoutPhase.SECTOR_SCOUT,
                    target=goal,
                    assigned_tick=1,
                )
            },
        )
        turn = make_turn(
            tick=2,
            core=friendly_core(position=(5, 0)),
            units=(worker,),
            obstacle_cells=((-1, 1), (1, 1), (0, 2)),
            resources=0,
        )

        BalancedTactic(memory=memory).choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertNotEqual(getattr(action, "direction", None), Direction.DOWN)

    def test_worker_may_enter_a_dead_end_that_contains_its_resource(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0))
        turn = make_turn(
            core=friendly_core(position=(5, 0)),
            units=(worker,),
            resource_cells=((0, 1),),
            obstacle_cells=((-1, 1), (1, 1), (0, 2)),
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.DOWN)

    def test_worker_escape_does_not_choose_a_known_dead_end(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0))
        enemy = unit(100, UnitType.RANGER, (0, -3), controlled=False)
        turn = make_turn(
            core=friendly_core(position=(5, 5)),
            units=(worker,),
            enemies=(enemy,),
            obstacle_cells=((-1, 1), (1, 1), (0, 2)),
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertNotEqual(getattr(action, "direction", None), Direction.DOWN)

    def test_full_health_worker_rejects_zero_survival_terminal_in_live_ambush(self) -> None:
        core = friendly_core(position=(405, -156))
        worker_id = 1
        tactic = BalancedTactic(
            memory=TacticMemory(core_id=core.id, core_position=core.position)
        )
        frames = (
            (
                101944,
                (351, -152),
                ((351, -151),),
                ((100, UnitType.VANGUARD, (353, -152)),),
            ),
            (
                101945,
                (350, -152),
                ((348, -153), (349, -154), (350, -149), (351, -151)),
                (
                    (100, UnitType.VANGUARD, (352, -152)),
                    (101, UnitType.RANGER, (350, -150)),
                ),
            ),
            (
                101946,
                (349, -152),
                ((348, -153), (349, -154), (351, -151)),
                (
                    (100, UnitType.VANGUARD, (351, -152)),
                    (101, UnitType.RANGER, (350, -150)),
                ),
            ),
        )
        for tick, position, obstacles, hostile_rows in frames:
            worker = unit(worker_id, UnitType.WORKER, position, cargo=1)
            enemies = tuple(
                unit(identifier, unit_type, hostile_position, controlled=False)
                for identifier, unit_type, hostile_position in hostile_rows
            )
            turn = make_turn(
                tick=tick,
                core=core,
                units=(worker,),
                enemies=enemies,
                obstacle_cells=obstacles,
                resources=0,
            )
            tactic.choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertFalse(
            isinstance(action, MoveAction) and action.direction is Direction.UP
        )
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertNotEqual(task["metadata"].get("survival_terminals"), 0)

    def test_escape_history_never_forces_worker_toward_visible_vanguard(self) -> None:
        core = friendly_core(position=(10, 10))
        worker = unit(1, UnitType.WORKER, (0, 0))
        enemy = unit(100, UnitType.VANGUARD, (2, 1), controlled=False)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            # Both cells that increase the enemy distance are recent.  The
            # live Tick 97332/97333 bug deleted them before scoring and forced
            # the novel RIGHT step toward the Vanguard.
            position_history={worker.id: ((0, -1), (-1, 0))},
        )
        turn = make_turn(
            tick=10,
            core=core,
            units=(worker,),
            enemies=(enemy,),
            resources=0,
        )

        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        before = manhattan(worker.position, enemy.position)
        after = manhattan(add_direction(worker.position, action.direction), enemy.position)
        self.assertGreater(after, before)
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertEqual(task["mission"], "ESCAPE")
        self.assertGreater(task["metadata"]["survival_terminals"], 0)
        self.assertFalse(task["metadata"]["nonfatal_budget_used"])

    def test_safe_escape_step_precedes_full_health_nonfatal_budget(self) -> None:
        core = friendly_core(position=(10, 10))
        worker = unit(1, UnitType.WORKER, (0, 0))
        enemy = unit(100, UnitType.VANGUARD, (1, 1), controlled=False)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            position_history={worker.id: ((0, -1), (-1, 0))},
        )
        turn = make_turn(
            tick=10,
            core=core,
            units=(worker,),
            enemies=(enemy,),
            resources=0,
        )

        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        destination = add_direction(worker.position, action.direction)
        tactical_map = tactic.last_tactical_map
        assert tactical_map is not None
        self.assertEqual(tactical_map.immediate_attackers(destination), 0)
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertFalse(task["metadata"]["nonfatal_budget_used"])

    def test_visible_shared_threat_at_six_cells_is_local_fleeing(self) -> None:
        core = friendly_core(position=(10, 10))
        worker = unit(1, UnitType.WORKER, (0, 0))
        observer = unit(2, UnitType.RANGER, (0, 1))
        enemy = unit(100, UnitType.VANGUARD, (0, 6), controlled=False)
        turn = make_turn(
            tick=10,
            core=core,
            units=(worker, observer),
            enemies=(enemy,),
            resources=0,
        )

        tactic = BalancedTactic()
        tactic.choose_actions(turn)

        state = tactic.memory.worker_escape_states[worker.id]
        self.assertEqual(state.phase, "FLEEING")

    def test_escape_maximizes_minimum_distance_from_multiple_vanguards(self) -> None:
        core = friendly_core(position=(10, 10))
        worker = unit(1, UnitType.WORKER, (0, 0))
        enemies = (
            unit(100, UnitType.VANGUARD, (2, 0), controlled=False),
            unit(101, UnitType.VANGUARD, (0, 2), controlled=False),
        )
        turn = make_turn(
            tick=10,
            core=core,
            units=(worker,),
            enemies=enemies,
            resources=0,
        )

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        destination = add_direction(worker.position, action.direction)
        self.assertGreater(
            min(manhattan(destination, enemy.position) for enemy in enemies),
            min(manhattan(worker.position, enemy.position) for enemy in enemies),
        )

    def test_full_health_worker_uses_nonfatal_budget_only_when_required(self) -> None:
        core = friendly_core(position=(-5, 0))
        worker = unit(1, UnitType.WORKER, (0, 0))
        enemy = unit(100, UnitType.VANGUARD, (1, 1), controlled=False)
        turn = make_turn(
            tick=10,
            core=core,
            units=(worker,),
            enemies=(enemy,),
            obstacle_cells=((0, -1), (-1, 0)),
            resources=0,
        )

        tactic = BalancedTactic()
        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertTrue(task["metadata"]["nonfatal_budget_used"])

    def test_one_hp_worker_never_spends_nonfatal_escape_budget(self) -> None:
        core = friendly_core(position=(-5, 0))
        worker = unit(1, UnitType.WORKER, (0, 0), hp=1)
        enemy = unit(100, UnitType.VANGUARD, (1, 1), controlled=False)
        turn = make_turn(
            tick=10,
            core=core,
            units=(worker,),
            enemies=(enemy,),
            obstacle_cells=((0, -1), (-1, 0)),
            resources=2,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[worker.id], WaitAction)

    def test_escape_loop_creates_a_new_bounded_egress_waypoint(self) -> None:
        core = friendly_core(position=(10, 10))
        worker = unit(1, UnitType.WORKER, (0, 1))
        enemy = unit(100, UnitType.VANGUARD, (2, 1), controlled=False)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            position_history={
                worker.id: (
                    (0, 0),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                    (0, 0),
                    (1, 0),
                    (1, 1),
                )
            },
        )
        turn = make_turn(
            tick=10,
            core=core,
            units=(worker,),
            enemies=(enemy,),
            resources=0,
        )

        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(turn)

        state = tactic.memory.worker_escape_states[worker.id]
        self.assertEqual(state.loop_period, 4)
        self.assertIsNotNone(state.waypoint)
        self.assertEqual(state.route_version, 1)


    def test_emergency_patient_can_enter_core_through_safe_service_exit(self) -> None:
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(0, -1),
        )
        patient = unit(2, UnitType.RANGER, (-1, -2), hp=1)
        turn = make_turn(
            units=(patient,),
            resources=2,
            obstacle_cells=((-1, -1), (-2, -2), (-1, -3), (0, -3), (1, -2)),
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[patient.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.RIGHT)
        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["patient_gateway"], [0, -1])
        self.assertFalse(queue["core_slot_reserved"])
        self.assertEqual(queue["timeline"]["next_service_eta"], 3)

    def test_patient_eta_three_does_not_block_current_core_work(self) -> None:
        memory = TacticMemory(
            opening_complete=True,
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(0, -1),
        )
        patient = unit(2, UnitType.RANGER, (-1, -2), hp=1)
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(
            units=(patient,),
            resources=25,
            obstacle_cells=((-1, -1), (-2, -2), (-1, -3), (0, -3), (1, -2)),
        )

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["timeline"]["next_service_eta"], 3)
        self.assertTrue(queue["timeline"]["production_allowed"])
        self.assertIsInstance(turn.plan.core_action, SpawnAction)

    def test_stalled_remote_patient_keeps_funds_without_owning_current_slot(self) -> None:
        memory = TacticMemory(
            opening_complete=True,
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(0, -1),
        )
        patient = unit(2, UnitType.RANGER, (-1, -2), hp=1)
        tactic = BalancedTactic(memory=memory)
        for tick in (1, 2, 3):
            turn = make_turn(
                tick=tick,
                units=(patient,),
                resources=25,
                obstacle_cells=((-1, -1), (-2, -2), (-1, -3), (0, -3), (1, -2)),
            )
            tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertGreaterEqual(queue["patient_progress"]["stalled_ticks"], 2)
        self.assertEqual(queue["reserved_resources"], 1)
        self.assertFalse(queue["core_slot_reserved"])
        self.assertTrue(queue["timeline"]["production_allowed"])

    def test_resource_route_uses_safe_detour_before_waiting_on_service_zone(self) -> None:
        memory = TacticMemory(
            opening_complete=True,
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(-1, 0),
        )
        memory.known_passable.update(
            (x, y) for x in range(-1, 5) for y in range(-2, 3)
        )
        worker = unit(1, UnitType.WORKER, (1, 1))
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(
            units=(worker,),
            resource_cells=((3, 0),),
            resources=0,
        )

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[worker.id], MoveAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertNotEqual(task["reason"], "RESOURCE_ROUTE_BLOCKED_THIS_TICK")

    def test_stalled_resource_job_releases_target_and_backs_off_same_tick(self) -> None:
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update((x, 0) for x in range(0, 8))
        worker = unit(1, UnitType.WORKER, (4, 0))
        tactic = BalancedTactic(memory=memory)

        for tick in (1, 2, 3):
            turn = make_turn(
                tick=tick,
                units=(worker,),
                resource_cells=((6, 0),),
                resources=0,
            )
            tactic.choose_actions(turn)

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertEqual(task["reason"], "RESOURCE_STALL_SCOUT_FALLBACK")
        progress = tactic.last_decision_trace["economy"]["worker_task_progress"][0]
        self.assertEqual(progress["rejection_reason"], "RESOURCE_TASK_STALLED")
        self.assertGreaterEqual(progress["backoff_until"], 11)

    def test_far_patient_does_not_preempt_loaded_worker_already_on_core(self) -> None:
        memory = TacticMemory(
            opening_complete=True,
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(0, -1),
        )
        carrier = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        patient = unit(2, UnitType.RANGER, (-1, -2), hp=1)
        turn = make_turn(
            units=(carrier, patient),
            resources=1,
            obstacle_cells=((-1, -1), (-2, -2), (-1, -3), (0, -3), (1, -2)),
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[carrier.id], DepositAction)
        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["admission_id"], str(carrier.id))
        self.assertEqual(queue["timeline"]["next_service_eta"], 0)

    def test_remote_future_carriers_keep_returning_before_transit_holds(self) -> None:
        carriers = tuple(
            unit(index, UnitType.WORKER, (8 + index, 0), cargo=1)
            for index in range(1, 5)
        )
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y) for x in range(-12, 13) for y in range(-12, 13)
        )
        tactic = BalancedTactic(memory=memory)

        turn = make_turn(units=carriers, resources=0)
        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        overflow = queue["overflow_slots"]
        self.assertEqual(overflow, [])
        reservations = queue["return_reservations"]
        self.assertEqual(len(reservations), len(carriers))
        waiting = {
            row["worker_id"]
            for row in reservations
            if row["status"] == "WAIT_FOR_DEPARTURE"
        }
        self.assertEqual(waiting, set())
        self.assertEqual(queue["holding_depositors"], [])
        for carrier in carriers:
            action = turn.plan.unit_actions[carrier.id]
            self.assertIsInstance(action, MoveAction)
            destination = (
                carrier.position[0] + action.direction.delta[0],
                carrier.position[1] + action.direction.delta[1],
            )
            self.assertLess(
                manhattan(destination, (0, 0)),
                manhattan(carrier.position, (0, 0)),
            )

    def test_local_emergency_patient_reserves_core_slot_and_blocks_spawn(self) -> None:
        memory = TacticMemory(opening_complete=True)
        patient = unit(1, UnitType.VANGUARD, (1, 0), hp=2)
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(units=(patient,), resources=25)

        tactic.choose_actions(turn)

        self.assertNotIsInstance(turn.plan.core_action, SpawnAction)
        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertTrue(queue["core_slot_reserved"])
        self.assertEqual(
            tactic.last_decision_trace["economy"]["production_candidates"][0]["reason"],
            "SERVICE_DUE_THIS_TICK",
        )

    def test_all_half_health_combat_patients_reserve_missing_hp(self) -> None:
        patients = (
            unit(1, UnitType.VANGUARD, (2, 0), hp=2),
            unit(2, UnitType.RANGER, (-2, 0), hp=1),
        )
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))

        tactic.choose_actions(make_turn(units=patients, resources=10))

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["reserved_resources"], 3)

    def test_admitted_outer_carrier_advances_one_service_slot_per_tick(self) -> None:
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(-1, 0),
            service_admission_id=uid(1),
            cargo_arrival_ticks={uid(1): 1},
        )
        carrier = unit(1, UnitType.WORKER, (2, 0), cargo=1)
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(tick=2, units=(carrier,), resources=0)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.LEFT)

    def test_full_storage_workers_return_to_bounded_staging_band(self) -> None:
        workers = (
            unit(1, UnitType.WORKER, (0, 0), cargo=1),
            unit(2, UnitType.WORKER, (1, 0), cargo=1),
            unit(3, UnitType.WORKER, (-1, 0), cargo=1),
            unit(4, UnitType.WORKER, (0, 1), cargo=1),
            unit(5, UnitType.WORKER, (5, 0)),
        )
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-12, 13)
            for y in range(-12, 13)
            if abs(x) + abs(y) <= 12
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(
            units=workers,
            resource_cells=((5, 0),),
            resources=25,
        )

        tactic.choose_actions(turn)

        worker_ids = {str(worker.id) for worker in workers}
        tasks = [
            task
            for task in tactic.last_decision_trace["tasks"]
            if task["actor_id"] in worker_ids
        ]
        self.assertTrue(tasks)
        cargo_ids = {str(worker.id) for worker in workers if worker.cargo > 0}
        cargo_tasks = [task for task in tasks if task["actor_id"] in cargo_ids]
        empty_tasks = [task for task in tasks if task["actor_id"] not in cargo_ids]
        self.assertTrue(
            all(task["mission"] == "FULL_STORAGE_STAGING" for task in cargo_tasks)
        )
        self.assertTrue(empty_tasks)
        self.assertTrue(all(task["mission"] == "HARVEST" for task in empty_tasks))
        self.assertEqual(
            tactic.memory.worker_economy_modes[workers[-1].id],
            WorkerEconomyMode.RESOURCE_ACQUISITION,
        )
        self.assertFalse(
            {
                "RETURN_CARGO",
                "DEPOSIT",
                "HARVEST",
            }
            & {task["mission"] for task in cargo_tasks}
        )
        service = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(service["admission_id"], None)
        self.assertEqual(service["depositors"], [])
        posts = tactic.memory.worker_home_guard_targets
        self.assertEqual(
            set(posts),
            {worker.id for worker in workers if worker.cargo > 0},
        )
        self.assertEqual(len(set(posts.values())), len(posts))
        distances = sorted(manhattan((0, 0), post) for post in posts.values())
        self.assertTrue(all(8 <= distance <= 12 for distance in distances))

    def test_full_storage_staging_preserves_an_already_legal_carrier_position(
        self,
    ) -> None:
        worker = unit(1, UnitType.WORKER, (8, 0), cargo=1)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-12, 13)
            for y in range(-12, 13)
            if abs(x) + abs(y) <= 12
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(units=(worker,), resources=10)

        tactic.choose_actions(turn)

        self.assertEqual(tactic.memory.worker_home_guard_targets[worker.id], (8, 0))
        self.assertIsInstance(turn.plan.unit_actions[worker.id], WaitAction)
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertEqual(task["mission"], "FULL_STORAGE_STAGING")

    def test_remote_contact_does_not_displace_full_storage_home_guard(self) -> None:
        workers = tuple(
            unit(index, UnitType.WORKER, (0, 0), cargo=1)
            for index in range(1, 7)
        )
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-20, 21)
            for y in range(-20, 21)
            if abs(x) + abs(y) <= 20
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(
            units=workers,
            enemies=(unit(100, UnitType.RANGER, (0, 20), controlled=False),),
            resources=30,
        )

        tactic.choose_actions(turn)

        distances = {
            manhattan((0, 0), post)
            for post in tactic.memory.worker_home_guard_targets.values()
        }
        self.assertTrue(distances)
        self.assertTrue(
            all(
                distance in tactic.config.worker_full_storage_guard_radii
                or tactic.config.worker_full_storage_parking_min_radius
                <= distance
                <= tactic.config.worker_full_storage_parking_max_radius
                for distance in distances
            )
        )
        self.assertLessEqual(
            sum(
                post.zone == "NEAR_RESERVE"
                for post in tactic.memory.worker_parking_assignments.values()
            ),
            tactic.config.worker_full_storage_near_reserve_count,
        )

    def test_all_full_storage_carriers_receive_unique_posts_within_twelve(self) -> None:
        workers = tuple(
            unit(index, UnitType.WORKER, (index - 8, 0), cargo=1)
            for index in range(1, 17)
        )
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-14, 15)
            for y in range(-14, 15)
            if abs(x) + abs(y) <= 14
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(make_turn(units=workers, resources=80))

        posts = tactic.memory.worker_parking_assignments
        self.assertEqual(len(posts), len(workers))
        self.assertEqual(len({post.position for post in posts.values()}), len(workers))
        self.assertTrue(all(post.zone == "CARGO_STAGING" for post in posts.values()))
        self.assertTrue(
            all(
                8 <= manhattan((0, 0), post.position) <= 12
                for post in posts.values()
            )
        )

    def test_damage_heat_persists_and_decays_after_exact_tracks_expire(self) -> None:
        worker = unit(1, UnitType.WORKER, (6, 0))
        enemy = unit(100, UnitType.RANGER, (3, 0), controlled=False)
        damaged = ResolutionEvent(
            event_id=uid(90_020),
            tick=1,
            event_type="UNIT_DAMAGED",
            actor_id=worker.id,
            position=(6, 0),
        )
        memory = TacticMemory()
        first = build_world_model(
            make_turn(
                tick=1,
                units=(worker,),
                enemies=(enemy,),
                events=(damaged,),
            ),
            memory,
        )
        first_heat = dict(first.threat_heat)
        self.assertEqual(first_heat[(6, 0)], 16)

        later = None
        for tick in range(2, 9):
            later = build_world_model(
                make_turn(tick=tick, units=(worker,), enemies=()),
                memory,
            )
        self.assertIsNotNone(later)
        self.assertEqual(memory.enemy_tracks, {})
        self.assertGreater(dict(later.threat_heat).get((6, 0), 0), 0)
        self.assertEqual(dict(later.danger_cells).get((6, 0), 0), 0)
        midpoint = build_world_model(
            make_turn(tick=21, units=(worker,), enemies=()),
            memory,
        )
        self.assertEqual(dict(midpoint.threat_heat)[(6, 0)], 8)
        expired = build_world_model(
            make_turn(tick=41, units=(worker,), enemies=()),
            memory,
        )
        self.assertNotIn((6, 0), dict(expired.threat_heat))

    def test_threat_heat_capacity_keeps_only_highest_risk_cells(self) -> None:
        config = TacticConfig(threat_heat_cell_limit=3)
        enemy = unit(100, UnitType.RANGER, (3, 0), controlled=False)
        memory = TacticMemory()

        world = build_world_model(
            make_turn(enemies=(enemy,)),
            memory,
            config,
        )

        self.assertLessEqual(len(world.threat_heat), 3)
        self.assertTrue(all(risk == 8 for _, risk in world.threat_heat))

    def test_unit_move_failure_uses_the_previous_destination_not_event_origin(self) -> None:
        worker = unit(1, UnitType.WORKER, (1, 0))
        tactic = BalancedTactic()
        first = make_turn(tick=1, units=(worker,), resources=0)
        tactic.choose_actions(first)
        action = first.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        destination = (
            worker.position[0] + action.direction.delta[0],
            worker.position[1] + action.direction.delta[1],
        )
        event = ResolutionEvent(
            event_id=uid(90_010),
            tick=1,
            event_type="UNIT_MOVE_FAILED",
            actor_id=worker.id,
            position=worker.position,
            reason_code="MOVE_CONTESTED",
        )

        second_turn = make_turn(
            tick=2,
            units=(worker,),
            resources=0,
            events=(event,),
        )
        tactic.choose_actions(second_turn)

        failure = tactic.memory.failed_unit_moves[worker.id]
        self.assertEqual(failure.destination, destination)
        second = second_turn.plan.unit_actions.get(worker.id)
        self.assertFalse(
            isinstance(second, MoveAction) and second.direction is action.direction
        )

    def test_weighted_route_prefers_a_longer_safe_path(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0))
        world = build_world_model(
            make_turn(
                core=friendly_core(position=(10, 10)),
                units=(worker,),
                resources=0,
            )
        )

        route = weighted_route_to(
            world,
            worker.position,
            (2, 0),
            node_limit=128,
            cell_costs={(1, 0): 100},
        )

        self.assertIsNotNone(route)
        self.assertNotEqual(route.first_direction, Direction.RIGHT)

    def test_opening_builds_four_workers_then_vanguard_then_ranger(self) -> None:
        tactic = BalancedTactic()
        two_workers = (
            unit(1, UnitType.WORKER, (1, 0)),
            unit(2, UnitType.WORKER, (-1, 0)),
        )
        first = make_turn(units=two_workers, resources=5)
        tactic.choose_actions(first)
        self.assertEqual(first.plan.core_action.unit_type, UnitType.WORKER)

        three_workers = (*two_workers, unit(3, UnitType.WORKER, (0, 1)))
        second = make_turn(tick=2, units=three_workers, resources=5)
        tactic.choose_actions(second)
        self.assertEqual(second.plan.core_action.unit_type, UnitType.WORKER)

        four_workers = (*three_workers, unit(4, UnitType.WORKER, (0, -1)))
        third = make_turn(tick=3, units=four_workers, resources=10)
        tactic.choose_actions(third)
        self.assertEqual(third.plan.core_action.unit_type, UnitType.VANGUARD)

        with_vanguard = (*four_workers, unit(5, UnitType.VANGUARD, (2, 0)))
        fourth = make_turn(tick=4, units=with_vanguard, resources=12)
        tactic.choose_actions(fourth)
        self.assertEqual(fourth.plan.core_action.unit_type, UnitType.RANGER)

    def test_visible_combat_threat_interrupts_worker_bootstrap(self) -> None:
        workers = (
            unit(1, UnitType.WORKER, (1, 0)),
            unit(2, UnitType.WORKER, (-1, 0)),
        )
        enemy = unit(100, UnitType.VANGUARD, (3, 0), controlled=False)
        turn = make_turn(units=workers, enemies=(enemy,), resources=10)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.VANGUARD)

    def test_worker_target_is_population_half_rounded_up(self) -> None:
        units = tuple(
            unit(index, UnitType.WORKER if index <= 5 else UnitType.VANGUARD, (index, 0))
            for index in range(1, 12)
        )
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(units=units, resources=0))

        self.assertEqual(tactic.last_decision_trace["economy"]["worker_target"], math.ceil(11 / 2))

    def test_post_25_production_builds_toward_mature_combat_target(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (20 + i, 0)) for i in range(1, 14)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 7)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 7)]
        )
        turn = make_turn(units=units, resources=125)

        BalancedTactic(memory=TacticMemory(opening_complete=True)).choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.VANGUARD)

    def test_population_35_waits_for_exact_full_storage_before_normal_spawn(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (20 + i, 0)) for i in range(1, 24)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 7)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 7)]
        )
        turn = make_turn(units=units, resources=100)
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, WaitAction)
        candidates = tactic.last_decision_trace["economy"]["production_candidates"]
        self.assertFalse(any(item["selected"] for item in candidates))
        self.assertEqual(
            {item["production_mode"] for item in candidates},
            {"WAIT_FOR_FULL_STORAGE"},
        )

    def test_population_35_exact_full_storage_spawns_one_needed_unit(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (20 + i, 0)) for i in range(1, 24)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 7)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 7)]
        )
        turn = make_turn(units=units, resources=175)
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.VANGUARD)
        candidates = tactic.last_decision_trace["economy"]["production_candidates"]
        selected = next(item for item in candidates if item["selected"])
        self.assertEqual(selected["production_mode"], "MATURE_COMBAT_TARGET")

    def test_population_25_and_69_wait_for_full_storage(self) -> None:
        cases = (
            tuple(
                [unit(i, UnitType.WORKER, (20 + i, 0)) for i in range(1, 14)]
                + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 7)]
                + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 7)]
            ),
            tuple(
                [unit(i, UnitType.WORKER, (40 + i, 0)) for i in range(1, 36)]
                + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 18)]
                + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 18)]
            ),
        )
        for units in cases:
            with self.subTest(population=len(units)):
                turn = make_turn(
                    units=units,
                    resources=len(units) * 5 - 5,
                )
                tactic = BalancedTactic(
                    memory=TacticMemory(opening_complete=True)
                )

                tactic.choose_actions(turn)

                self.assertIsInstance(turn.plan.core_action, WaitAction)
                self.assertEqual(
                    tactic.last_decision_trace["economy"]["production_mode"],
                    "WAIT_FOR_FULL_STORAGE",
                )

    def test_population_24_can_spawn_without_full_storage(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (20 + i, 0)) for i in range(1, 13)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 7)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 7)]
        )
        turn = make_turn(units=units, resources=20)

        BalancedTactic(memory=TacticMemory(opening_complete=True)).choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)

    def test_mature_force_fills_missing_combat_before_stockpiling(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (40 + i, 0)) for i in range(1, 36)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 18)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 18)]
        )
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))
        turn = make_turn(units=units, resources=345)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertIn(
            turn.plan.core_action.unit_type,
            {UnitType.VANGUARD, UnitType.RANGER},
        )
        candidates = tactic.last_decision_trace["economy"]["production_candidates"]
        selected = next(item for item in candidates if item["selected"])
        self.assertEqual(selected["production_mode"], "MATURE_COMBAT_TARGET")

    def test_mature_force_fills_missing_worker_before_stockpiling(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (40 + i, 0)) for i in range(1, 35)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 19)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 18)]
        )
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))
        turn = make_turn(units=units, resources=345)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.WORKER)
        candidates = tactic.last_decision_trace["economy"]["production_candidates"]
        selected = next(item for item in candidates if item["selected"])
        self.assertEqual(selected["production_mode"], "MATURE_WORKER_TARGET")

    def test_population_70_balanced_force_keeps_full_stockpile(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (40 + i, 0)) for i in range(1, 36)]
            + [
                unit(100 + i, UnitType.VANGUARD, (i % 9, 5 + i // 9))
                for i in range(1, 19)
            ]
            + [
                unit(200 + i, UnitType.RANGER, (i % 9, -5 - i // 9))
                for i in range(1, 18)
            ]
        )
        tactic = BalancedTactic(
            memory=TacticMemory(
                opening_complete=True,
                home_force_high_water=22,
            )
        )
        turn = make_turn(units=units, resources=350)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, WaitAction)
        candidates = tactic.last_decision_trace["economy"]["production_candidates"]
        self.assertEqual({item["production_mode"] for item in candidates}, {"HIGH_POP_STOCKPILE"})
        self.assertFalse(any(item["selected"] for item in candidates))
        economy = tactic.last_decision_trace["economy"]
        self.assertTrue(economy["stockpile_active"])
        self.assertEqual(economy["stockpile_worker_target"], 35)
        self.assertEqual(economy["stockpile_combat_target"], 35)
        self.assertEqual(economy["stockpile_worker_gap"], 0)
        self.assertEqual(economy["stockpile_combat_gap"], 0)
        self.assertTrue(economy["mature_stockpile_ready"])
        self.assertTrue(economy["saturated_patrol_active"])
        self.assertEqual(
            tactic.memory.worker_economy_modes[units[0].id],
            WorkerEconomyMode.SATURATED_PATROL,
        )

    def test_mature_patrol_uses_storage_hysteresis_but_releases_at_five_space(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (40 + i, 0)) for i in range(1, 36)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 19)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 18)]
        )
        tactic = BalancedTactic(
            memory=TacticMemory(opening_complete=True, home_force_high_water=22)
        )

        tactic.choose_actions(make_turn(tick=1, units=units, resources=350))
        self.assertTrue(
            tactic.last_decision_trace["economy"]["saturated_patrol_active"]
        )

        tactic.choose_actions(make_turn(tick=2, units=units, resources=349))
        self.assertTrue(
            tactic.last_decision_trace["economy"]["saturated_patrol_active"]
        )
        self.assertEqual(
            tactic.memory.worker_economy_modes[units[0].id],
            WorkerEconomyMode.SATURATED_PATROL,
        )

        tactic.choose_actions(make_turn(tick=3, units=units, resources=345))
        self.assertFalse(
            tactic.last_decision_trace["economy"]["saturated_patrol_active"]
        )
        self.assertEqual(
            tactic.memory.worker_economy_modes[units[0].id],
            WorkerEconomyMode.RESOURCE_SEARCH,
        )

    def test_dynamic_home_force_gap_blocks_mature_patrol_and_spawns_at_full(self) -> None:
        units = tuple(
            [unit(i, UnitType.WORKER, (40 + i, 0)) for i in range(1, 36)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 19)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 18)]
        )
        tactic = BalancedTactic(
            memory=TacticMemory(
                core_id=uid(10_000),
                core_position=(0, 0),
                opening_complete=True,
                home_force_high_water=40,
            )
        )
        turn = make_turn(units=units, resources=350)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertFalse(
            tactic.last_decision_trace["economy"]["mature_stockpile_ready"]
        )
        self.assertFalse(
            tactic.last_decision_trace["economy"]["saturated_patrol_active"]
        )
        self.assertNotEqual(
            tactic.memory.worker_economy_modes[units[0].id],
            WorkerEconomyMode.SATURATED_PATROL,
        )

    def test_high_population_post_crisis_rebuild_uses_available_stockpile(self) -> None:
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            opening_complete=True,
            crisis_force_baseline=CrisisForceBaseline(
                vanguards=6,
                rangers=6,
                started_tick=10,
                phase="REBUILD",
                safe_ticks=4,
            ),
        )
        units = tuple(
            [unit(i, UnitType.WORKER, (30 + i, 0)) for i in range(1, 25)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 6)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 7)]
        )
        turn = make_turn(tick=20, units=units, resources=100)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.VANGUARD)
        candidates = tactic.last_decision_trace["economy"]["production_candidates"]
        selected = next(item for item in candidates if item["selected"])
        self.assertEqual(selected["production_mode"], "POST_CRISIS_REBUILD")

        full = make_turn(tick=21, units=units, resources=175)
        tactic.choose_actions(full)
        self.assertIsInstance(full.plan.core_action, SpawnAction)
        self.assertEqual(full.plan.core_action.unit_type, UnitType.VANGUARD)
        selected = next(
            item
            for item in tactic.last_decision_trace["economy"]["production_candidates"]
            if item["selected"]
        )
        self.assertEqual(selected["production_mode"], "POST_CRISIS_REBUILD")
        self.assertEqual(selected["vanguard_rebuild_gap"], 1)
        self.assertEqual(selected["ranger_rebuild_gap"], 0)

    def test_high_population_active_crisis_accumulates_until_affordable(self) -> None:
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            opening_complete=True,
            home_force_high_water=17,
            crisis_force_baseline=CrisisForceBaseline(
                vanguards=8,
                rangers=8,
                started_tick=10,
                phase="ACTIVE",
            ),
        )
        units = tuple(
            [unit(i, UnitType.WORKER, (30 + i, 0)) for i in range(1, 24)]
            + [unit(100 + i, UnitType.VANGUARD, (i, 5)) for i in range(1, 9)]
            + [unit(200 + i, UnitType.RANGER, (i, -5)) for i in range(1, 9)]
        )
        enemies = tuple(
            unit(300 + i, UnitType.VANGUARD, (3 + i, 0), controlled=False)
            for i in range(1, 5)
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(tick=20, units=units, enemies=enemies, resources=10)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, WaitAction)
        candidates = tactic.last_decision_trace["economy"]["production_candidates"]
        self.assertFalse(any(item.get("selected") for item in candidates))
        self.assertTrue(
            all(
                item["production_mode"] == "CRISIS_REINFORCEMENT"
                for item in candidates
            )
        )

        full = make_turn(
            tick=21,
            units=units,
            enemies=enemies,
            resources=195,
        )
        tactic.choose_actions(full)
        self.assertIsInstance(full.plan.core_action, SpawnAction)
        self.assertEqual(full.plan.core_action.unit_type, UnitType.VANGUARD)
        selected = next(
            item
            for item in tactic.last_decision_trace["economy"]["production_candidates"]
            if item["selected"]
        )
        self.assertEqual(selected["production_mode"], "CRISIS_REINFORCEMENT")

    def test_cargo_on_stationary_core_always_deposits(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        turn = make_turn(units=(carrier,), resources=0)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[carrier.id], DepositAction)

    def test_delivery_admission_remains_stable(self) -> None:
        tactic = BalancedTactic()
        carriers = (
            unit(1, UnitType.WORKER, (2, 0), cargo=1),
            unit(2, UnitType.WORKER, (1, 1), cargo=1),
        )
        first = make_turn(units=carriers, resources=0)
        tactic.choose_actions(first)
        admitted = tactic.last_decision_trace["economy"]["service_queue"]["admission_id"]

        second = make_turn(tick=2, units=carriers, resources=0)
        tactic.choose_actions(second)

        self.assertEqual(
            tactic.last_decision_trace["economy"]["service_queue"]["admission_id"],
            admitted,
        )

    def test_remote_old_cargo_does_not_block_a_ready_worker(self) -> None:
        far = unit(1, UnitType.WORKER, (5, 0), cargo=1)
        ready = unit(2, UnitType.WORKER, (0, 2), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=far.id,
            service_kind="DEPOSIT",
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
            cargo_arrival_ticks={far.id: 1, ready.id: 2},
        )
        turn = make_turn(
            tick=3,
            core=friendly_core(position=(0, 0)),
            units=(far, ready),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertIsNone(queue["admission_id"])
        self.assertIsInstance(turn.plan.unit_actions[ready.id], MoveAction)
        self.assertEqual(turn.plan.unit_actions[ready.id].direction, Direction.UP)

    def test_front_of_lane_clears_before_an_older_outer_worker(self) -> None:
        older_outer = unit(1, UnitType.WORKER, (0, 2), cargo=1)
        front = unit(2, UnitType.WORKER, (0, 1), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=older_outer.id,
            service_kind="DEPOSIT",
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
            cargo_arrival_ticks={older_outer.id: 1, front.id: 2},
        )
        turn = make_turn(
            tick=3,
            core=friendly_core(position=(0, 0)),
            units=(older_outer, front),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["admission_id"], str(front.id))
        self.assertEqual(turn.plan.unit_actions[front.id].direction, Direction.UP)

    def test_ready_queue_advances_on_reserved_departure_ticks(self) -> None:
        front = unit(1, UnitType.WORKER, (0, 1), cargo=1)
        outer = unit(2, UnitType.WORKER, (0, 2), cargo=1)
        approaching = unit(3, UnitType.WORKER, (0, 3), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=front.id,
            service_kind="DEPOSIT",
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
            cargo_arrival_ticks={front.id: 1, outer.id: 2},
        )
        turn = make_turn(
            tick=3,
            core=friendly_core(position=(0, 0)),
            units=(front, outer, approaching),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[front.id], MoveAction)
        self.assertEqual(turn.plan.unit_actions[front.id].direction, Direction.UP)
        for worker in (outer, approaching):
            self.assertIsInstance(turn.plan.unit_actions[worker.id], MoveAction)
            task = next(
                row
                for row in tactic.last_decision_trace["tasks"]
                if row["actor_id"] == str(worker.id)
            )
            self.assertIn(
                task["reason"],
                {
                    "SERVICE_PIPELINE_ADVANCE",
                    "SERVICE_QUEUE_APPROACH",
                    "SERVICE_TRANSIT_HOLD_APPROACH",
                },
            )
        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["ready_depositors"], [str(front.id), str(outer.id)])
        self.assertEqual(queue["holding_depositors"], [])

    def test_blocked_pipeline_wait_keeps_its_service_reason(self) -> None:
        depositing = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        front = unit(2, UnitType.WORKER, (0, 1), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=front.id,
            service_kind="DEPOSIT",
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
            cargo_arrival_ticks={front.id: 1},
        )
        turn = make_turn(units=(depositing, front), resources=0)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[front.id], WaitAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(front.id)
        )
        self.assertEqual(task["mission"], "RETURN_CARGO")
        self.assertEqual(task["reason"], "WAITING_FOR_SERVICE_SLOT")

    def test_remote_carrier_approaches_the_outer_slot_without_cutting_the_line(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, -2), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
        )
        turn = make_turn(
            tick=2,
            core=friendly_core(position=(0, 0)),
            units=(carrier,),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(action, MoveAction)
        self.assertIn(action.direction, {Direction.LEFT, Direction.RIGHT})

    def test_side_adjacent_carrier_funds_urgent_recovery_without_joining_lane(self) -> None:
        first = unit(1, UnitType.WORKER, (0, -1), cargo=1)
        second = unit(2, UnitType.WORKER, (0, -1), cargo=1)
        patient = unit(3, UnitType.RANGER, (3, -1), hp=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(-1, 0),
        )
        turn = make_turn(
            tick=20,
            core=friendly_core(position=(0, 0)),
            units=(first, second, patient),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["service"], "HEAL_FUNDING")
        self.assertEqual(queue["admission_id"], str(first.id))
        action = turn.plan.unit_actions[first.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.DOWN)

    def test_dead_end_persisted_lane_is_replanned_before_cargo_return(self) -> None:
        carrier = unit(1, UnitType.WORKER, (4, 0), cargo=1)
        stale_lane = ((0, 1), (0, 2))
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=stale_lane[0],
            service_queue_cells=stale_lane,
            service_exit_cell=(0, -1),
        )
        turn = make_turn(
            tick=2,
            core=friendly_core(position=(0, 0)),
            units=(carrier,),
            resources=0,
            # The outer slot is a cul-de-sac: it is locally valid terrain but
            # can only be entered through the protected inner queue cell.
            obstacle_cells=((-1, 2), (1, 2), (0, 3)),
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertNotEqual(queue["queue_cells"], [list(cell) for cell in stale_lane])
        self.assertIsInstance(turn.plan.unit_actions[carrier.id], MoveAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(carrier.id)
        )
        self.assertNotEqual(task["reason"], "NO_RETURN_ROUTE")

    def test_service_lane_change_restarts_ready_tick(self) -> None:
        carrier = unit(1, UnitType.WORKER, (1, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=carrier.id,
            service_kind="DEPOSIT",
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
            cargo_arrival_ticks={carrier.id: 1},
        )
        turn = make_turn(
            tick=20,
            core=friendly_core(position=(0, 0)),
            units=(carrier,),
            resources=0,
            obstacle_cells=((0, 1),),
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        ready_ticks = {
            item["worker_id"]: item["tick"] for item in queue["ready_ticks"]
        }
        self.assertEqual(ready_ticks[str(carrier.id)], 20)

    def test_manual_position_change_releases_ready_admission(self) -> None:
        first = unit(1, UnitType.WORKER, (0, 1), cargo=1)
        second = unit(2, UnitType.WORKER, (0, 2), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=first.id,
            service_kind="DEPOSIT",
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
            cargo_arrival_ticks={first.id: 1, second.id: 2},
        )
        tactic = BalancedTactic(memory=memory)

        turn = make_turn(
            tick=3,
            units=(unit(1, UnitType.WORKER, (5, 0), cargo=1), second),
            resources=0,
        )
        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertIsNone(queue["admission_id"])
        self.assertIsInstance(turn.plan.unit_actions[second.id], MoveAction)
        self.assertEqual(turn.plan.unit_actions[second.id].direction, Direction.UP)
        self.assertEqual(queue["release_reason"], "LEFT_READY_LINE")

    def test_moving_core_clears_ready_fifo(self) -> None:
        carrier = unit(1, UnitType.WORKER, (1, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=carrier.id,
            service_kind="DEPOSIT",
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(-1, 0),
            cargo_arrival_ticks={carrier.id: 1},
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(
                tick=4,
                core=friendly_core(position=(0, 0), moving=True),
                units=(carrier,),
                resources=0,
            )
        )

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertIsNone(queue["admission_id"])
        self.assertEqual(tactic.memory.cargo_arrival_ticks, {})

    def test_cargo_routes_around_a_visible_enemy_core(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, -2), cargo=1)
        blocking_core = enemy_core(99, (0, -1))
        turn = make_turn(
            tick=1,
            units=(carrier,),
            enemies=(blocking_core,),
            resources=0,
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(action, MoveAction)
        dx, dy = action.direction.delta
        self.assertGreater(
            manhattan(
                (carrier.position[0] + dx, carrier.position[1] + dy),
                blocking_core.position,
            ),
            manhattan(carrier.position, blocking_core.position),
        )

    def test_loaded_worker_may_approach_a_remote_enemy_before_spatial_detour(self) -> None:
        core = friendly_core(position=(-10, 0))
        carrier = unit(1, UnitType.WORKER, (8, 0), cargo=1)
        observer = unit(2, UnitType.RANGER, (0, 1))
        enemy = unit(100, UnitType.VANGUARD, (0, 0), controlled=False)
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        memory.known_passable.update(
            (x, y)
            for x in range(-12, 11)
            for y in range(-12, 13)
        )
        turn = make_turn(
            tick=2,
            core=core,
            units=(carrier, observer),
            enemies=(enemy,),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.LEFT)

        destination = (carrier.position[0] - 1, carrier.position[1])
        projection = tactic.last_tactical_map
        assert projection is not None
        self.assertEqual(projection.immediate_attackers(destination), 0)
        self.assertLess(
            manhattan(destination, core.position),
            manhattan(carrier.position, core.position),
        )

    def test_resource_distance_field_routes_around_visible_enemy_occupancy(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, -2))
        blocking_core = enemy_core(99, (0, -1))
        turn = make_turn(
            units=(worker,),
            enemies=(blocking_core,),
            resources=0,
            resource_cells=((0, 2),),
        )

        BalancedTactic().choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        dx, dy = action.direction.delta
        self.assertGreater(
            manhattan(
                (worker.position[0] + dx, worker.position[1] + dy),
                blocking_core.position,
            ),
            manhattan(worker.position, blocking_core.position),
        )

    def test_empty_worker_exits_as_the_next_carrier_enters_core(self) -> None:
        empty = unit(1, UnitType.WORKER, (0, 0))
        carrier = unit(2, UnitType.WORKER, (0, 1), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=carrier.id,
            service_kind="DEPOSIT",
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
            cargo_arrival_ticks={carrier.id: 1},
        )
        turn = make_turn(
            tick=2,
            core=friendly_core(position=(0, 0)),
            units=(empty, carrier),
            resources=0,
            obstacle_cells=((-1, 0), (1, 0)),
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        empty_action = turn.plan.unit_actions[empty.id]
        carrier_action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(empty_action, MoveAction)
        self.assertEqual(empty_action.direction, Direction.UP)
        self.assertIsInstance(carrier_action, MoveAction)
        self.assertEqual(carrier_action.direction, Direction.UP)

    def test_deposited_worker_clears_a_resource_exit_without_reharvesting(self) -> None:
        empty = unit(1, UnitType.WORKER, (0, -1))
        carrier = unit(2, UnitType.WORKER, (1, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(0, -1),
            service_egress_worker_ids={empty.id},
            cargo_arrival_ticks={carrier.id: 1},
        )
        turn = make_turn(
            tick=2,
            units=(empty, carrier),
            resources=0,
            resource_cells=((0, -1),),
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[empty.id], MoveAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(empty.id)
        )
        self.assertEqual(task["mission"], "CLEAR_CORE")
        self.assertEqual(task["reason"], "CLEAR_SERVICE_EXIT")

    def test_empty_core_worker_can_share_a_single_occupied_exit(self) -> None:
        empty = unit(1, UnitType.WORKER, (0, 0))
        exit_carrier = unit(2, UnitType.WORKER, (0, -1), cargo=1)
        admitted = unit(3, UnitType.WORKER, (1, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=admitted.id,
            service_kind="DEPOSIT",
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(0, -1),
            cargo_arrival_ticks={admitted.id: 1},
        )
        turn = make_turn(
            tick=2,
            units=(
                empty,
                exit_carrier,
                admitted,
            ),
            resources=0,
            obstacle_cells=((-1, 0), (0, 1)),
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[empty.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.UP)

    def test_remote_carriers_follow_departure_ticks_without_staging_herd(self) -> None:
        carriers = (
            unit(1, UnitType.WORKER, (6, 0), cargo=1),
            unit(2, UnitType.WORKER, (6, 1), cargo=1),
        )
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(-1, 0),
        )
        turn = make_turn(tick=2, units=carriers, resources=0)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(len(queue["approaching_depositors"]), 2)
        self.assertEqual(len(queue["holding_depositors"]), 0)
        for carrier in carriers:
            action = turn.plan.unit_actions[carrier.id]
            self.assertIsInstance(action, MoveAction)
            destination = (
                carrier.position[0] + action.direction.delta[0],
                carrier.position[1] + action.direction.delta[1],
            )
            self.assertLess(
                manhattan(destination, (0, 0)),
                manhattan(carrier.position, (0, 0)),
            )

    def test_safe_empty_service_lane_advances_two_carriers_in_parallel(self) -> None:
        carriers = (
            unit(1, UnitType.WORKER, (4, 0), cargo=1),
            unit(2, UnitType.WORKER, (0, 4), cargo=1),
            unit(3, UnitType.WORKER, (-4, 0), cargo=1),
            unit(4, UnitType.WORKER, (0, -4), cargo=1),
        )
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-8, 9)
            for y in range(-8, 9)
        )
        turn = make_turn(tick=100, units=carriers, resources=0)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(len(queue["approaching_depositors"]), 4)
        schedule = queue["scheduled_deposits"]
        self.assertEqual(
            {row["worker_id"] for row in schedule},
            {str(carrier.id) for carrier in carriers},
        )
        scheduled_ticks = [row["tick"] for row in schedule]
        self.assertEqual(scheduled_ticks, sorted(set(scheduled_ticks)))
        self.assertTrue(
            all(
                later - earlier >= 2
                for earlier, later in zip(scheduled_ticks, scheduled_ticks[1:])
            )
        )
        timeline_deposits = [
            request
            for request in queue["timeline"]["requests"]
            if request["operation"] == "DEPOSIT"
        ]
        self.assertEqual(len(timeline_deposits), len(carriers))
        progressing = [
            carrier
            for carrier in carriers
            if isinstance(turn.plan.unit_actions[carrier.id], MoveAction)
        ]
        self.assertGreaterEqual(len(progressing), 2)

        # A future appointment must remain fixed while an early Worker is
        # staging.  Sliding every Tick recreates permanent overflow WAIT.
        next_units = []
        for index, carrier in enumerate(carriers, start=1):
            action = turn.plan.unit_actions[carrier.id]
            position = carrier.position
            if isinstance(action, MoveAction):
                position = (
                    position[0] + action.direction.delta[0],
                    position[1] + action.direction.delta[1],
                )
            next_units.append(unit(index, UnitType.WORKER, position, cargo=1))
        next_turn = make_turn(tick=101, units=tuple(next_units), resources=0)
        tactic.choose_actions(next_turn)
        next_schedule = tactic.last_decision_trace["economy"]["service_queue"][
            "scheduled_deposits"
        ]
        self.assertEqual(next_schedule, schedule)

    def test_remote_worker_stall_does_not_replace_a_valid_service_lane(self) -> None:
        lane = ((0, 1), (0, 2))
        queued = (
            unit(1, UnitType.WORKER, lane[-1], cargo=1),
            unit(2, UnitType.WORKER, lane[-1], cargo=1),
        )
        remote = unit(3, UnitType.WORKER, (18, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            opening_complete=True,
            service_entrance=lane[0],
            service_queue_cells=lane,
            service_exit_cell=(0, -1),
            service_return_progress={remote.id: (20, 4)},
        )
        memory.known_passable.update(
            (x, y) for x in range(-20, 21) for y in range(-6, 7)
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=50, units=(*queued, remote), resources=0)
        )

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["queue_cells"], [list(cell) for cell in lane])
        self.assertEqual(tactic.memory.service_queue_cells, lane)
        self.assertEqual(queue["lane_lease"]["version"], 1)
        self.assertIsNone(queue["lane_replan_reason"])

    def test_long_authoritative_return_route_is_not_rejected_by_short_recheck(self) -> None:
        lane = ((1, 0), (2, 0))
        carrier = unit(1, UnitType.WORKER, (600, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            opening_complete=True,
            service_entrance=lane[0],
            service_queue_cells=lane,
            service_exit_cell=(0, -1),
        )
        memory.known_passable.update((x, 0) for x in range(0, 601))
        memory.known_passable.update({(0, -1), (0, 1), (1, -1), (1, 1)})
        tactic = BalancedTactic(memory=memory)

        turn = make_turn(tick=70, units=(carrier,), resources=0)
        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        reservation = next(
            row
            for row in queue["return_reservations"]
            if row["worker_id"] == str(carrier.id)
        )
        self.assertGreater(reservation["route_distance"], 512)
        action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.LEFT)

    def test_segmented_return_keeps_moving_when_full_route_exceeds_budget(self) -> None:
        lane = ((1, 0), (2, 0))
        carrier = unit(1, UnitType.WORKER, (100, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            opening_complete=True,
            service_entrance=lane[0],
            service_queue_cells=lane,
            service_exit_cell=(0, -1),
        )
        memory.known_passable.update((x, 0) for x in range(0, 101))
        memory.known_passable.update({(0, -1), (0, 1), (1, -1), (1, 1)})
        tactic = BalancedTactic(
            config=TacticConfig(path_node_limit=32),
            memory=memory,
        )

        turn = make_turn(tick=80, units=(carrier,), resources=0)
        tactic.choose_actions(turn)

        reservation = tactic.last_decision_trace["economy"]["service_queue"][
            "return_reservations"
        ][0]
        self.assertEqual(reservation["route_mode"], "SEGMENTED")
        self.assertEqual(reservation["status"], "SEGMENTED_RETURN")
        self.assertIsNotNone(reservation["waypoint"])
        action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.LEFT)

        first_waypoint = tuple(reservation["waypoint"])
        moved = unit(1, UnitType.WORKER, (99, 0), cargo=1)
        next_turn = make_turn(tick=81, units=(moved,), resources=0)
        tactic.choose_actions(next_turn)
        next_reservation = tactic.last_decision_trace["economy"]["service_queue"][
            "return_reservations"
        ][0]
        self.assertEqual(tuple(next_reservation["waypoint"]), first_waypoint)
        self.assertIsInstance(next_turn.plan.unit_actions[moved.id], MoveAction)

    def test_new_nearby_cargo_fills_free_tick_without_delaying_old_appointment(self) -> None:
        far = unit(1, UnitType.WORKER, (8, 0), cargo=1)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-10, 11)
            for y in range(-10, 11)
        )
        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(make_turn(tick=100, units=(far,), resources=0))
        first_schedule = {
            row["worker_id"]: row["tick"]
            for row in tactic.last_decision_trace["economy"]["service_queue"][
                "scheduled_deposits"
            ]
        }

        # Simulate the first planned return step so the old appointment remains
        # physically achievable on the next Tick.
        far = unit(1, UnitType.WORKER, (7, 0), cargo=1)
        near = unit(2, UnitType.WORKER, (1, 0), cargo=1)
        tactic.choose_actions(
            make_turn(tick=101, units=(far, near), resources=0)
        )
        second_schedule = {
            row["worker_id"]: row["tick"]
            for row in tactic.last_decision_trace["economy"]["service_queue"][
                "scheduled_deposits"
            ]
        }

        self.assertEqual(second_schedule[str(far.id)], first_schedule[str(far.id)])
        self.assertLess(second_schedule[str(near.id)], second_schedule[str(far.id)])

    def test_distance_five_cargo_fills_gap_before_distance_ten_appointment(self) -> None:
        far = unit(1, UnitType.WORKER, (10, 0), cargo=1)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y)
            for x in range(-12, 13)
            for y in range(-6, 7)
        )
        tactic = BalancedTactic(memory=memory)
        tactic.choose_actions(make_turn(tick=100, units=(far,), resources=0))
        first = tactic.last_decision_trace["economy"]["service_queue"]
        far_reservation = first["return_reservations"][0]
        self.assertEqual(far_reservation["route_distance"], 10)
        self.assertEqual(far_reservation["scheduled_deposit_tick"], 110)

        moved_far = unit(1, UnitType.WORKER, (9, 0), cargo=1)
        near = unit(2, UnitType.WORKER, (5, 0), cargo=1)
        tactic.choose_actions(
            make_turn(tick=101, units=(moved_far, near), resources=0)
        )
        reservations = {
            row["worker_id"]: row
            for row in tactic.last_decision_trace["economy"]["service_queue"][
                "return_reservations"
            ]
        }
        self.assertEqual(
            reservations[str(moved_far.id)]["scheduled_deposit_tick"],
            110,
        )
        self.assertEqual(reservations[str(near.id)]["route_distance"], 5)
        self.assertLess(
            reservations[str(near.id)]["scheduled_deposit_tick"],
            reservations[str(moved_far.id)]["scheduled_deposit_tick"],
        )

    def test_overdue_cargo_is_rescheduled_from_its_real_eta(self) -> None:
        far = unit(1, UnitType.WORKER, (8, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            opening_complete=True,
            service_deposit_ticks={far.id: 99},
            service_cargo_first_seen_ticks={far.id: 90},
        )
        memory.known_passable.update(
            (x, y)
            for x in range(-10, 11)
            for y in range(-10, 11)
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=101, units=(far,), resources=0)
        )

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertIn(str(far.id), queue["approaching_depositors"])
        reservation = next(
            row
            for row in queue["return_reservations"]
            if row["worker_id"] == str(far.id)
        )
        self.assertGreaterEqual(
            reservation["scheduled_deposit_tick"],
            reservation["earliest_deposit_tick"],
        )
        self.assertEqual(reservation["delay_reason"], "MISSED_APPOINTMENT")
        self.assertGreater(queue["timeline"]["next_service_eta"], 0)
        self.assertNotEqual(
            queue["timeline"]["reason"],
            "SERVICE_DUE_THIS_TICK",
        )
        far_task = next(
            task
            for task in tactic.last_decision_trace["tasks"]
            if task["actor_id"] == str(far.id)
        )
        self.assertEqual(far_task["action"], "MOVE")
        self.assertEqual(far_task["reason"], "SERVICE_QUEUE_APPROACH")

    def test_narrow_core_corridor_allows_only_the_service_handoff_swap(self) -> None:
        empty = unit(1, UnitType.WORKER, (0, 0))
        carrier = unit(2, UnitType.WORKER, (1, 0), cargo=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_admission_id=carrier.id,
            service_kind="DEPOSIT",
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(1, 0),
            cargo_arrival_ticks={carrier.id: 1},
        )
        turn = make_turn(
            tick=2,
            units=(empty, carrier),
            resources=0,
            obstacle_cells=((-1, 0), (0, -1), (0, 1)),
        )

        BalancedTactic(memory=memory).choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[empty.id], MoveAction)
        self.assertEqual(turn.plan.unit_actions[empty.id].direction, Direction.RIGHT)
        self.assertIsInstance(turn.plan.unit_actions[carrier.id], MoveAction)
        self.assertEqual(turn.plan.unit_actions[carrier.id].direction, Direction.LEFT)

    def test_clear_core_worker_does_not_swap_with_an_exploring_worker(self) -> None:
        clearing = unit(1, UnitType.WORKER, (0, 0))
        exploring = unit(2, UnitType.WORKER, (1, 0))
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(0, -1),
            service_queue_cells=((0, -1), (0, -2)),
            service_exit_cell=(1, 0),
            unit_missions={
                exploring.id: MissionState(UnitMission.EXPLORE, (-6, 0), 1)
            },
        )
        memory.known_passable.update((x, 0) for x in range(-6, 2))
        turn = make_turn(
            tick=2,
            units=(clearing, exploring),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        clearing_action = turn.plan.unit_actions[clearing.id]
        exploring_action = turn.plan.unit_actions[exploring.id]
        self.assertIsInstance(clearing_action, MoveAction)
        clear_dx, clear_dy = clearing_action.direction.delta
        clearing_destination = (
            clearing.position[0] + clear_dx,
            clearing.position[1] + clear_dy,
        )
        exploring_destination = exploring.position
        if isinstance(exploring_action, MoveAction):
            explore_dx, explore_dy = exploring_action.direction.delta
            exploring_destination = (
                exploring.position[0] + explore_dx,
                exploring.position[1] + explore_dy,
            )
        self.assertNotEqual(clearing_destination, clearing.position)
        self.assertFalse(
            clearing_destination == exploring.position
            and exploring_destination == clearing.position,
            "CLEAR_CORE and ordinary exploration must not exchange cells",
        )

    def test_empty_worker_reports_a_blocked_core_exit(self) -> None:
        empty = unit(1, UnitType.WORKER, (0, 0))
        turn = make_turn(
            units=(empty,),
            resources=0,
            obstacle_cells=((-1, 0), (1, 0), (0, -1), (0, 1)),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(empty.id)
        )
        self.assertEqual(task["mission"], "CLEAR_CORE")
        self.assertEqual(task["reason"], "CORE_EXIT_BLOCKED")

    def test_four_readying_carriers_finish_within_ten_ticks(self) -> None:
        positions = {1: (0, 1), 2: (0, 2), 3: (1, 3), 4: (-1, 3)}
        cargo = {identifier: 1 for identifier in positions}
        resources = 0
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(0, 1),
            service_queue_cells=((0, 1), (0, 2)),
            service_exit_cell=(0, -1),
        )
        tactic = BalancedTactic(memory=memory)

        for tick in range(1, 11):
            workers = tuple(
                unit(
                    identifier,
                    UnitType.WORKER,
                    positions[identifier],
                    cargo=cargo[identifier],
                )
                for identifier in sorted(positions)
            )
            turn = make_turn(tick=tick, units=workers, resources=resources)
            tactic.choose_actions(turn)
            prior_positions = dict(positions)
            for identifier in sorted(positions):
                action = turn.plan.unit_actions[uid(identifier)]
                if isinstance(action, MoveAction):
                    dx, dy = action.direction.delta
                    x, y = prior_positions[identifier]
                    positions[identifier] = (x + dx, y + dy)
                elif isinstance(action, DepositAction):
                    cargo[identifier] = 0
                    resources += 1
            if not any(cargo.values()):
                break

        self.assertEqual(cargo, {1: 0, 2: 0, 3: 0, 4: 0})
        self.assertLessEqual(tick, 10)

    def test_full_core_does_not_repeat_a_failed_deposit(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        turn = make_turn(units=(carrier,), resources=10)

        BalancedTactic().choose_actions(turn)

        self.assertNotIsInstance(turn.plan.unit_actions[carrier.id], DepositAction)
        self.assertIsInstance(turn.plan.unit_actions[carrier.id], MoveAction)

    def test_unit_can_leave_core_and_free_same_tick_spawn_slot(self) -> None:
        workers = (
            unit(1, UnitType.WORKER, (0, 0)),
            unit(2, UnitType.WORKER, (2, 0)),
            unit(3, UnitType.WORKER, (0, 2)),
            unit(4, UnitType.WORKER, (-2, 0)),
        )
        turn = make_turn(units=workers, resources=10)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[workers[0].id], MoveAction)
        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.VANGUARD)

    def test_combat_unit_releases_core_for_ready_carrier_same_tick(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, -1), cargo=1)
        ranger = unit(2, UnitType.RANGER, (0, 0))
        turn = make_turn(
            units=(carrier, ranger),
            resources=0,
            obstacle_cells=((-1, 0), (0, 1)),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[ranger.id], MoveAction)
        self.assertEqual(turn.plan.unit_actions[ranger.id].direction, Direction.RIGHT)
        self.assertIsInstance(turn.plan.unit_actions[carrier.id], MoveAction)

    def test_safe_core_cargo_deposits_before_urgent_patient_enters(self) -> None:
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(-1, 0),
        )
        carrier = unit(1, UnitType.WORKER, (0, 0), cargo=1)
        patient = unit(2, UnitType.RANGER, (1, 0), hp=1)
        turn = make_turn(units=(carrier, patient), resources=1)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        carrier_action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(carrier_action, DepositAction)
        self.assertNotEqual(tactic.memory.service_admission_id, patient.id)

    def test_patrol_routes_around_protected_service_cells(self) -> None:
        ring = {
            (x, y)
            for x in range(-5, 6)
            for y in range(-5, 6)
            if abs(x) + abs(y) == 5
        }
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            visit_counts=Counter({cell: 100 for cell in ring}),
        )
        memory.visit_counts[(-5, 0)] = 0
        carrier = unit(1, UnitType.WORKER, (0, -1), cargo=1)
        vanguard = unit(2, UnitType.VANGUARD, (2, 0))
        turn = make_turn(units=(carrier, vanguard), resources=0)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[vanguard.id]
        self.assertIsInstance(action, MoveAction)
        self.assertNotEqual(action.direction, Direction.LEFT)
        rejected = tactic.last_decision_trace["resolution"]["rejected"]
        self.assertFalse(
            any(
                row["intent"]["actor_id"] == str(vanguard.id)
                and row["reason"] == "STATIC_CONFLICT"
                for row in rejected
            )
        )

    def test_patrol_cannot_reenter_core_after_service_egress(self) -> None:
        ring = {
            (x, y)
            for x in range(-5, 6)
            for y in range(-5, 6)
            if abs(x) + abs(y) == 5
        }
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            visit_counts=Counter({cell: 100 for cell in ring}),
        )
        # Force the squad's least-visited anchor to the far side of the Core.
        # The shortest geometric route from (1, 0) starts LEFT through Core,
        # which ordinary patrol is never allowed to use as a shortcut.
        memory.visit_counts[(-5, 0)] = 0
        carrier = unit(1, UnitType.WORKER, (0, -1), cargo=1)
        vanguard = unit(2, UnitType.VANGUARD, (1, 0))
        ranger = unit(3, UnitType.RANGER, (2, 1))
        turn = make_turn(
            units=(carrier, vanguard, ranger),
            resources=0,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[vanguard.id]
        self.assertFalse(
            isinstance(action, MoveAction) and action.direction is Direction.LEFT,
            "ordinary patrol must not move back into the Core service slot",
        )

    def test_admitted_wounded_unit_cannot_fall_back_to_patrol_when_route_blocks(self) -> None:
        wounded = unit(2, UnitType.VANGUARD, (2, 0), hp=3)
        turn = make_turn(
            units=(wounded,),
            resources=5,
            obstacle_cells=((1, 0), (3, 0), (2, -1), (2, 1)),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[wounded.id], WaitAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(wounded.id)
        )
        self.assertEqual(task["mission"], UnitMission.RECOVER.value)
        self.assertEqual(task["reason"], "RECOVERY_ROUTE_BLOCKED_THIS_TICK")

    def test_remote_mild_patient_is_future_work_not_current_admission(self) -> None:
        wounded = unit(2, UnitType.VANGUARD, (3, 0), hp=3)
        turn = make_turn(units=(wounded,), resources=1)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertIsNone(tactic.memory.service_admission_id)
        self.assertIsInstance(turn.plan.unit_actions[wounded.id], MoveAction)
        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertNotEqual(queue["service"], "MAINTENANCE_HEAL")
        self.assertFalse(queue["core_slot_reserved"])
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(wounded.id)
        )
        self.assertEqual(task["mission"], UnitMission.RECOVER.value)

    def test_wounded_core_occupant_is_healed_before_adjacent_cargo_enters(self) -> None:
        carrier = unit(1, UnitType.WORKER, (1, 0), cargo=1)
        patient = unit(2, UnitType.VANGUARD, (0, 0), hp=3)
        turn = make_turn(units=(carrier, patient), resources=1)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertEqual(tactic.memory.service_admission_id, patient.id)
        self.assertIsInstance(turn.plan.unit_actions[patient.id], HealAction)
        self.assertNotIsInstance(turn.plan.unit_actions[carrier.id], MoveAction)

    def test_future_carrier_clears_active_service_cell_instead_of_waiting(self) -> None:
        carrier = unit(1, UnitType.WORKER, (-1, 0), cargo=1)
        patient = unit(2, UnitType.RANGER, (-2, 0), hp=1)
        memory = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            service_entrance=(1, 0),
            service_queue_cells=((1, 0), (2, 0)),
            service_exit_cell=(-1, 0),
            service_deposit_ticks={carrier.id: 20},
        )
        turn = make_turn(tick=10, units=(carrier, patient), resources=2)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(action, MoveAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(carrier.id)
        )
        self.assertEqual(task["reason"], "CLEAR_FUTURE_SERVICE_CELL")

    def test_ready_cargo_precedes_mild_maintenance_treatment(self) -> None:
        carrier = unit(1, UnitType.WORKER, (1, 0), cargo=1)
        wounded = unit(2, UnitType.VANGUARD, (3, 0), hp=3)
        turn = make_turn(units=(carrier, wounded), resources=1)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertEqual(tactic.memory.service_admission_id, carrier.id)

    def test_remote_emergency_patient_reserves_funds_without_freezing_ready_cargo(self) -> None:
        carrier = unit(1, UnitType.WORKER, (1, 0), cargo=1)
        patient = unit(2, UnitType.RANGER, (8, 0), hp=1)
        turn = make_turn(units=(carrier, patient), resources=1)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertEqual(tactic.memory.service_admission_id, carrier.id)
        self.assertIsInstance(turn.plan.unit_actions[carrier.id], MoveAction)
        patient_task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(patient.id)
        )
        self.assertEqual(patient_task["mission"], UnitMission.RECOVER.value)

    def test_adjacent_maintenance_patient_fills_gap_before_far_work(self) -> None:
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y) for x in range(-12, 13) for y in range(-12, 13)
        )
        carrier = unit(1, UnitType.WORKER, (5, 0), cargo=1)
        far_urgent = unit(2, UnitType.RANGER, (8, 0), hp=1)
        near_maintenance = unit(3, UnitType.VANGUARD, (0, 1), hp=3)
        turn = make_turn(
            tick=100,
            units=(carrier, far_urgent, near_maintenance),
            resources=5,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["admission_id"], str(near_maintenance.id))
        action = turn.plan.unit_actions[near_maintenance.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.UP)

    def test_ready_urgent_patient_preempts_sticky_remote_patient_and_cargo(self) -> None:
        remote_id = uid(2)
        memory = TacticMemory(
            opening_complete=True,
            patient_admission_progress=PatientAdmissionProgress(
                patient_id=remote_id,
                gateway=(1, 0),
                started_tick=90,
                last_position=(8, 0),
                stalled_ticks=0,
                entry_distance=8,
            ),
        )
        memory.known_passable.update(
            (x, y) for x in range(-12, 13) for y in range(-12, 13)
        )
        carrier = unit(1, UnitType.WORKER, (1, 0), cargo=1)
        remote_urgent = unit(2, UnitType.RANGER, (8, 0), hp=1)
        ready_urgent = unit(3, UnitType.RANGER, (0, 1), hp=1)
        turn = make_turn(
            tick=100,
            units=(carrier, remote_urgent, ready_urgent),
            resources=2,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["admission_id"], str(ready_urgent.id))
        action = turn.plan.unit_actions[ready_urgent.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.UP)

    def test_every_injured_worker_keeps_a_core_service_job_and_moves_home(self) -> None:
        near_patient = unit(2, UnitType.RANGER, (1, 0), hp=1)
        injured_worker = unit(3, UnitType.WORKER, (0, 5), hp=1)
        memory = TacticMemory(opening_complete=True)
        memory.known_passable.update(
            (x, y) for x in range(-2, 3) for y in range(-1, 7)
        )
        turn = make_turn(
            units=(near_patient, injured_worker),
            resources=5,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        worker_jobs = [
            job
            for job in queue["jobs"]
            if job["actor_id"] == str(injured_worker.id)
        ]
        self.assertEqual(len(worker_jobs), 1)
        self.assertIn("HEAL", worker_jobs[0]["operations"])
        action = turn.plan.unit_actions[injured_worker.id]
        self.assertIsInstance(action, MoveAction)
        dx, dy = action.direction.delta
        destination = (
            injured_worker.position[0] + dx,
            injured_worker.position[1] + dy,
        )
        self.assertLess(
            manhattan(destination, (0, 0)),
            manhattan(injured_worker.position, (0, 0)),
        )

    def test_wounded_loaded_worker_has_one_compound_core_service_job(self) -> None:
        worker = unit(1, UnitType.WORKER, (1, 0), hp=1, cargo=1)
        turn = make_turn(units=(worker,), resources=0)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        jobs = [job for job in queue["jobs"] if job["actor_id"] == str(worker.id)]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["operations"], ["DEPOSIT", "HEAL"])

    def test_underfunded_core_patient_yields_to_one_funding_carrier(self) -> None:
        patient = unit(2, UnitType.RANGER, (0, 0), hp=1)
        carrier = unit(1, UnitType.WORKER, (1, 0), cargo=1)
        turn = make_turn(units=(patient, carrier), resources=0)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[patient.id], MoveAction)
        patient_task = next(
            task
            for task in tactic.last_decision_trace["tasks"]
            if task["actor_id"] == str(patient.id)
        )
        self.assertEqual(patient_task["reason"], "PATIENT_YIELD_FOR_FUNDING")
        queue = tactic.last_decision_trace["economy"]["service_queue"]
        jobs = queue["jobs"]
        self.assertEqual(jobs[0]["actor_id"], str(carrier.id))
        self.assertEqual(jobs[0]["operations"], ["DEPOSIT"])

    def test_admitted_patient_can_traverse_the_only_service_entrance(self) -> None:
        patient = unit(2, UnitType.RANGER, (2, 0), hp=1)
        turn = make_turn(
            units=(patient,),
            resources=1,
            obstacle_cells=((0, -1), (-1, 0), (0, 1), (1, -1), (1, 1)),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["timeline"]["requests"][0]["actor_id"], str(patient.id))
        self.assertEqual(queue["timeline"]["requests"][0]["operation"], "HEAL")
        action = turn.plan.unit_actions[patient.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.LEFT)

    def test_pre25_combat_floor_fills_the_larger_type_gap(self) -> None:
        memory = TacticMemory(opening_complete=True)
        units = tuple(
            [unit(index + 1, UnitType.WORKER, (index + 2, 2)) for index in range(4)]
            + [
                unit(20, UnitType.VANGUARD, (-2, 1)),
                unit(21, UnitType.VANGUARD, (-2, 2)),
            ]
            + [unit(30, UnitType.RANGER, (2, 1))]
        )
        turn = make_turn(units=units, resources=20)

        BalancedTactic(memory=memory).choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.RANGER)

    def test_reinforcement_uses_configured_asymmetric_type_floors(self) -> None:
        config = TacticConfig(minimum_vanguards=2, minimum_rangers=4)
        units = tuple(
            [unit(index + 1, UnitType.WORKER, (index + 4, 3)) for index in range(6)]
            + [
                unit(20, UnitType.VANGUARD, (-2, 1)),
                unit(21, UnitType.VANGUARD, (-2, 2)),
            ]
            + [
                unit(30, UnitType.RANGER, (2, 1)),
                unit(31, UnitType.RANGER, (2, 2)),
                unit(32, UnitType.RANGER, (2, 3)),
            ]
        )
        turn = make_turn(units=units, resources=100)

        BalancedTactic(
            config,
            memory=TacticMemory(opening_complete=True),
        ).choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.RANGER)

    def test_reinforcement_falls_back_to_affordable_missing_type_after_heal_reserve(self) -> None:
        workers = [
            unit(index, UnitType.WORKER, (20 + index, 0), hp=1 if index == 1 else 2)
            for index in range(1, 6)
        ]
        units = tuple(
            workers
            + [
                unit(20 + index, UnitType.VANGUARD, (-index, 2))
                for index in range(1, 4)
            ]
            + [unit(30, UnitType.RANGER, (2, 2))]
        )
        turn = make_turn(units=units, resources=11)
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.core_action, SpawnAction)
        self.assertEqual(turn.plan.core_action.unit_type, UnitType.VANGUARD)
        selected = next(
            item
            for item in tactic.last_decision_trace["economy"]["production_candidates"]
            if item.get("selected")
        )
        self.assertEqual(selected["unit_type"], UnitType.VANGUARD.value)
        self.assertEqual(selected["reserved_for_recovery"], 1)
        self.assertEqual(selected["fallback_reason"], "PRIMARY_UNAFFORDABLE")

    def test_injured_empty_worker_enters_recovery_instead_of_exploring(self) -> None:
        worker = unit(1, UnitType.WORKER, (2, 0), hp=1)
        turn = make_turn(units=(worker,), resources=1)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["timeline"]["requests"][0]["actor_id"], str(worker.id))
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertEqual(task["mission"], UnitMission.RECOVER.value)
        self.assertEqual(
            tactic.memory.unit_missions[worker.id].mission,
            UnitMission.RECOVER,
        )
        self.assertNotIn(
            str(worker.id),
            {
                item["worker_id"]
                for item in tactic.last_decision_trace["economy"]["resource_assignments"]
            },
        )

    def test_injured_loaded_worker_deposits_then_heals_next_tick(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0), hp=1, cargo=1)
        tactic = BalancedTactic()
        first = make_turn(tick=1, units=(worker,), resources=0)

        tactic.choose_actions(first)

        self.assertIsInstance(first.plan.unit_actions[worker.id], DepositAction)
        second = make_turn(
            tick=2,
            units=(unit(1, UnitType.WORKER, (0, 0), hp=1, cargo=0),),
            resources=1,
        )
        tactic.choose_actions(second)
        self.assertIsInstance(second.plan.unit_actions[worker.id], HealAction)

    def test_lethal_core_line_makes_injured_carrier_escape_before_deposit(self) -> None:
        carrier = unit(1, UnitType.WORKER, (0, 0), hp=1, cargo=1)
        enemy = unit(100, UnitType.RANGER, (0, 3), controlled=False)
        turn = make_turn(units=(carrier,), enemies=(enemy,), resources=0)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[carrier.id], MoveAction)

    def test_all_injured_workers_reserve_their_missing_healing_resources(self) -> None:
        workers = (
            unit(1, UnitType.WORKER, (2, 0), hp=1),
            unit(2, UnitType.WORKER, (-2, 0), hp=1),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(make_turn(units=workers, resources=10))

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertEqual(queue["reserved_resources"], 2)

    def test_injured_worker_with_no_survivable_step_waits_explicitly(self) -> None:
        worker = unit(1, UnitType.WORKER, (2, 0), hp=1)
        turn = make_turn(
            units=(worker,),
            resources=1,
            obstacle_cells=((1, 0), (3, 0), (2, -1), (2, 1)),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[worker.id], WaitAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertEqual(task["reason"], "NO_SURVIVABLE_RECOVERY_STEP")

    def test_loaded_worker_route_prefers_detour_around_persistent_heat(self) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (2, 0), cargo=1)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            threat_heat={
                (1, 0): ThreatHeatCell(
                    position=(1, 0),
                    risk=24,
                    updated_tick=1,
                    expires_tick=65,
                    source="UNIT_DESTROYED",
                )
            },
        )
        turn = make_turn(tick=2, core=core, units=(worker,), resources=0)

        BalancedTactic(memory=memory).choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        self.assertNotEqual(action.direction, Direction.LEFT)

    def test_one_hp_recovery_avoids_a_single_forward_exit_when_detour_exists(self) -> None:
        core = friendly_core(position=(0, 0))
        passable = {
            (4, 0),
            (3, 0),
            (2, 0),
            (1, 0),
            (0, 0),
            (4, 1),
            (4, 2),
            (3, 2),
            (2, 2),
            (1, 2),
            (0, 2),
            (0, 1),
        }
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            known_passable=set(passable),
            known_obstacles={(3, -1), (3, 1)},
        )
        worker = unit(1, UnitType.WORKER, (4, 0), hp=1)
        turn = make_turn(
            core=core,
            units=(worker,),
            resources=1,
            obstacle_cells=((3, -1), (3, 1)),
        )

        BalancedTactic(memory=memory).choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        self.assertNotEqual(action.direction, Direction.LEFT)

    def test_injured_worker_detours_instead_of_closing_on_remote_enemy(self) -> None:
        core = friendly_core(position=(20, 0))
        worker = unit(1, UnitType.WORKER, (0, 0), hp=1)
        observer = unit(2, UnitType.RANGER, (8, 2))
        enemy = unit(100, UnitType.RANGER, (8, 1), controlled=False)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            opening_complete=True,
            known_passable={
                (x, y)
                for x in range(-2, 23)
                for y in range(-8, 9)
            },
        )
        turn = make_turn(
            core=core,
            units=(worker, observer),
            enemies=(enemy,),
            resources=1,
        )

        BalancedTactic(memory=memory).choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        self.assertNotIn(action.direction, {Direction.RIGHT, Direction.DOWN})

    def test_global_resource_matching_is_one_to_one_and_nearest(self) -> None:
        workers = (
            unit(1, UnitType.WORKER, (-5, 0)),
            unit(2, UnitType.WORKER, (5, 0)),
        )
        turn = make_turn(
            units=workers,
            resources=0,
            resource_cells=((-7, 0), (7, 0)),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        assignments = {
            item["worker_id"]: tuple(item["target"])
            for item in tactic.last_decision_trace["economy"]["resource_assignments"]
        }
        self.assertEqual(assignments[str(workers[0].id)], (-7, 0))
        self.assertEqual(assignments[str(workers[1].id)], (7, 0))

    def test_outside_band_worker_takes_adjacent_resource_before_scout_recall(self) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (59, 0))
        resource = (60, 0)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            known_passable={
                (x, y)
                for x in range(-5, 66)
                for y in range(-35, 36)
            },
            unit_missions={
                worker.id: MissionState(UnitMission.EXPLORE, (59, -10), 1)
            },
            opening_complete=True,
        )
        turn = make_turn(
            tick=10,
            core=core,
            units=(worker,),
            resources=0,
            resource_cells=(resource,),
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(turn)

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertEqual(task["mission"], UnitMission.HARVEST.value)
        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        self.assertEqual(action.direction, Direction.RIGHT)

    def test_non_full_economy_assigns_all_remembered_resources_beyond_scout_band(
        self,
    ) -> None:
        core = friendly_core(position=(0, 0))
        workers = (
            unit(1, UnitType.WORKER, (20, 0)),
            unit(2, UnitType.WORKER, (0, 20)),
            unit(3, UnitType.WORKER, (-20, 0)),
            unit(4, UnitType.WORKER, (0, -20)),
        )
        resources = ((31, 0), (0, 34), (-35, 0), (0, -35))
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            known_passable={
                (x, y)
                for x in range(-40, 41)
                for y in range(-40, 41)
            },
            resource_memory={cell: 5 for cell in resources},
            opening_complete=True,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=10, core=core, units=workers, resources=5)
        )

        assignments = {
            tuple(row["target"])
            for row in tactic.last_decision_trace["economy"]["resource_assignments"]
            if row["mission"] == UnitMission.HARVEST.value
        }
        self.assertEqual(assignments, set(resources))
        self.assertFalse(tactic.last_decision_trace["economy"]["storage_full_now"])
        self.assertEqual(
            tactic.last_decision_trace["economy"]["worker_economy_mode_counts"],
            {WorkerEconomyMode.RESOURCE_ACQUISITION.value: 4},
        )

    def test_resource_assignment_over_one_hundred_cells_is_not_scout_limited(
        self,
    ) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (1, 0))
        target = (101, 0)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            known_passable={
                (x, y)
                for x in range(-2, 103)
                for y in range(-3, 4)
            },
            resource_memory={target: 1},
            opening_complete=True,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=10, core=core, units=(worker,), resources=0)
        )

        action = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertEqual(action["mission"], UnitMission.HARVEST.value)
        self.assertEqual(tuple(action["target"]), (2, 0))
        self.assertEqual(
            tactic.memory.worker_economy_modes[worker.id],
            WorkerEconomyMode.RESOURCE_ACQUISITION,
        )

    def test_worker_economy_modes_separate_search_from_saturated_patrol(self) -> None:
        worker = unit(1, UnitType.WORKER, (2, 0))
        tactic = BalancedTactic(memory=TacticMemory(opening_complete=True))

        tactic.choose_actions(make_turn(tick=1, units=(worker,), resources=0))
        self.assertEqual(
            tactic.memory.worker_economy_modes[worker.id],
            WorkerEconomyMode.RESOURCE_SEARCH,
        )
        self.assertFalse(tactic.memory.storage_saturated)

        tactic.choose_actions(make_turn(tick=2, units=(worker,), resources=10))
        self.assertEqual(
            tactic.memory.worker_economy_modes[worker.id],
            WorkerEconomyMode.RESOURCE_SEARCH,
        )
        self.assertTrue(tactic.memory.storage_saturated)

        tactic.choose_actions(make_turn(tick=3, units=(worker,), resources=9))
        self.assertEqual(
            tactic.memory.worker_economy_modes[worker.id],
            WorkerEconomyMode.RESOURCE_SEARCH,
        )

        tactic.choose_actions(make_turn(tick=4, units=(worker,), resources=5))
        self.assertEqual(
            tactic.memory.worker_economy_modes[worker.id],
            WorkerEconomyMode.RESOURCE_SEARCH,
        )
        self.assertFalse(tactic.memory.storage_saturated)

    def test_temporary_full_storage_preserves_empty_worker_resource_order(self) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (2, 0))
        target = (8, 0)
        order = ResourceWorkOrder(
            worker_id=worker.id,
            target=target,
            assigned_tick=1,
            last_confirmed_tick=1,
            last_route_distance=6,
        )
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            known_passable={(x, y) for x in range(-2, 10) for y in range(-2, 3)},
            resource_memory={target: 1},
            resource_work_orders={worker.id: order},
            opening_complete=True,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=2, core=core, units=(worker,), resources=10)
        )

        self.assertEqual(tactic.memory.resource_work_orders[worker.id], order)
        self.assertEqual(
            tactic.memory.worker_economy_modes[worker.id],
            WorkerEconomyMode.RESOURCE_ACQUISITION,
        )
        self.assertFalse(
            tactic.last_decision_trace["economy"]["saturated_patrol_active"]
        )

    def test_temporary_full_service_block_keeps_empty_worker_economy_task(self) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (2, 0))
        patient = unit(2, UnitType.VANGUARD, (0, 0), hp=3)
        target = (8, 0)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            known_passable={(x, y) for x in range(-2, 10) for y in range(-2, 3)},
            resource_memory={target: 1},
            resource_work_orders={
                worker.id: ResourceWorkOrder(
                    worker_id=worker.id,
                    target=target,
                    assigned_tick=1,
                    last_confirmed_tick=1,
                    last_route_distance=6,
                )
            },
            opening_complete=True,
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(
            tick=2,
            core=core,
            units=(worker, patient),
            resources=10,
        )

        tactic.choose_actions(turn)

        self.assertNotIsInstance(turn.plan.core_action, SpawnAction)
        self.assertIn(worker.id, tactic.memory.resource_work_orders)
        self.assertEqual(
            tactic.memory.worker_economy_modes[worker.id],
            WorkerEconomyMode.RESOURCE_ACQUISITION,
        )
        self.assertFalse(
            tactic.last_decision_trace["economy"]["saturated_patrol_active"]
        )

    def test_loaded_worker_clears_remote_order_and_returns_toward_core(self) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (100, 0), cargo=1)
        target = (101, 0)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            known_passable={
                (x, y)
                for x in range(-3, 103)
                for y in range(-4, 5)
            },
            resource_memory={target: 1},
            opening_complete=True,
        )
        memory.resource_work_orders[worker.id] = ResourceWorkOrder(
            worker_id=worker.id,
            target=target,
            assigned_tick=1,
            last_confirmed_tick=1,
            last_route_distance=1,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=2, core=core, units=(worker,), resources=0)
        )

        action = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertEqual(action["mission"], UnitMission.RETURN_CARGO.value)
        self.assertNotIn(worker.id, tactic.memory.resource_work_orders)
        planned = next(
            row["final"]
            for row in tactic.last_decision_trace["decisions"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertEqual(planned["mission"], UnitMission.RETURN_CARGO.value)

    def test_reachable_resource_work_order_does_not_thrash_for_a_new_nearby_node(self) -> None:
        tactic = BalancedTactic()
        worker = unit(1, UnitType.WORKER, (0, 0))
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=friendly_core(position=(0, -2)),
                units=(worker,),
                resource_cells=((3, 0),),
                resources=0,
            )
        )
        self.assertEqual(tactic.memory.unit_missions[worker.id].target, (3, 0))

        tactic.choose_actions(
            make_turn(
                tick=2,
                core=friendly_core(position=(0, -2)),
                units=(unit(1, UnitType.WORKER, (1, 0)),),
                resource_cells=((3, 0), (1, 1)),
                resources=0,
            )
        )

        assignment = next(
            item
            for item in tactic.last_decision_trace["economy"]["resource_assignments"]
            if item["worker_id"] == str(worker.id)
        )
        self.assertEqual(tuple(assignment["target"]), (3, 0))

    def test_global_resource_matching_beats_uuid_ordered_greedy_choice(self) -> None:
        worker_ids = (uid(1), uid(2))
        resources = ((1, 0), (0, 2))
        costs = {
            (worker_ids[0], resources[0]): 1,
            (worker_ids[0], resources[1]): 2,
            (worker_ids[1], resources[0]): 1,
            (worker_ids[1], resources[1]): 4,
        }

        result = minimum_cost_matching(worker_ids, resources, costs)

        self.assertEqual(
            {(worker_id, resource) for worker_id, resource, _ in result},
            {
                (worker_ids[0], resources[1]),
                (worker_ids[1], resources[0]),
            },
        )
        self.assertEqual(sum(cost for _, _, cost in result), 3)

    def test_new_visible_resource_does_not_erase_a_fogged_remembered_node(self) -> None:
        tactic = BalancedTactic()
        first_worker = unit(1, UnitType.WORKER, (0, 0))
        tactic.choose_actions(
            make_turn(
                tick=1,
                core=friendly_core(position=(-10, 0)),
                units=(first_worker,),
                resources=0,
                resource_cells=((3, 0),),
            )
        )

        second_worker = unit(1, UnitType.WORKER, (-1, 0))
        tactic.choose_actions(
            make_turn(
                tick=2,
                core=friendly_core(position=(-10, 0)),
                units=(second_worker,),
                resources=0,
                resource_cells=((-1, 2),),
            )
        )

        self.assertIn((3, 0), tactic.memory.resource_memory)
        self.assertIn((-1, 2), tactic.memory.resource_memory)

    def test_information_gain_goal_persists_until_observed(self) -> None:
        worker = unit(1, UnitType.WORKER, (1, 0))
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(tick=1, units=(worker,), resources=0))
        first = tactic.memory.unit_missions[worker.id]

        tactic.choose_actions(make_turn(tick=2, units=(worker,), resources=0))
        second = tactic.memory.unit_missions[worker.id]

        self.assertEqual(second.target, first.target)
        self.assertEqual(second.assigned_tick, first.assigned_tick)

    def test_every_worker_gets_a_scout_task_despite_the_precision_scan_budget(self) -> None:
        workers = (
            unit(1, UnitType.WORKER, (2, 0)),
            unit(2, UnitType.WORKER, (0, 2)),
            unit(3, UnitType.WORKER, (-2, 0)),
        )
        tactic = BalancedTactic()

        tactic.choose_actions(make_turn(tick=1, units=workers, resources=0))
        first_count = sum(
            mission.mission is UnitMission.EXPLORE
            for mission in tactic.memory.unit_missions.values()
        )
        tactic.choose_actions(make_turn(tick=2, units=workers, resources=0))
        second_count = sum(
            mission.mission is UnitMission.EXPLORE
            for mission in tactic.memory.unit_missions.values()
        )

        self.assertEqual(first_count, 3)
        self.assertEqual(second_count, 3)

    def test_home_alert_keeps_a_sticky_explorer_on_a_safe_local_step(self) -> None:
        worker = unit(1, UnitType.WORKER, (1, 0))
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(tick=1, units=(worker,), resources=0))
        tactic.memory.unit_missions[worker.id] = MissionState(
            UnitMission.EXPLORE,
            (8, 0),
            1,
        )
        distant_home_threat = unit(
            100,
            UnitType.RANGER,
            (20, 0),
            controlled=False,
        )
        turn = make_turn(
            tick=2,
            units=(unit(1, UnitType.WORKER, (2, 0)),),
            enemies=(distant_home_threat,),
            resources=0,
        )

        tactic.choose_actions(turn)

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker.id)
        )
        self.assertEqual(task["mission"], "EXPLORE")
        self.assertEqual(task["reason"], "INFORMATION_GAIN")
        self.assertNotEqual(task["mission"], "ESCAPE")

    def test_recent_fogged_threat_keeps_worker_in_escape_mission(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0))
        enemy = unit(100, UnitType.RANGER, (0, 3), controlled=False)
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(tick=1, units=(worker,), enemies=(enemy,), resources=0))

        second = make_turn(tick=2, units=(worker,), enemies=(), resources=0)
        tactic.choose_actions(second)

        task = next(item for item in tactic.last_decision_trace["tasks"] if item["actor_id"] == str(worker.id))
        self.assertEqual(task["mission"], "ESCAPE")

    def test_escape_preserves_fresh_threats_and_waits_without_survival_terminal(self) -> None:
        core = friendly_core(position=(40, -1))
        worker_id = uid(1)
        tactic = BalancedTactic()
        first = make_turn(
            tick=1,
            core=core,
            units=(unit(1, UnitType.WORKER, (0, 0), cargo=1),),
            enemies=(
                unit(100, UnitType.RANGER, (-1, 2), controlled=False),
                unit(101, UnitType.VANGUARD, (0, 2), controlled=False),
            ),
            obstacle_cells=((-1, -2), (0, -2), (0, 3), (1, -1), (1, 1), (2, 0)),
            resources=0,
        )
        tactic.choose_actions(first)
        self.assertEqual(first.plan.unit_actions[worker_id].direction, Direction.UP)

        second = make_turn(
            tick=2,
            core=core,
            units=(unit(1, UnitType.WORKER, (0, -1), cargo=1),),
            enemies=(
                unit(102, UnitType.VANGUARD, (-2, -1), controlled=False),
            ),
            obstacle_cells=((-3, -1), (0, -2), (1, -1), (1, 1)),
            resources=0,
        )
        tactic.choose_actions(second)

        state = tactic.memory.worker_escape_states[worker_id]
        self.assertEqual(set(state.threat_ids), {uid(100), uid(101), uid(102)})
        action = second.plan.unit_actions[worker_id]
        self.assertIsInstance(action, WaitAction)
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker_id)
        )
        self.assertEqual(task["reason"], "NO_SURVIVABLE_ROUTE")

    def test_deposited_worker_receives_scout_target_before_leaving_core(self) -> None:
        worker_id = uid(1)
        tactic = BalancedTactic()
        first = make_turn(
            tick=1,
            units=(unit(1, UnitType.WORKER, (0, 0), cargo=1),),
            resources=0,
        )
        tactic.choose_actions(first)
        self.assertIsInstance(first.plan.unit_actions[worker_id], DepositAction)
        deposited = ResolutionEvent(
            event_id=uid(90_100),
            tick=1,
            event_type="DEPOSIT_SUCCEEDED",
            actor_id=worker_id,
            position=(0, 0),
        )
        second = make_turn(
            tick=2,
            units=(unit(1, UnitType.WORKER, (0, 0)),),
            resources=1,
            events=(deposited,),
        )

        tactic.choose_actions(second)

        state = tactic.memory.worker_scout_states[worker_id]
        target = state.target
        self.assertIsNotNone(target)
        self.assertIsInstance(second.plan.unit_actions[worker_id], MoveAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(worker_id)
        )
        self.assertEqual(task["reason"], "SCOUT_CORE_EXIT")

        direction = second.plan.unit_actions[worker_id].direction
        position = direction.delta
        third = make_turn(
            tick=3,
            units=(unit(1, UnitType.WORKER, position),),
            resources=1,
        )
        tactic.choose_actions(third)
        self.assertEqual(tactic.memory.worker_scout_states[worker_id].target, target)
        self.assertEqual(
            tactic.memory.unit_missions[worker_id].mission,
            UnitMission.EXPLORE,
        )

    def test_remote_stale_cells_cannot_displace_reachable_frontier(self) -> None:
        worker = unit(1, UnitType.WORKER, (8, 0))
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(tick=1, units=(worker,), resources=0))
        tactic.memory.unit_missions.clear()
        tactic.memory.worker_scout_states.clear()
        for x in range(80, 90):
            for y in range(80, 90):
                tactic.memory.known_passable.add((x, y))
                tactic.memory.cell_last_visible[(x, y)] = 0

        turn = make_turn(tick=2, units=(worker,), resources=0)
        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[worker.id]
        state = tactic.memory.worker_scout_states[worker.id]
        self.assertIsInstance(action, MoveAction)
        self.assertIsNotNone(state.target)
        self.assertLess(manhattan(worker.position, state.target), 40)
        self.assertNotEqual(
            next(
                item["reason"]
                for item in tactic.last_decision_trace["tasks"]
                if item["actor_id"] == str(worker.id)
            ),
            "NO_REACHABLE_FRONTIER",
        )

    def test_far_persisted_scout_is_recalled_to_a_bounded_patrol_band(self) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (40, 0))
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            worker_scout_states={
                worker.id: WorkerScoutState(
                    worker_id=worker.id,
                    slot=0,
                    sector_index=0,
                    stage=19,
                    phase=WorkerScoutPhase.SECTOR_SCOUT,
                    target=(80, 0),
                    assigned_tick=1,
                )
            },
        )
        memory.known_passable.update(
            (x, y) for x in range(-5, 91) for y in range(-5, 6)
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(tick=2, core=core, units=(worker,), resources=0)

        tactic.choose_actions(turn)

        state = tactic.memory.worker_scout_states[worker.id]
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertIn(state.stage, range(3))
        self.assertIsNotNone(state.target)
        self.assertLessEqual(manhattan(core.position, state.target), 30)
        self.assertEqual(task["mission"], "RETURN_TO_SCOUT_BAND")
        action = turn.plan.unit_actions[worker.id]
        self.assertIsInstance(action, MoveAction)
        destination = add_direction(worker.position, action.direction)
        self.assertLess(
            manhattan(destination, core.position),
            manhattan(worker.position, core.position),
        )

    def test_failed_precision_scan_does_not_starve_other_workers(self) -> None:
        workers = tuple(
            unit(index, UnitType.WORKER, (index * 2, 0))
            for index in range(1, 5)
        )
        tactic = BalancedTactic(TacticConfig(path_node_limit=1))
        turn = make_turn(tick=1, units=workers, resources=0)

        tactic.choose_actions(turn)

        self.assertTrue(
            all(isinstance(turn.plan.unit_actions[worker.id], MoveAction) for worker in workers)
        )
        self.assertEqual(len(tactic.memory.worker_scout_states), len(workers))
        self.assertTrue(
            all(state.target is not None for state in tactic.memory.worker_scout_states.values())
        )

    def test_scout_slots_remain_unique_while_workers_are_busy(self) -> None:
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(
                tick=1,
                units=(
                    unit(1, UnitType.WORKER, (2, 0)),
                    unit(2, UnitType.WORKER, (4, 0), cargo=1),
                ),
                resources=0,
            )
        )
        self.assertIn(uid(1), tactic.memory.worker_scout_states)
        self.assertNotIn(uid(2), tactic.memory.worker_scout_states)
        tactic.choose_actions(
            make_turn(
                tick=2,
                units=(
                    unit(1, UnitType.WORKER, (2, 0)),
                    unit(2, UnitType.WORKER, (3, 0), cargo=1),
                    unit(3, UnitType.WORKER, (6, 0)),
                ),
                resources=0,
            )
        )
        tactic.choose_actions(
            make_turn(
                tick=3,
                units=(
                    unit(1, UnitType.WORKER, (2, 0)),
                    unit(2, UnitType.WORKER, (2, 1)),
                    unit(3, UnitType.WORKER, (6, 0)),
                ),
                resources=0,
            )
        )

        slots = [state.slot for state in tactic.memory.worker_scout_states.values()]
        self.assertEqual(len(slots), len(set(slots)))

    def test_active_empty_scouts_are_balanced_across_all_eight_sectors(self) -> None:
        workers = tuple(
            unit(index, UnitType.WORKER, ((index % 5) - 2, index // 5 + 2))
            for index in range(1, 21)
        )
        tactic = BalancedTactic()

        tactic.choose_actions(make_turn(tick=1, units=workers, resources=0))

        states = tuple(tactic.memory.worker_scout_states.values())
        sector_counts = Counter(state.sector_index for state in states)
        stage_counts = Counter(state.stage for state in states)
        self.assertEqual(len(states), 20)
        self.assertEqual(set(sector_counts), set(range(8)))
        self.assertLessEqual(max(sector_counts.values()) - min(sector_counts.values()), 1)
        self.assertLessEqual(max(stage_counts.values()) - min(stage_counts.values()), 1)
        self.assertTrue(all(state.scout_eligible for state in states))

    def test_twenty_scouts_form_a_low_overlap_obstacle_aware_ring(self) -> None:
        core = friendly_core(position=(0, 0))
        workers = tuple(
            unit(index, UnitType.WORKER, ((index % 5) - 2, index // 5 + 2))
            for index in range(1, 21)
        )
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            opening_complete=True,
        )
        memory.known_passable.update(
            (x, y) for x in range(-35, 36) for y in range(-35, 36)
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=1, core=core, units=workers, resources=0)
        )

        states = tuple(
            state
            for state in tactic.memory.worker_scout_states.values()
            if state.scout_eligible
        )
        metrics = tactic.last_decision_trace["economy"]["worker_activity_metrics"]
        self.assertTrue(
            all(
                state.target is not None
                and manhattan(core.position, state.target) <= 30
                for state in states
            )
        )
        self.assertGreaterEqual(metrics["scout_coverage_percent"], 25.0)
        self.assertLessEqual(metrics["scout_overlap_percent"], 15.0)
        self.assertLessEqual(metrics["scout_max_angular_gap_degrees"], 45.0)

    def test_cargo_worker_does_not_reserve_an_active_scout_sector(self) -> None:
        workers = tuple(
            unit(index, UnitType.WORKER, (index, 1), cargo=(1 if index == 9 else 0))
            for index in range(1, 10)
        )
        tactic = BalancedTactic()

        tactic.choose_actions(make_turn(tick=1, units=workers, resources=0))

        active = tuple(
            state
            for state in tactic.memory.worker_scout_states.values()
            if state.scout_eligible
        )
        self.assertEqual(len(active), 8)
        self.assertEqual({state.sector_index for state in active}, set(range(8)))
        self.assertNotIn(uid(9), tactic.memory.worker_scout_states)

    def test_return_to_band_uses_real_path_instead_of_reversing_at_a_wall(self) -> None:
        core = friendly_core(position=(0, 0))
        worker_id = uid(1)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            worker_scout_states={
                worker_id: WorkerScoutState(
                    worker_id=worker_id,
                    slot=0,
                    sector_index=0,
                    stage=0,
                    phase=WorkerScoutPhase.RETURN_TO_BAND,
                    target=(0, -20),
                    assigned_tick=1,
                )
            },
            opening_complete=True,
        )
        memory.known_passable.update(
            (x, y) for x in range(-4, 5) for y in range(-34, 2)
        )
        memory.known_obstacles.add((0, -31))
        tactic = BalancedTactic(memory=memory)
        first = make_turn(
            tick=2,
            core=core,
            units=(unit(1, UnitType.WORKER, (0, -32)),),
            obstacle_cells=((0, -31),),
            resources=0,
        )

        tactic.choose_actions(first)
        first_action = first.plan.unit_actions[worker_id]
        self.assertIsInstance(first_action, MoveAction)
        self.assertEqual(first_action.direction, Direction.LEFT)

        second = make_turn(
            tick=3,
            core=core,
            units=(unit(1, UnitType.WORKER, (-1, -32)),),
            obstacle_cells=((0, -31),),
            resources=0,
        )
        tactic.choose_actions(second)
        second_action = second.plan.unit_actions[worker_id]
        self.assertIsInstance(second_action, MoveAction)
        self.assertEqual(second_action.direction, Direction.DOWN)
        self.assertNotEqual(second_action.direction, Direction.RIGHT)

    def test_return_phase_stays_sticky_after_crossing_inside_radius_thirty(self) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (0, -29))
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            worker_scout_states={
                worker.id: WorkerScoutState(
                    worker_id=worker.id,
                    slot=0,
                    sector_index=0,
                    stage=0,
                    phase=WorkerScoutPhase.RETURN_TO_BAND,
                    target=(0, -20),
                    assigned_tick=1,
                )
            },
            opening_complete=True,
        )
        memory.known_passable.update(
            (x, y) for x in range(-3, 4) for y in range(-32, 2)
        )
        tactic = BalancedTactic(memory=memory)
        turn = make_turn(tick=2, core=core, units=(worker,), resources=0)

        tactic.choose_actions(turn)

        state = tactic.memory.worker_scout_states[worker.id]
        task = next(
            row
            for row in tactic.last_decision_trace["tasks"]
            if row["actor_id"] == str(worker.id)
        )
        self.assertEqual(state.phase, WorkerScoutPhase.RETURN_TO_BAND)
        self.assertEqual(state.target, (0, -20))
        self.assertEqual(task["mission"], UnitMission.RETURN_TO_SCOUT_BAND.value)

    def test_return_loop_backs_off_the_next_reverse_edge(self) -> None:
        core = friendly_core(position=(0, 0))
        worker = unit(1, UnitType.WORKER, (0, -32))
        target = (0, -20)
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            worker_scout_states={
                worker.id: WorkerScoutState(
                    worker_id=worker.id,
                    slot=0,
                    sector_index=0,
                    stage=0,
                    phase=WorkerScoutPhase.RETURN_TO_BAND,
                    target=target,
                    assigned_tick=1,
                )
            },
            scout_return_route_leases={
                worker.id: ScoutReturnRouteLease(
                    worker_id=worker.id,
                    target=target,
                    waypoint=target,
                    assigned_tick=1,
                    last_position=worker.position,
                    last_route_distance=12,
                )
            },
            opening_complete=True,
        )
        memory.position_history[worker.id] = (
            (-1, -32),
            (0, -32),
            (-1, -32),
        )
        memory.known_passable.update(
            (x, y) for x in range(-4, 5) for y in range(-34, 2)
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=5, core=core, units=(worker,), resources=0)
        )

        lease = tactic.memory.scout_return_route_leases[worker.id]
        self.assertEqual(lease.blocked_edge, ((0, -32), (-1, -32)))
        self.assertGreaterEqual(lease.backoff_until, 13)


if __name__ == "__main__":
    unittest.main()
