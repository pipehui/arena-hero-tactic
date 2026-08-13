from __future__ import annotations

import dataclasses
import json
import unittest
from collections.abc import Mapping

from arena_hero import Core, Turn, Unit, UnitType, WaitAction

from arena_tactic import (
    ActionIntent,
    BalancedTactic,
    CargoReturnReservation,
    CargoRouteProgress,
    CoreServiceWindow,
    CoreServiceJob,
    CoreServicePhase,
    CoreSlotSchedule,
    CoreServiceQueue,
    CoreOperationRequest,
    CoreOperationTimeline,
    CrisisForceBaseline,
    EnemyRangerFireEstimate,
    FireMission,
    HomeCombatAssignment,
    HostileApproachEstimate,
    HomeCounterSiegeDecision,
    IntentResolution,
    PatientAdmissionProgress,
    PatientQueueEntry,
    PendingSubmission,
    RangerStanceOption,
    ScreeningContactDecision,
    ServiceCellLease,
    ServiceLaneLease,
    SubmissionResult,
    SegmentedReturnLease,
    SquadRendezvousLease,
    TacticalMap,
    ThreatHeatCell,
    UnitMission,
    VanguardIntentEstimate,
    VanguardAssignmentCandidate,
    VanguardInterceptTask,
    WorkerTaskProgress,
    WorldModel,
)
from tests.helpers import make_turn, unit


class PublicContractTests(unittest.TestCase):
    def test_new_kernel_types_are_frozen_values(self) -> None:
        for value_type in (
            WorldModel,
            ActionIntent,
            CargoReturnReservation,
            CargoRouteProgress,
            CoreServiceWindow,
            CoreServiceJob,
            CoreSlotSchedule,
            IntentResolution,
            CoreServiceQueue,
            CoreOperationRequest,
            CoreOperationTimeline,
            FireMission,
            ThreatHeatCell,
            TacticalMap,
            VanguardIntentEstimate,
            EnemyRangerFireEstimate,
            CrisisForceBaseline,
            HomeCombatAssignment,
            HostileApproachEstimate,
            HomeCounterSiegeDecision,
            PatientAdmissionProgress,
            PatientQueueEntry,
            PendingSubmission,
            RangerStanceOption,
            ScreeningContactDecision,
            ServiceCellLease,
            ServiceLaneLease,
            SubmissionResult,
            SegmentedReturnLease,
            SquadRendezvousLease,
            VanguardAssignmentCandidate,
            VanguardInterceptTask,
            WorkerTaskProgress,
        ):
            self.assertTrue(dataclasses.is_dataclass(value_type))
            self.assertTrue(value_type.__dataclass_params__.frozen)

    def test_respawning_turn_does_not_reuse_old_controllers(self) -> None:
        tactic = BalancedTactic()
        live = make_turn(units=(unit(1, UnitType.WORKER, (0, 0)),))
        tactic.choose_actions(live)
        respawning = make_turn(
            tick=2,
            core=None,
            units=(),
            respawning=True,
        )

        tactic.choose_actions(respawning)

        self.assertEqual(respawning.plan.unit_actions, {})
        self.assertIsNone(respawning.plan.core_action)

    def test_unit_without_a_core_gets_only_a_safe_wait(self) -> None:
        worker = unit(1, UnitType.WORKER, (4, 4))
        turn = make_turn(core=None, units=(worker,), respawning=True)

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[worker.id], WaitAction)
        self.assertIsNone(turn.plan.core_action)

    def test_complete_plan_assigns_one_explicit_action_per_unit(self) -> None:
        workers = tuple(
            unit(index, UnitType.WORKER, (index, 0)) for index in range(1, 4)
        )
        turn = make_turn(units=workers, resources=0)

        BalancedTactic().choose_actions(turn)

        self.assertEqual(set(turn.plan.unit_actions), {worker.id for worker in workers})
        self.assertTrue(all(action is not None for action in turn.plan.unit_actions.values()))

    def test_no_task_becomes_explicit_wait(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0), cargo=0)
        turn = make_turn(units=(worker,), obstacle_cells=((1, 0), (-1, 0), (0, 1), (0, -1)))

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[worker.id], WaitAction)

    def test_decision_trace_is_detached_schema_33_data(self) -> None:
        turn = make_turn(units=(unit(1, UnitType.WORKER, (1, 0)),))
        tactic = BalancedTactic()
        tactic.choose_actions(turn)

        trace = tactic.last_decision_trace

        self.assertEqual(trace["schema_version"], 41)
        self.assertIn("resolution", trace)
        self.assertIn("world", trace)
        global_map = trace["world"]["global_map"]
        self.assertIn("vision", global_map)
        self.assertIn("resources", global_map)
        self.assertIn("enemies", global_map)
        self.assertIn("threat", global_map)
        self.assertIn("friendlies", global_map)
        self.assertIn("operations", global_map)
        json.dumps(trace)
        self.assertIsInstance(next(iter(trace["tasks"])), dict)

    def test_failed_planning_never_reuses_the_previous_tick_trace(self) -> None:
        tactic = BalancedTactic()
        tactic.choose_actions(
            make_turn(tick=1, units=(unit(1, UnitType.WORKER, (1, 0)),))
        )

        class BrokenKernel:
            @staticmethod
            def decide(turn):
                raise RuntimeError("planned failure")

        tactic._kernel = BrokenKernel()
        with self.assertRaisesRegex(RuntimeError, "planned failure"):
            tactic.choose_actions(
                make_turn(tick=2, units=(unit(1, UnitType.WORKER, (1, 0)),))
            )

        trace = tactic.last_decision_trace
        self.assertEqual(trace["mode"], "DECIDING")
        self.assertEqual(trace["tick"], 2)
        self.assertEqual(trace["tasks"], [])

    def test_persistent_memory_contains_no_sdk_controllers(self) -> None:
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(units=(unit(1, UnitType.WORKER, (1, 0)),)))

        def walk(value):
            if dataclasses.is_dataclass(value):
                for item in dataclasses.fields(value):
                    yield from walk(getattr(value, item.name))
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    yield from walk(key)
                    yield from walk(item)
            elif isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    yield from walk(item)
            else:
                yield value

        self.assertFalse(
            any(isinstance(value, (Turn, Unit, Core)) for value in walk(tactic.memory))
        )
        self.assertIsNotNone(tactic.last_tactical_map)
        self.assertFalse(
            any(
                isinstance(value, (Turn, Unit, Core))
                for value in walk(tactic.last_tactical_map)
            )
        )

    def test_unit_mission_is_public(self) -> None:
        self.assertEqual(UnitMission.DEPOSIT.value, "DEPOSIT")
        self.assertEqual(UnitMission.CLEAR_CORE.value, "CLEAR_CORE")


if __name__ == "__main__":
    unittest.main()
