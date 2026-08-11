from __future__ import annotations

import unittest

from arena_hero import BeaconStatus, ChampionBeacon, MoveAction, UnitType, WaitAction

from arena_tactic import BalancedTactic, UnitMission
from tests.helpers import make_turn, unit


class BeaconTests(unittest.TestCase):
    def test_injured_worker_is_excluded_from_beacon_observation(self) -> None:
        workers = (
            unit(1, UnitType.WORKER, (2, 0), hp=1),
            unit(2, UnitType.WORKER, (0, 2)),
            unit(3, UnitType.WORKER, (-2, 0)),
            unit(4, UnitType.WORKER, (0, -2)),
        )
        beacon = ChampionBeacon(
            position=(3, 0),
            status=BeaconStatus.GROUND,
            carrier_id=None,
        )
        tactic = BalancedTactic()

        tactic.choose_actions(
            make_turn(units=workers, resources=1, beacon=beacon)
        )

        self.assertNotEqual(tactic.memory.beacon_mission_actor_id, workers[0].id)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(workers[0].id)
        )
        self.assertEqual(task["mission"], UnitMission.RECOVER.value)

    def test_friendly_beacon_carrier_returns_to_a_near_core_guard_cell(self) -> None:
        carrier = unit(1, UnitType.WORKER, (8, 0))
        beacon = ChampionBeacon(
            position=carrier.position,
            status=BeaconStatus.CARRIED,
            carrier_id=carrier.id,
        )
        turn = make_turn(units=(carrier,), resources=0, beacon=beacon)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        action = turn.plan.unit_actions[carrier.id]
        self.assertIsInstance(action, MoveAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(carrier.id)
        )
        self.assertEqual(task["mission"], UnitMission.BEACON.value)
        self.assertLess(abs(task["target"][0]) + abs(task["target"][1]), 8)

    def test_secured_beacon_carrier_does_not_resume_remote_exploration(self) -> None:
        carrier = unit(1, UnitType.WORKER, (2, 0))
        beacon = ChampionBeacon(
            position=carrier.position,
            status=BeaconStatus.CARRIED,
            carrier_id=carrier.id,
        )
        turn = make_turn(units=(carrier,), resources=0, beacon=beacon)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[carrier.id], WaitAction)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(carrier.id)
        )
        self.assertEqual(task["reason"], "BEACON_SECURED_NEAR_CORE")

    def test_secured_beacon_carrier_does_not_take_a_remote_resource_order(self) -> None:
        carrier = unit(1, UnitType.WORKER, (2, 0))
        beacon = ChampionBeacon(
            position=carrier.position,
            status=BeaconStatus.CARRIED,
            carrier_id=carrier.id,
        )
        turn = make_turn(
            units=(carrier,),
            resource_cells=((3, 0),),
            resources=0,
            beacon=beacon,
        )

        BalancedTactic().choose_actions(turn)

        self.assertIsInstance(turn.plan.unit_actions[carrier.id], WaitAction)

    def test_loaded_beacon_worker_keeps_cargo_delivery_priority(self) -> None:
        carrier = unit(1, UnitType.WORKER, (3, 0), cargo=1)
        beacon = ChampionBeacon(
            position=carrier.position,
            status=BeaconStatus.CARRIED,
            carrier_id=carrier.id,
        )
        turn = make_turn(units=(carrier,), resources=0, beacon=beacon)
        tactic = BalancedTactic()

        tactic.choose_actions(turn)

        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(carrier.id)
        )
        self.assertIn(task["mission"], {"RETURN_CARGO", "DEPOSIT"})

    def test_safe_visible_beacon_gets_a_persistent_worker_mission(self) -> None:
        workers = (
            unit(1, UnitType.WORKER, (1, 0)),
            unit(2, UnitType.WORKER, (0, 1)),
            unit(3, UnitType.WORKER, (-1, 0)),
            unit(4, UnitType.WORKER, (0, -1)),
        )
        beacon = ChampionBeacon(
            position=(3, 0),
            status=BeaconStatus.GROUND,
            carrier_id=None,
        )
        tactic = BalancedTactic()
        first = make_turn(units=workers, resources=0, beacon=beacon)

        tactic.choose_actions(first)

        self.assertIsInstance(first.plan.unit_actions[workers[0].id], MoveAction)
        self.assertEqual(tactic.memory.beacon_mission_actor_id, workers[0].id)
        task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(workers[0].id)
        )
        self.assertEqual(task["mission"], UnitMission.BEACON.value)

        moved_workers = (
            unit(1, UnitType.WORKER, (2, 0)),
            *workers[1:],
        )
        hidden = ChampionBeacon(position=(3, 0), status=None, carrier_id=None)
        second = make_turn(tick=2, units=moved_workers, resources=0, beacon=hidden)

        tactic.choose_actions(second)

        self.assertIsInstance(second.plan.unit_actions[moved_workers[0].id], MoveAction)
        second_task = next(
            item
            for item in tactic.last_decision_trace["tasks"]
            if item["actor_id"] == str(moved_workers[0].id)
        )
        self.assertEqual(second_task["reason"], "CONTINUE_BEACON_APPROACH")

    def test_beacon_worker_aborts_when_it_becomes_loaded(self) -> None:
        workers = tuple(
            unit(index, UnitType.WORKER, (index - 1, 0))
            for index in range(1, 5)
        )
        beacon = ChampionBeacon(
            position=(5, 0),
            status=BeaconStatus.GROUND,
            carrier_id=None,
        )
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(units=workers, resources=0, beacon=beacon))
        actor_id = tactic.memory.beacon_mission_actor_id
        self.assertIsNotNone(actor_id)
        loaded = tuple(
            unit(
                worker.id.int,
                UnitType.WORKER,
                worker.position,
                cargo=1 if worker.id == actor_id else 0,
            )
            for worker in workers
        )
        hidden = ChampionBeacon(position=(5, 0), status=None, carrier_id=None)

        tactic.choose_actions(
            make_turn(tick=2, units=loaded, resources=0, beacon=hidden)
        )

        self.assertIsNone(tactic.memory.beacon_mission_actor_id)


if __name__ == "__main__":
    unittest.main()
