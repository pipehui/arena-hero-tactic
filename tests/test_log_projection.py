from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone

from arena_hero import (
    Accepted,
    CommandPlan,
    CommandSource,
    Received,
    UnitType,
)

from arena_tactic import BalancedTactic
from arena_tactic.log_projection import compact_logged_state, compact_strategy_trace
from replay_log import ReplayLogger
from tests.helpers import friendly_core, make_turn, uid, unit


class LogProjectionTests(unittest.TestCase):
    def test_escape_projection_keeps_safety_and_loop_evidence(self) -> None:
        worker = unit(1, UnitType.WORKER, (0, 0))
        enemy = unit(100, UnitType.VANGUARD, (2, 1), controlled=False)
        turn = make_turn(
            tick=10,
            core=friendly_core(position=(10, 10)),
            units=(worker,),
            enemies=(enemy,),
            resources=0,
        )
        tactic = BalancedTactic()
        tactic.choose_actions(turn)

        compact = compact_strategy_trace(tactic.last_decision_trace)

        assert compact is not None
        decision = next(
            row for row in compact["decisions"] if row["actor_id"] == str(worker.id)
        )
        metadata = decision["final"]["metadata"]
        self.assertGreater(metadata["survival_terminals"], 0)
        self.assertIn("visible_enemy_distance_before", metadata)
        self.assertIn("visible_enemy_distance_after", metadata)
        self.assertIn("nonfatal_budget_used", metadata)
        self.assertIn("escape_filter_rejections", metadata)

    def test_strategy_projection_is_detached_and_keeps_final_reason(self) -> None:
        trace = self._representative_trace(actor_count=4, cargo_count=2)
        original = copy.deepcopy(trace)

        compact = compact_strategy_trace(trace)

        self.assertEqual(trace, original)
        self.assertEqual(compact["schema_version"], 41)
        self.assertEqual(compact["source_trace_schema"], 36)
        self.assertNotIn("tasks", compact)
        decision = compact["decisions"][0]
        self.assertEqual(decision["final"]["reason"], "NO_VIABLE_MOVE")
        self.assertNotIn("final_reason", decision)
        self.assertEqual(
            decision["rejection_reason_counts"],
            {"CELL_CAPACITY": 1, "RESERVATION_CONFLICT": 1},
        )
        self.assertEqual(decision["key_rejections"][0]["mission"], "EXPLORE")
        self.assertIn("forward_exits", decision["key_rejections"][0]["metadata"])
        self.assertNotIn("debug_blob", decision["key_rejections"][0]["metadata"])

    def test_projection_keeps_one_canonical_service_job_per_actor(self) -> None:
        compact = compact_strategy_trace(
            self._representative_trace(actor_count=4, cargo_count=2)
        )
        queue = compact["economy"]["service_queue"]

        self.assertEqual(len(queue["jobs"]), 2)
        first = queue["jobs"][0]
        self.assertEqual(first["operations"], ["DEPOSIT"])
        self.assertEqual(first["route_mode"], "FULL")
        self.assertEqual(first["service_tick"], 110)
        self.assertNotIn("return_reservations", queue)
        self.assertNotIn("service_windows", queue)
        self.assertNotIn("timeline", queue)
        self.assertNotIn("worker_progress", queue)

    def test_routine_moves_are_compact_but_exceptional_events_remain_full(self) -> None:
        move = {
            "event_id": str(uid(8_001)),
            "tick": 5,
            "event_type": "UNIT_MOVE_SUCCEEDED",
            "reason_code": None,
            "actor_id": str(uid(1)),
            "target_id": None,
            "position": [2, 0],
            "values": None,
        }
        hit = {
            "event_id": str(uid(8_002)),
            "tick": 5,
            "event_type": "SHOT_HIT",
            "reason_code": None,
            "actor_id": str(uid(2)),
            "target_id": str(uid(3)),
            "position": [3, 0],
            "values": {"damage": 1},
        }
        state = {"status": "ACTIVE", "events": [move, hit], "objects": []}

        compact = compact_logged_state(state)

        self.assertEqual(compact["events"], [hit])
        self.assertEqual(
            compact["routine_moves"],
            [{"actor_id": str(uid(1)), "position": [2, 0]}],
        )
        self.assertEqual(state["events"], [move, hit])

    def test_agent_receipt_references_matching_turn_but_manual_keeps_plan(self) -> None:
        worker = unit(1, UnitType.WORKER, (1, 0))
        turn = make_turn(tick=9, units=(worker,))
        tactic = BalancedTactic()
        tactic.choose_actions(turn)
        accepted = Accepted(
            accepted=True,
            tick=9,
            source=CommandSource.AGENT,
            received_at=datetime.now(timezone.utc),
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = ReplayLogger(directory)
            logger.record_turn(
                turn,
                decision_ms=1.0,
                submission_ms=2.0,
                accepted=accepted,
                strategy=tactic.last_decision_trace,
            )
            logger.record_receipt(
                Received(
                    tick=9,
                    source=CommandSource.AGENT,
                    received_at=datetime.now(timezone.utc),
                    plan=turn.plan,
                )
            )
            logger.record_receipt(
                Received(
                    tick=9,
                    source=CommandSource.MANUAL,
                    received_at=datetime.now(timezone.utc),
                    plan=CommandPlan(tick=9),
                )
            )
            logger.close(status="completed", last_tick=9)
            records = [
                json.loads(line)
                for line in logger.path.read_text(encoding="utf-8").splitlines()
            ]

        receipts = [row for row in records if row["record_type"] == "canonical_receipt"]
        self.assertTrue(receipts[0]["matches_turn_plan"])
        self.assertEqual(receipts[0]["plan_ref"]["tick"], 9)
        self.assertNotIn("plan", receipts[0])
        self.assertFalse(receipts[1]["matches_turn_plan"])
        self.assertIn("plan", receipts[1])

    def test_unmatched_agent_receipt_keeps_its_complete_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = ReplayLogger(directory)
            logger.record_receipt(
                Received(
                    tick=19,
                    source=CommandSource.AGENT,
                    received_at=datetime.now(timezone.utc),
                    plan=CommandPlan(tick=19),
                )
            )
            logger.close(status="completed", last_tick=19)
            records = [
                json.loads(line)
                for line in logger.path.read_text(encoding="utf-8").splitlines()
            ]

        receipt = next(
            row for row in records if row["record_type"] == "canonical_receipt"
        )
        self.assertFalse(receipt["matches_turn_plan"])
        self.assertIn("plan", receipt)

    def test_representative_projection_reduces_raw_json_by_at_least_45_percent(self) -> None:
        trace = self._representative_trace(actor_count=53, cargo_count=21)

        before = len(json.dumps(trace, ensure_ascii=False, separators=(",", ":")))
        after = len(
            json.dumps(
                compact_strategy_trace(trace),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        self.assertLessEqual(after, before * 0.55)

    @staticmethod
    def _representative_trace(*, actor_count: int, cargo_count: int) -> dict:
        def intent(actor: int, *, reason: str) -> dict:
            return {
                "actor_id": f"actor-{actor:03}",
                "action": "WAIT",
                "mission": "EXPLORE",
                "priority": 70,
                "target": [actor, 1],
                "target_id": None,
                "expected_cell": None,
                "direction": None,
                "reason": reason,
                "risk": 0,
                "destination_exclusivity": "NONE",
                "metadata": {
                    "forward_exits": 2,
                    "continuation_reachable": False,
                    "debug_blob": "x" * 180,
                },
            }

        decisions = []
        rejected = []
        for actor in range(actor_count):
            final = intent(actor, reason="NO_VIABLE_MOVE")
            key_rejections = []
            for rejection_reason in ("CELL_CAPACITY", "RESERVATION_CONFLICT"):
                rejected_intent = intent(actor, reason="SCOUT_ROUTE")
                rejection = {
                    "intent": rejected_intent,
                    "rejection_reason": rejection_reason,
                    "blocking_actor_ids": [f"blocker-{actor:03}"],
                }
                key_rejections.append(rejection)
                rejected.append({"intent": rejected_intent, "reason": rejection_reason})
            decisions.append(
                {
                    "actor_id": f"actor-{actor:03}",
                    "actor_type": "WORKER" if actor < cargo_count else "RANGER",
                    "position": [actor, 0],
                    "final": final,
                    "final_reason": final["reason"],
                    "key_rejections": key_rejections,
                    "service": (
                        {"scheduled_deposit_tick": 110 + actor, "stalled_ticks": 0}
                        if actor < cargo_count
                        else None
                    ),
                }
            )
        jobs = []
        reservations = []
        for actor in range(cargo_count):
            actor_id = f"actor-{actor:03}"
            jobs.append(
                {
                    "actor_id": actor_id,
                    "operations": ["DEPOSIT"],
                    "phase": "APPROACHING",
                    "route_distance": 10 + actor,
                    "first_direction": "LEFT",
                    "first_position": [actor - 1, 0],
                    "gateway": [1, 0],
                    "earliest_service_tick": 109 + actor,
                    "service_tick": 110 + actor,
                    "exit_tick": 111 + actor,
                    "priority": 3,
                    "ready_since_tick": None,
                    "resource_cost": 0,
                    "resource_gain": 1,
                    "reason": "CARGO_RETURN",
                }
            )
            reservations.append(
                {
                    "worker_id": actor_id,
                    "route_target": [1, 0],
                    "route_distance": 10 + actor,
                    "first_direction": "LEFT",
                    "first_position": [actor - 1, 0],
                    "earliest_deposit_tick": 109 + actor,
                    "scheduled_deposit_tick": 110 + actor,
                    "departure_tick": 100,
                    "slack_ticks": 0,
                    "status": "RETURNING",
                    "delay_reason": None,
                    "route_mode": "FULL",
                    "waypoint": None,
                    "lane_version": 1,
                    "previous_scheduled_tick": 110 + actor,
                    "schedule_change_reason": None,
                    "schedule_drift": 0,
                }
            )
        friendlies = [
            {"actor_id": f"actor-{actor:03}", "position": [actor, 0]}
            for actor in range(actor_count)
        ]
        return {
            "schema_version": 36,
            "mode": "GLOBAL_MAP_SURVIVAL_ECONOMY",
            "tasks": [copy.deepcopy(row["final"]) for row in decisions],
            "decisions": decisions,
            "decision_summary": {"wait_reason_counts": {"NO_VIABLE_MOVE": actor_count}},
            "resolution": {
                "selected_count": actor_count,
                "rejected_count": len(rejected),
                "rejected": rejected[:32],
                "reserved_positions": [[actor, 0] for actor in range(actor_count)],
                "resource_spent": 0,
                "resource_gained": 0,
            },
            "world": {
                "tick": 100,
                "global_map": {
                    "terrain": {"obstacles": 10, "passable": 100},
                    "vision": {
                        "visible_cells": 200,
                        "sources": copy.deepcopy(friendlies),
                    },
                    "resources": [],
                    "enemies": [],
                    "threat": {},
                    "friendlies": friendlies,
                    "operations": {},
                },
            },
            "economy": {
                "worker_scouts": [
                    {
                        "worker_id": f"actor-{actor:03}",
                        "slot": actor,
                        "sector": actor % 8,
                        "stage": "SCOUT",
                        "mode": "SECTOR_SCOUT",
                        "target": [actor + 20, 0],
                        "assigned_tick": 90,
                        "best_route_cost": 20,
                        "stalled_ticks": 0,
                        "backoff_until": 0,
                        "reachable_candidates": 8,
                        "scan_budget": "STICKY_TARGET",
                        "action": "MOVE",
                        "reason": "EXPLORE",
                    }
                    for actor in range(min(actor_count, 26))
                ],
                "service_queue": {
                    "service": "DEPOSIT",
                    "admission_id": None,
                    "jobs": jobs,
                    "return_reservations": reservations,
                    "service_windows": copy.deepcopy(jobs),
                    "scheduled_deposits": [
                        {"worker_id": row["worker_id"], "tick": row["scheduled_deposit_tick"]}
                        for row in reservations
                    ],
                    "worker_progress": [
                        {"worker_id": row["worker_id"], "position": [0, 0], "stalled_ticks": 0}
                        for row in reservations
                    ],
                    "timeline": {"requests": copy.deepcopy(jobs)},
                    "lane_lease": {"version": 1, "entrance": [1, 0]},
                    "liveness_indicators": [],
                },
            },
            "combat": {
                "formation": {
                    "assignment": {
                        "bundles": [],
                        "rejected": [
                            {
                                "vanguard_id": f"v-{index}",
                                "ranger_id": f"r-{index}",
                                "anchor": [index, 0],
                                "support": [index, 2],
                                "reason": "RESERVATION_CONFLICT",
                            }
                            for index in range(24)
                        ],
                    },
                    "leases": [],
                    "move_feedback": [],
                    "waits": {"blocked_or_idle": 0},
                },
                "fire_missions": [],
            },
        }


if __name__ == "__main__":
    unittest.main()
