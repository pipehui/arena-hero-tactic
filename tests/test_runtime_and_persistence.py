from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from datetime import datetime, timezone

from arena_hero import (
    APIError,
    Accepted,
    CommandPlan,
    CommandSource,
    CoreState,
    Direction,
    MoveAction,
    Received,
    ResolutionEvent,
    TransportError,
    UnitType,
)

import balanced_tactic
from arena_tactic import (
    BalancedTactic,
    CrisisForceBaseline,
    SubmissionCoordinator,
    SubmissionOutcome,
    TacticMemory,
    ThreatHeatCell,
    WorkerScoutPhase,
    WorkerScoutState,
)
from arena_tactic.persistence import (
    EXPLORATION_MEMORY_SCHEMA_VERSION,
    ExplorationMemoryStore,
)
from arena_tactic.models import (
    EnemyCoreControlZone,
    EnemyCoreIntel,
    EnemyTrack,
    LongRangeRaidCampaign,
    RaidAttemptMemory,
    RaidDistanceBand,
    SiegeApproachPlan,
)
from arena_tactic.runtime import InstanceAlreadyRunning, SingleInstanceLock
from replay_log import LOG_SCHEMA_VERSION, ReplayLogger
from tests.helpers import friendly_core, make_turn, uid, unit


class _EventGame:
    def __init__(self, events):
        self.events_to_send = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def events(self):
        yield from self.events_to_send

    def close(self):
        return None


class _SubmissionClient:
    def __init__(self, submitter):
        self._submitter = submitter
        self.closed = False

    def submit(self, plan, *, idempotency_key=None):
        return self._submitter(plan, idempotency_key)

    def close(self):
        self.closed = True


class RuntimeAndPersistenceTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_long_range_raid_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = TacticMemory(core_id=uid(10_000), core_position=(0, 0))
            memory.raid_long_range_campaign = LongRangeRaidCampaign(
                target_id=uid(900),
                member_ids=(uid(1), uid(2), uid(3), uid(4)),
                phase="ADVANCING",
                started_tick=100,
                route_eta=50,
                search_deadline_tick=180,
                last_position=(50, 0),
                last_group_distance=160,
                no_progress_ticks=1,
            )
            memory.raid_target_id = uid(900)
            memory.raid_member_ids = (uid(1), uid(2), uid(3), uid(4))
            memory.raid_phase = "ADVANCING"
            memory.raid_distance_band = RaidDistanceBand.LONG_RANGE
            memory.raid_siege_approach = SiegeApproachPlan(
                target_id=uid(900),
                target_position=(50, 0),
                distance_band=RaidDistanceBand.LONG_RANGE,
                vanguard_positions=((49, 0),),
                ranger_positions=((47, 0),),
                route_eta=49,
            )
            memory.enemy_core_intel[uid(900)] = EnemyCoreIntel(
                id=uid(900),
                position=(50, 0),
                hp=5,
                shield=5,
                state=CoreState.NORMAL,
                destination=None,
                last_seen_tick=100,
                sighting_count=2,
            )
            store = ExplorationMemoryStore(directory, save_interval_ticks=1)
            self.assertTrue(store.save(memory, tick=110, force=True))

            restored = ExplorationMemoryStore(directory).load()

            self.assertEqual(restored.raid_long_range_campaign, memory.raid_long_range_campaign)
            self.assertEqual(restored.raid_target_id, uid(900))
            self.assertEqual(restored.raid_phase, "ADVANCING")
            self.assertEqual(restored.raid_distance_band, RaidDistanceBand.LONG_RANGE)
            self.assertEqual(restored.raid_siege_approach, memory.raid_siege_approach)
            self.assertIn(uid(900), restored.enemy_core_intel)

    def test_schema_versions_are_upgraded(self) -> None:
        self.assertEqual(LOG_SCHEMA_VERSION, 47)
        self.assertEqual(EXPLORATION_MEMORY_SCHEMA_VERSION, 17)

    def test_schema_15_preserves_core_control_attempts_and_returning_raid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_id = uid(900)
            member_id = uid(1)
            memory = TacticMemory(core_id=uid(10_000), core_position=(0, 0))
            memory.enemy_core_intel[target_id] = EnemyCoreIntel(
                id=target_id,
                position=(33, 0),
                hp=5,
                shield=5,
                state=CoreState.NORMAL,
                destination=None,
                last_seen_tick=100,
                sighting_count=3,
            )
            memory.enemy_core_control_zones[target_id] = EnemyCoreControlZone(
                core_id=target_id,
                center=(33, 0),
                exclusion_radius=6,
                clear_radius=8,
                last_seen_tick=100,
                visible_now=False,
                expires_tick=612,
            )
            memory.raid_attempts[target_id] = RaidAttemptMemory(
                core_id=target_id,
                failed_attempts=1,
                last_failure_tick=105,
                last_failure_reason="MEMBER_LOW_HP",
                last_failure_sighting_tick=100,
            )
            memory.raid_member_ids = (member_id,)
            memory.raid_phase = "RETURNING"
            memory.raid_return_reason = "TARGET_DESTROYED"
            memory.raid_handoff_targets[member_id] = (12, 0)
            store = ExplorationMemoryStore(directory, save_interval_ticks=1)

            self.assertTrue(store.save(memory, tick=110, force=True))
            restored = ExplorationMemoryStore(directory).load()

            self.assertIn(target_id, restored.enemy_core_control_zones)
            self.assertEqual(restored.raid_attempts[target_id].failed_attempts, 1)
            self.assertEqual(restored.raid_phase, "RETURNING")
            self.assertEqual(restored.raid_target_id, None)
            self.assertEqual(restored.raid_member_ids, (member_id,))
            self.assertEqual(restored.raid_handoff_targets[member_id], (12, 0))

    def test_main_translates_sigterm_into_a_graceful_service_stop(self) -> None:
        previous_handler = object()
        installed: list[tuple[object, object]] = []

        def record_signal(signal_number, handler):
            installed.append((signal_number, handler))

        with (
            patch.dict(
                balanced_tactic.os.environ,
                {"ARENA_HERO_API_KEY": "test-key"},
                clear=False,
            ),
            patch.object(
                balanced_tactic.signal,
                "getsignal",
                return_value=previous_handler,
            ),
            patch.object(
                balanced_tactic.signal,
                "signal",
                side_effect=record_signal,
            ),
            patch.object(balanced_tactic, "play") as play,
        ):
            balanced_tactic.main()

        self.assertGreaterEqual(len(installed), 2)
        self.assertTrue(callable(installed[0][1]))
        self.assertIs(installed[-1][1], previous_handler)
        play.assert_called_once()

    def test_checkpoint_round_trip_preserves_world_values(self) -> None:
        memory = TacticMemory()
        memory.known_obstacles.add((4, 5))
        memory.cell_last_visible[(1, 2)] = 99
        memory.resource_seen_count[(3, 3)] = 7
        memory.home_force_high_water = 14
        memory.crisis_force_baseline = CrisisForceBaseline(
            vanguards=7,
            rangers=6,
            started_tick=97,
            phase="REBUILD",
            safe_ticks=4,
        )
        memory.core_position = (2, 2)
        memory.core_position_history = ((1, 2), (2, 2))
        memory.last_congestion_decay_tick = 90
        memory.service_entrance = (1, 0)
        memory.service_queue_cells = ((1, 0), (2, 0))
        memory.service_exit_cell = (-1, 0)
        memory.cargo_arrival_ticks[uid(1)] = 88
        memory.threat_heat[(6, 7)] = ThreatHeatCell(
            position=(6, 7),
            risk=16,
            updated_tick=90,
            expires_tick=130,
            source="UNIT_DAMAGED",
        )
        memory.enemy_tracks[uid(90)] = EnemyTrack(
            id=uid(90),
            unit_type=UnitType.RANGER,
            samples=((99, (8, 8)), (100, (9, 8))),
            last_seen_tick=100,
        )
        memory.enemy_core_intel[uid(91)] = EnemyCoreIntel(
            id=uid(91),
            position=(12, 8),
            hp=5,
            shield=3,
            state=CoreState.NORMAL,
            destination=None,
            last_seen_tick=100,
            sighting_count=2,
        )
        memory.worker_scout_states[uid(2)] = WorkerScoutState(
            worker_id=uid(2),
            slot=3,
            sector_index=3,
            stage=2,
            phase=WorkerScoutPhase.SECTOR_SCOUT,
            target=(20, 10),
            assigned_tick=91,
            best_route_cost=17,
            stalled_ticks=1,
            last_scan_tick=90,
            reachable_candidates=6,
            scout_eligible=True,
            coverage_version=1,
            lease_until=155,
            visible_gain=21,
            overlap_cells=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ExplorationMemoryStore(directory)
            store.save(memory, tick=100)
            restored = store.load()

        self.assertIn((4, 5), restored.known_obstacles)
        self.assertEqual(restored.cell_last_visible[(1, 2)], 99)
        self.assertEqual(restored.resource_seen_count[(3, 3)], 7)
        self.assertEqual(restored.home_force_high_water, 14)
        self.assertEqual(restored.crisis_force_baseline, memory.crisis_force_baseline)
        self.assertEqual(restored.core_position_history, ((1, 2), (2, 2)))
        self.assertEqual(restored.last_congestion_decay_tick, 90)
        self.assertIsNone(restored.service_entrance)
        self.assertEqual(restored.service_queue_cells, ())
        self.assertIsNone(restored.service_exit_cell)
        self.assertEqual(restored.cargo_arrival_ticks, {})
        self.assertEqual(restored.threat_heat[(6, 7)].risk, 16)
        self.assertEqual(restored.threat_heat[(6, 7)].expires_tick, 130)
        self.assertEqual(restored.worker_scout_states[uid(2)].target, (20, 10))
        self.assertEqual(restored.worker_scout_states[uid(2)].slot, 3)
        self.assertTrue(restored.worker_scout_states[uid(2)].scout_eligible)
        self.assertEqual(restored.worker_scout_states[uid(2)].coverage_version, 1)
        self.assertEqual(restored.worker_scout_states[uid(2)].lease_until, 155)
        self.assertEqual(restored.worker_scout_states[uid(2)].visible_gain, 21)
        self.assertEqual(restored.worker_scout_states[uid(2)].overlap_cells, 2)
        self.assertEqual(restored.enemy_tracks[uid(90)].position, (9, 8))
        self.assertEqual(restored.enemy_tracks[uid(90)].last_seen_tick, 100)
        self.assertEqual(restored.enemy_core_intel[uid(91)].position, (12, 8))
        self.assertEqual(restored.enemy_core_intel[uid(91)].sighting_count, 2)

    def test_schema_8_checkpoint_migrates_with_empty_threat_heat(self) -> None:
        old = {
            "schema_version": 8,
            "saved_tick": 20,
            "known_obstacles": [[1, 1]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "balanced_tactic_memory.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            restored = ExplorationMemoryStore(directory).load()

        self.assertEqual(restored.threat_heat, {})

    def test_schema_14_remote_scout_is_migrated_into_bounded_band(self) -> None:
        worker_id = uid(2)
        old = {
            "schema_version": 14,
            "saved_tick": 100,
            "core_position": [0, 0],
            "worker_scout_states": [
                {
                    "worker_id": str(worker_id),
                    "slot": 5,
                    "sector_index": 5,
                    "stage": 23,
                    "phase": "SECTOR_SCOUT",
                    "target": [230, 0],
                    "assigned_tick": 90,
                    "stalled_ticks": 0,
                    "backoff_until": 0,
                    "reachable_candidates": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "balanced_tactic_memory.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            restored = ExplorationMemoryStore(directory).load()

        state = restored.worker_scout_states[worker_id]
        self.assertEqual(state.stage, 2)
        self.assertEqual(state.sector_index, 1)
        self.assertIsNone(state.target)
        self.assertEqual(state.phase, WorkerScoutPhase.SECTOR_SCOUT)

    def test_schema_16_scout_slot_is_marked_for_active_pool_remapping(self) -> None:
        worker_id = uid(2)
        old = {
            "schema_version": 16,
            "saved_tick": 100,
            "core_position": [0, 0],
            "worker_scout_states": [
                {
                    "worker_id": str(worker_id),
                    "slot": 5,
                    "sector_index": 1,
                    "stage": 2,
                    "phase": "SECTOR_SCOUT",
                    "target": [20, 0],
                    "assigned_tick": 90,
                    "stalled_ticks": 0,
                    "backoff_until": 0,
                    "reachable_candidates": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "balanced_tactic_memory.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            restored = ExplorationMemoryStore(directory).load()

        state = restored.worker_scout_states[worker_id]
        self.assertEqual(state.coverage_version, 0)
        self.assertEqual(state.lease_until, 0)

    def test_old_checkpoint_is_read_with_safe_defaults(self) -> None:
        old = {
            "schema_version": 3,
            "saved_tick": 8,
            "obstacles": [[5, 6]],
            "resource_memory": [[7, 8, 4]],
            "patrol_visits": [[1, 2, 3]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "balanced_tactic_memory.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            restored = ExplorationMemoryStore(directory).load()

        self.assertIn((5, 6), restored.known_obstacles)
        self.assertEqual(restored.resource_memory[(7, 8)], 4)

    def test_checkpoint_writes_are_batched_and_force_flushable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExplorationMemoryStore(directory, save_interval_ticks=16)
            memory = TacticMemory()
            self.assertTrue(store.save(memory, tick=10))
            first = json.loads(store.path.read_text(encoding="utf-8"))
            memory.known_obstacles.add((9, 9))
            self.assertTrue(store.save(memory, tick=11))
            batched = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(batched["saved_tick"], first["saved_tick"])

            self.assertTrue(store.save(memory, tick=11, force=True))
            flushed = json.loads(store.path.read_text(encoding="utf-8"))

        self.assertEqual(flushed["saved_tick"], 11)
        self.assertIn([9, 9], flushed["known_obstacles"])

    def test_log_catch_up_deduplicates_and_orders_overlapping_turns(self) -> None:
        core_id = str(uid(10_000))

        def record(tick: int, position: tuple[int, int]) -> dict:
            return {
                "record_type": "turn",
                "tick": tick,
                "state": {
                    "objects": [
                        {
                            "kind": "CORE",
                            "id": core_id,
                            "controlled": True,
                            "position": list(position),
                        }
                    ]
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Deliberately put the newer Tick in the lexically older file and
            # duplicate it in the second file, as can happen after reconnect.
            (root / "arena_hero_1.jsonl").write_text(
                "\n".join(json.dumps(item) for item in (record(12, (2, 0)), record(11, (1, 0)))),
                encoding="utf-8",
            )
            (root / "arena_hero_2.jsonl").write_text(
                json.dumps(record(12, (2, 0))),
                encoding="utf-8",
            )
            restored = ExplorationMemoryStore(directory).load()

        self.assertEqual(restored.last_tick, 12)
        self.assertEqual(restored.core_position, (2, 0))
        self.assertEqual(restored.visit_counts[(2, 0)], 1)

    def test_log_catch_up_resets_core_relative_state_after_respawn(self) -> None:
        old = TacticMemory(
            core_id=uid(10_000),
            core_position=(0, 0),
            opening_complete=True,
            home_force_high_water=19,
            strategic_relocation_pending=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ExplorationMemoryStore(directory)
            store.save(old, tick=10, force=True)
            record = {
                "record_type": "turn",
                "tick": 11,
                "state": {
                    "objects": [
                        {
                            "kind": "CORE",
                            "id": str(uid(20_000)),
                            "controlled": True,
                            "position": [50, 50],
                        }
                    ]
                },
            }
            (Path(directory) / "arena_hero_respawn.jsonl").write_text(
                json.dumps(record),
                encoding="utf-8",
            )
            restored = store.load()

        self.assertEqual(restored.core_id, uid(20_000))
        self.assertEqual(restored.core_position, (50, 50))
        self.assertFalse(restored.opening_complete)
        self.assertEqual(restored.home_force_high_water, 12)
        self.assertFalse(restored.strategic_relocation_pending)

    def test_log_catch_up_rebuilds_threat_heat_after_last_checkpoint(self) -> None:
        core_id = str(uid(10_000))
        with tempfile.TemporaryDirectory() as directory:
            store = ExplorationMemoryStore(directory)
            store.save(
                TacticMemory(core_id=uid(10_000), core_position=(0, 0)),
                tick=10,
                force=True,
            )
            record = {
                "record_type": "turn",
                "tick": 11,
                "state": {
                    "objects": [
                        {
                            "kind": "CORE",
                            "id": core_id,
                            "controlled": True,
                            "position": [0, 0],
                        },
                        {
                            "kind": "UNIT",
                            "id": str(uid(99)),
                            "controlled": False,
                            "position": [3, 0],
                            "unit_type": "RANGER",
                        },
                        {
                            "kind": "CORE",
                            "id": str(uid(100)),
                            "controlled": False,
                            "position": [6, 0],
                            "hp": 5,
                            "shield": 4,
                            "state": "NORMAL",
                            "destination": None,
                        },
                    ],
                    "events": [
                        {
                            "event_type": "UNIT_DESTROYED",
                            "position": [5, 0],
                        }
                    ],
                },
            }
            (Path(directory) / "arena_hero_after_checkpoint.jsonl").write_text(
                json.dumps(record),
                encoding="utf-8",
            )
            restored = ExplorationMemoryStore(directory).load()

        self.assertEqual(restored.threat_heat[(5, 0)].score(11), 24)
        self.assertGreater(restored.threat_heat[(3, 0)].score(11), 0)
        self.assertEqual(restored.enemy_tracks[uid(99)].position, (3, 0))
        self.assertEqual(restored.enemy_tracks[uid(99)].last_seen_tick, 11)
        self.assertEqual(restored.enemy_core_intel[uid(100)].position, (6, 0))

    def test_schema_6_remote_admission_is_invalidated_after_restart(self) -> None:
        old = {
            "schema_version": 6,
            "saved_tick": 10,
            "core_id": str(uid(10_000)),
            "core_position": [0, 0],
            "service_admission_id": str(uid(1)),
            "service_kind": "DEPOSIT",
            "service_entrance": [0, 1],
            "service_queue_cells": [[0, 1], [0, 2]],
            "service_exit_cell": [0, -1],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "balanced_tactic_memory.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            restored = ExplorationMemoryStore(directory).load()

        far = unit(1, UnitType.WORKER, (8, 0), cargo=1)
        ready = unit(2, UnitType.WORKER, (0, 2), cargo=1)
        tactic = BalancedTactic(memory=restored)
        turn = make_turn(
            tick=11,
            core=friendly_core(position=(0, 0)),
            units=(far, ready),
            resources=0,
        )
        tactic.choose_actions(turn)

        queue = tactic.last_decision_trace["economy"]["service_queue"]
        self.assertIsNone(queue["admission_id"])
        self.assertIsInstance(turn.plan.unit_actions[ready.id], MoveAction)

    def test_replay_logger_writes_schema_41_and_redacts_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = ReplayLogger(directory)
            logger.record_error(
                stage="connecting",
                tick=None,
                error=RuntimeError("Bearer hidden-token"),
                secret="hidden-token",
            )
            logger.close(status="error", last_tick=None)
            text = logger.path.read_text(encoding="utf-8")
            first = json.loads(text.splitlines()[0])

        self.assertEqual(first["schema_version"], 47)
        self.assertNotIn("hidden-token", text)

    def test_turn_log_contains_detached_schema_41_strategy(self) -> None:
        turn = make_turn(tick=9, units=(unit(1, UnitType.WORKER, (1, 0)),))
        tactic = BalancedTactic()
        tactic.choose_actions(turn)
        receipt = Accepted(
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
                accepted=receipt,
                strategy=tactic.last_decision_trace,
            )
            logger.close(status="completed", last_tick=9)
            records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(records[0]["schema_version"], 47)
        record = next(item for item in records if item["record_type"] == "turn")
        self.assertEqual(record["strategy"]["schema_version"], 47)
        self.assertEqual(record["strategy"]["source_trace_schema"], 45)
        self.assertNotIn("tasks", record["strategy"])
        self.assertIn("resolution", record["strategy"])
        decisions = record["strategy"]["decisions"]
        self.assertTrue(decisions)
        self.assertTrue(all(row["final"]["reason"] for row in decisions))
        self.assertTrue(all("final_reason" not in row for row in decisions))
        self.assertTrue(all(len(row["key_rejections"]) <= 3 for row in decisions))
        self.assertIn("wait_reason_counts", record["strategy"]["decision_summary"])

    def test_capacity_trace_keeps_wartime_policy_through_alert_lease(self) -> None:
        core = friendly_core(position=(0, 0))
        memory = TacticMemory(
            core_id=core.id,
            core_position=core.position,
            home_defense_alert_until=20,
        )
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(make_turn(tick=18, core=core, resources=0))
        active = tactic.last_decision_trace["capacity_policy"]
        self.assertTrue(active["home_defense_active"])
        self.assertTrue(active["wartime_worker_exclusive"])

        tactic.choose_actions(make_turn(tick=21, core=core, resources=0))
        safe = tactic.last_decision_trace["capacity_policy"]
        self.assertFalse(safe["home_defense_active"])

    def test_visible_home_warning_starts_four_tick_capacity_lease(self) -> None:
        core = friendly_core(position=(0, 0))
        enemy = unit(
            900,
            UnitType.RANGER,
            (0, -17),
            controlled=False,
        )
        memory = TacticMemory(core_id=core.id, core_position=core.position)
        tactic = BalancedTactic(memory=memory)

        tactic.choose_actions(
            make_turn(tick=10, core=core, enemies=(enemy,), resources=0)
        )
        self.assertTrue(
            tactic.last_decision_trace["capacity_policy"]["home_defense_active"]
        )
        self.assertEqual(memory.home_defense_alert_until, 14)

        tactic.choose_actions(make_turn(tick=14, core=core, resources=0))
        self.assertTrue(
            tactic.last_decision_trace["capacity_policy"]["home_defense_active"]
        )
        tactic.choose_actions(make_turn(tick=15, core=core, resources=0))
        self.assertFalse(
            tactic.last_decision_trace["capacity_policy"]["home_defense_active"]
        )

    def test_single_instance_lock_rejects_overlap_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tactic.lock"
            first = SingleInstanceLock(path)
            second = SingleInstanceLock(path)
            with first:
                with self.assertRaises(InstanceAlreadyRunning):
                    second.acquire()
            second.acquire()
            second.release()

    def test_trace_accumulates_session_outcomes(self) -> None:
        tactic = BalancedTactic()
        actor = unit(1, UnitType.RANGER, (1, 0))
        event = ResolutionEvent(
            event_id=uid(99_000),
            tick=1,
            event_type="SHOT_HIT",
            actor_id=actor.id,
            position=(2, 0),
            values={},
        )
        tactic.choose_actions(make_turn(tick=1, units=(actor,), events=(event,)))

        outcomes = tactic.last_decision_trace["outcomes"]
        self.assertEqual(outcomes["events"]["SHOT_HIT"], 1)
        self.assertEqual(outcomes["ranger_accuracy_percent"], 100.0)

    def test_duplicate_tick_submits_once_with_stable_key(self) -> None:
        submitted: list[str | None] = []
        source = make_turn(tick=42, units=(unit(1, UnitType.WORKER, (1, 0)),))

        def submit(plan, key):
            submitted.append(key)
            return Accepted(
                accepted=True,
                tick=42,
                source=CommandSource.AGENT,
                received_at=datetime.now(timezone.utc),
            )

        first = make_turn(tick=42, units=source.state.objects[1:2], submitter=submit)
        duplicate = make_turn(tick=42, units=source.state.objects[1:2], submitter=submit)
        clients = iter(
            (
                _EventGame((first, duplicate)),
                _SubmissionClient(submit),
                _SubmissionClient(submit),
                _SubmissionClient(submit),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                balanced_tactic,
                "ArenaHeroClient",
                side_effect=lambda **_kwargs: next(clients),
            ):
                with redirect_stdout(io.StringIO()):
                    balanced_tactic.play("test-key", directory)
            heartbeat = Path(directory) / "watchdog" / "tactic_heartbeat.json"
            heartbeat_record = json.loads(heartbeat.read_text(encoding="utf-8"))

        self.assertEqual(submitted, ["arena-balanced-tactic-42"])
        self.assertEqual(heartbeat_record["status"], "COMPLETED")
        self.assertEqual(heartbeat_record["tick"], 42)

    def test_slow_submission_does_not_block_receipt_or_next_turn(self) -> None:
        submission_started = threading.Event()
        allow_http_completion = threading.Event()
        next_turn_reached = threading.Event()
        blocked_event_loop: list[bool] = []
        submitted: list[tuple[int, str | None]] = []

        def accepted(tick: int) -> Accepted:
            return Accepted(
                accepted=True,
                tick=tick,
                source=CommandSource.AGENT,
                received_at=datetime.now(timezone.utc),
            )

        def slow_submit(plan, key):
            submitted.append((plan.tick, key))
            submission_started.set()
            if not allow_http_completion.wait(0.5):
                blocked_event_loop.append(True)
            return accepted(plan.tick)

        def fast_submit(plan, key):
            submitted.append((plan.tick, key))
            return accepted(plan.tick)

        first = make_turn(
            tick=1,
            units=(unit(1, UnitType.WORKER, (1, 0)),),
            submitter=slow_submit,
        )
        second = make_turn(tick=2, submitter=fast_submit)

        class StreamingGame(_EventGame):
            def events(self):
                yield first
                self.assert_submission_started()
                yield Received(
                    tick=1,
                    source=CommandSource.AGENT,
                    received_at=datetime.now(timezone.utc),
                    plan=first.plan.model_copy(deep=True),
                )
                allow_http_completion.set()
                next_turn_reached.set()
                yield second

            @staticmethod
            def assert_submission_started():
                if not submission_started.wait(1.0):
                    raise AssertionError("background submission did not start")

        event_game = StreamingGame(())
        clients = iter(
            (
                event_game,
                _SubmissionClient(slow_submit),
                _SubmissionClient(fast_submit),
                _SubmissionClient(fast_submit),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                balanced_tactic,
                "ArenaHeroClient",
                side_effect=lambda **_kwargs: next(clients),
            ):
                with redirect_stdout(io.StringIO()):
                    balanced_tactic.play("test-key", directory)

            records = [
                json.loads(line)
                for path in Path(directory).glob("*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertFalse(blocked_event_loop)
        self.assertTrue(next_turn_reached.is_set())
        self.assertEqual([tick for tick, _key in submitted], [1, 2])
        self.assertEqual(
            [key for _tick, key in submitted],
            ["arena-balanced-tactic-1", "arena-balanced-tactic-2"],
        )
        self.assertEqual(
            [record["tick"] for record in records if record["record_type"] == "turn"],
            [1, 2],
        )
        receipt_index = next(
            index
            for index, record in enumerate(records)
            if record["record_type"] == "canonical_receipt" and record["tick"] == 1
        )
        http_index = next(
            index
            for index, record in enumerate(records)
            if record["record_type"] == "submission_update"
            and record["tick"] == 1
            and record["event"] == "HTTP_COMPLETED"
        )
        self.assertLess(receipt_index, http_index)
        self.assertTrue(records[http_index]["late_result"])

    def test_submission_recovery_reuses_exact_plan_and_idempotency_key(self) -> None:
        calls: list[tuple[dict[str, object], str | None]] = []

        def submit(plan, key):
            calls.append((plan.model_dump(mode="json"), key))
            if len(calls) == 1:
                raise TransportError("unknown after upload")
            return Accepted(
                accepted=True,
                tick=plan.tick,
                source=CommandSource.AGENT,
                received_at=datetime.now(timezone.utc),
            )

        clients = [_SubmissionClient(submit) for _ in range(3)]
        client_iter = iter(clients)
        coordinator = SubmissionCoordinator(
            lambda: next(client_iter),
            max_inflight_submissions=3,
            soft_deadline=5.0,
        )
        plan = CommandPlan(
            tick=7,
            unit_actions={uid(1): MoveAction(direction=Direction.RIGHT)},
        )
        try:
            first = coordinator.enqueue(
                plan,
                idempotency_key="arena-balanced-tactic-7",
            )
            self.assertIsNotNone(first)
            self.assertTrue(
                self._wait_for_submission_outcome(
                    coordinator,
                    SubmissionOutcome.HTTP_OUTCOME_UNKNOWN,
                )
            )
            recovered = coordinator.recover(7)
            self.assertIsNotNone(recovered)
            self.assertTrue(recovered.reused_request)
            self.assertTrue(
                self._wait_for_submission_outcome(
                    coordinator,
                    SubmissionOutcome.HTTP_ACCEPTED,
                )
            )
        finally:
            coordinator.close(timeout=1.0)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertTrue(all(client.closed for client in clients))

    def test_submission_coordinator_never_exceeds_three_active_lanes(self) -> None:
        release = threading.Event()
        started = [threading.Event() for _ in range(3)]

        def client_for(index: int) -> _SubmissionClient:
            def submit(plan, _key):
                started[index].set()
                release.wait(1.0)
                return Accepted(
                    accepted=True,
                    tick=plan.tick,
                    source=CommandSource.AGENT,
                    received_at=datetime.now(timezone.utc),
                )

            return _SubmissionClient(submit)

        clients = [client_for(index) for index in range(3)]
        client_iter = iter(clients)
        coordinator = SubmissionCoordinator(
            lambda: next(client_iter),
            max_inflight_submissions=3,
            soft_deadline=5.0,
        )
        try:
            for tick in range(1, 4):
                self.assertIsNotNone(
                    coordinator.enqueue(
                        CommandPlan(tick=tick),
                        idempotency_key=f"arena-balanced-tactic-{tick}",
                    )
                )
            self.assertTrue(all(event.wait(1.0) for event in started))
            self.assertTrue(coordinator.saturated)
            self.assertIsNone(
                coordinator.enqueue(
                    CommandPlan(tick=4),
                    idempotency_key="arena-balanced-tactic-4",
                )
            )
            saturated = coordinator.saturated_result(4)
            self.assertEqual(
                saturated.outcome,
                SubmissionOutcome.SUBMITTER_SATURATED,
            )
            self.assertEqual(saturated.concurrent_count, 3)
        finally:
            release.set()
            coordinator.close(timeout=1.0)

    def test_soft_deadline_marks_pending_without_cancelling_submission(self) -> None:
        release = threading.Event()
        started = threading.Event()

        def submit(plan, _key):
            started.set()
            release.wait(1.0)
            return Accepted(
                accepted=True,
                tick=plan.tick,
                source=CommandSource.AGENT,
                received_at=datetime.now(timezone.utc),
            )

        clients = [_SubmissionClient(submit)]
        coordinator = SubmissionCoordinator(
            lambda: clients[0],
            max_inflight_submissions=1,
            soft_deadline=0.01,
        )
        try:
            coordinator.enqueue(
                CommandPlan(tick=1),
                idempotency_key="arena-balanced-tactic-1",
            )
            self.assertTrue(started.wait(1.0))
            threading.Event().wait(0.02)
            events = coordinator.poll()
            self.assertTrue(any(result.event == "SLOW_PENDING" for result in events))
            self.assertEqual(coordinator.state_for_tick(1), "PENDING")
            release.set()
            self.assertTrue(
                self._wait_for_submission_outcome(
                    coordinator,
                    SubmissionOutcome.HTTP_ACCEPTED,
                )
            )
        finally:
            release.set()
            coordinator.close(timeout=1.0)

    def test_background_authentication_rejection_remains_fatal(self) -> None:
        def rejected(_plan, _key):
            raise APIError(
                status_code=401,
                error="UNAUTHORIZED",
                message="unauthorized",
            )

        client = _SubmissionClient(rejected)
        coordinator = SubmissionCoordinator(
            lambda: client,
            max_inflight_submissions=1,
            soft_deadline=5.0,
        )
        try:
            coordinator.enqueue(
                CommandPlan(tick=1),
                idempotency_key="arena-balanced-tactic-1",
            )
            self.assertTrue(
                self._wait_for_submission_outcome(
                    coordinator,
                    SubmissionOutcome.FATAL_REJECTED,
                )
            )
            fatal = coordinator.take_fatal_error()
            self.assertIsInstance(fatal, APIError)
            self.assertEqual(fatal.error, "UNAUTHORIZED")
        finally:
            coordinator.close(timeout=1.0)

    def test_old_tick_confirmation_never_labels_future_memory_as_old(self) -> None:
        release = threading.Event()
        started = {1: threading.Event(), 2: threading.Event()}

        def submit(plan, _key):
            started[plan.tick].set()
            release.wait(1.0)
            return Accepted(
                accepted=True,
                tick=plan.tick,
                source=CommandSource.AGENT,
                received_at=datetime.now(timezone.utc),
            )

        first = make_turn(tick=1)
        second = make_turn(tick=2)

        class OrderedReceiptGame(_EventGame):
            def events(self):
                yield first
                yield second
                if not all(event.wait(1.0) for event in started.values()):
                    raise AssertionError("submissions did not start")
                yield Received(
                    tick=1,
                    source=CommandSource.AGENT,
                    received_at=datetime.now(timezone.utc),
                    plan=first.plan.model_copy(deep=True),
                )
                yield Received(
                    tick=2,
                    source=CommandSource.AGENT,
                    received_at=datetime.now(timezone.utc),
                    plan=second.plan.model_copy(deep=True),
                )
                release.set()

        class RecordingStore:
            def __init__(self):
                self.path = Path("recording-memory.json")
                self.restored_through_tick = None
                self.restored_visit_count = 0
                self.saved_ticks: list[int] = []

            @staticmethod
            def load():
                return TacticMemory()

            def save(self, _memory, *, tick, force=False):
                self.saved_ticks.append(tick)
                return True

        store = RecordingStore()
        clients = iter(
            (
                OrderedReceiptGame(()),
                _SubmissionClient(submit),
                _SubmissionClient(submit),
                _SubmissionClient(submit),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    balanced_tactic,
                    "ArenaHeroClient",
                    side_effect=lambda **_kwargs: next(clients),
                ),
                patch.object(
                    balanced_tactic,
                    "ExplorationMemoryStore",
                    return_value=store,
                ),
                redirect_stdout(io.StringIO()),
            ):
                balanced_tactic.play("test-key", directory)

        self.assertTrue(store.saved_ticks)
        self.assertNotIn(1, store.saved_ticks)
        self.assertTrue(all(tick == 2 for tick in store.saved_ticks))

    @staticmethod
    def _wait_for_submission_outcome(
        coordinator: SubmissionCoordinator,
        outcome: SubmissionOutcome,
    ) -> bool:
        deadline = datetime.now().timestamp() + 1.0
        while datetime.now().timestamp() < deadline:
            if any(result.outcome is outcome for result in coordinator.poll()):
                return True
            threading.Event().wait(0.005)
        return False

    def test_command_window_closed_is_recoverable(self) -> None:
        calls: list[int] = []

        def closed(plan, key):
            if plan.tick == 1:
                calls.append(plan.tick)
                raise APIError(
                    status_code=409,
                    error="COMMAND_WINDOW_CLOSED",
                    message="closed",
                )
            return Accepted(
                accepted=True,
                tick=plan.tick,
                source=CommandSource.AGENT,
                received_at=datetime.now(timezone.utc),
            )

        second = make_turn(tick=2)
        first = make_turn(tick=1, submitter=closed)
        clients = iter(
            (
                _EventGame((first, second)),
                _SubmissionClient(closed),
                _SubmissionClient(closed),
                _SubmissionClient(closed),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                balanced_tactic,
                "ArenaHeroClient",
                side_effect=lambda **_kwargs: next(clients),
            ):
                with redirect_stdout(io.StringIO()):
                    balanced_tactic.play("test-key", directory)

        self.assertEqual(calls, [1])

    def test_submission_transport_timeout_is_recoverable_and_client_is_bounded(self) -> None:
        calls: list[int] = []
        client_options: list[dict[str, object]] = []

        def uncertain(plan, key):
            if plan.tick == 1:
                calls.append(plan.tick)
                raise TransportError("timed out after upload")
            return Accepted(
                accepted=True,
                tick=plan.tick,
                source=CommandSource.AGENT,
                received_at=datetime.now(timezone.utc),
            )

        first = make_turn(tick=1, submitter=uncertain)
        second = make_turn(tick=2)

        event_game = _EventGame((first, second))
        clients = iter(
            (
                event_game,
                _SubmissionClient(uncertain),
                _SubmissionClient(uncertain),
                _SubmissionClient(uncertain),
            )
        )

        def client_factory(**kwargs):
            client_options.append(kwargs)
            return next(clients)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                balanced_tactic,
                "ArenaHeroClient",
                side_effect=client_factory,
            ):
                with redirect_stdout(io.StringIO()):
                    balanced_tactic.play("test-key", directory)

            records = [
                json.loads(line)
                for path in Path(directory).glob("*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(calls, [1])
        self.assertEqual(len(client_options), 4)
        self.assertTrue(
            all(options["request_timeout"] == 2.5 for options in client_options)
        )
        self.assertTrue(
            all(options["request_retries"] == 1 for options in client_options)
        )
        failure = next(
            record
            for record in records
            if record.get("tick") == 1
            and record.get("record_type") == "submission_update"
            and record.get("outcome") == "HTTP_OUTCOME_UNKNOWN"
        )
        self.assertEqual(
            failure["error"]["type"],
            "TransportError",
        )

    def test_manual_move_clears_old_mission_and_blocks_immediate_reversal(self) -> None:
        worker = unit(1, UnitType.WORKER, (1, 0))
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(tick=1, units=(worker,), resources=0))
        receipt = Received(
            tick=1,
            source=CommandSource.MANUAL,
            received_at=datetime.now(timezone.utc),
            plan=CommandPlan(
                tick=1,
                unit_actions={worker.id: MoveAction(direction=Direction.RIGHT)},
            ),
        )
        tactic.observe_receipt(receipt)

        second = make_turn(tick=2, units=(unit(1, UnitType.WORKER, (2, 0)),), resources=0)
        tactic.choose_actions(second)

        action = second.plan.unit_actions[worker.id]
        self.assertFalse(isinstance(action, MoveAction) and action.direction is Direction.LEFT)

    def test_manual_combat_move_does_not_create_a_direction_lease(self) -> None:
        vanguard = unit(1, UnitType.VANGUARD, (0, 0))
        tactic = BalancedTactic()
        tactic.choose_actions(make_turn(tick=1, units=(vanguard,), resources=0))
        receipt = Received(
            tick=1,
            source=CommandSource.MANUAL,
            received_at=datetime.now(timezone.utc),
            plan=CommandPlan(
                tick=1,
                unit_actions={vanguard.id: MoveAction(direction=Direction.RIGHT)},
            ),
        )

        tactic.observe_receipt(receipt)

        self.assertNotIn(vanguard.id, tactic.memory.manual_move_leases)


if __name__ == "__main__":
    unittest.main()
