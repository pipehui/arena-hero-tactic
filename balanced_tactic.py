from __future__ import annotations

import os
import signal
from getpass import getpass
from pathlib import Path
from time import perf_counter

from arena_hero import ArenaHeroClient, Received, Turn

from arena_tactic import (
    DEFAULT_CONFIG,
    BalancedTactic,
    TacticConfig,
    TacticMemory,
    WorkerPatrolMode,
)
from arena_tactic.persistence import ExplorationMemoryStore
from arena_tactic.runtime import (
    HeartbeatWriter,
    InstanceAlreadyRunning,
    SingleInstanceLock,
)
from arena_tactic.submission import (
    SubmissionCoordinator,
    SubmissionOutcome,
    SubmissionResult,
)
from replay_log import ReplayLogger


def choose_actions(turn: Turn, tactic: BalancedTactic | None = None) -> BalancedTactic:
    """Convenience entry point for tests or embedding in another client loop."""

    active_tactic = tactic or BalancedTactic()
    active_tactic.choose_actions(turn)
    return active_tactic


def play(
    api_key: str,
    log_directory: str | os.PathLike[str] | None = None,
    *,
    request_timeout: float = 2.5,
    request_retries: int = 1,
    submission_soft_deadline: float = 5.0,
    max_inflight_submissions: int = 3,
) -> None:
    normalized_api_key = api_key.strip(" \t\r\n\v\f")
    if not normalized_api_key:
        raise ValueError("Arena Hero API key must not be empty")

    log_root = Path(log_directory) if log_directory else Path(__file__).parent / "logs"
    instance_lock = SingleInstanceLock(
        log_root / "watchdog" / "tactic_process.lock"
    )
    with instance_lock:
        _play_locked(
            normalized_api_key,
            log_root,
            request_timeout=request_timeout,
            request_retries=request_retries,
            submission_soft_deadline=submission_soft_deadline,
            max_inflight_submissions=max_inflight_submissions,
        )


def _play_locked(
    normalized_api_key: str,
    log_root: Path,
    *,
    request_timeout: float,
    request_retries: int,
    submission_soft_deadline: float,
    max_inflight_submissions: int,
) -> None:
    memory_store = ExplorationMemoryStore(log_root)
    tactic = BalancedTactic(memory=memory_store.load())
    replay = ReplayLogger(
        log_root,
        request_timeout=request_timeout,
        request_retries=request_retries,
        submission_soft_deadline=submission_soft_deadline,
        max_inflight_submissions=max_inflight_submissions,
    )
    heartbeat = HeartbeatWriter(log_root / "watchdog" / "tactic_heartbeat.json")
    heartbeat.mark("CONNECTING")
    print(f"Replay log: {replay.path}", flush=True)
    print(
        "Exploration memory: "
        f"{memory_store.path} "
        f"(through tick={memory_store.restored_through_tick}, "
        f"visits={memory_store.restored_visit_count})",
        flush=True,
    )

    status = "completed"
    stage = "connecting"
    last_tick: int | None = None
    latest_planned_tick: int | None = None
    last_checkpoint_tick: int | None = None
    turn_failure_logged = False
    terminal_ticks: dict[int, None] = {}
    confirmed_ticks: dict[int, None] = {}
    coordinator: SubmissionCoordinator | None = None

    def mark_terminal(tick: int) -> None:
        terminal_ticks[tick] = None
        while len(terminal_ticks) > 256:
            terminal_ticks.pop(next(iter(terminal_ticks)))

    def mark_confirmed(tick: int) -> None:
        confirmed_ticks[tick] = None
        while len(confirmed_ticks) > 256:
            confirmed_ticks.pop(next(iter(confirmed_ticks)))

    def handle_submission_result(result: SubmissionResult) -> None:
        nonlocal last_checkpoint_tick, stage
        replay.record_submission_result(result, secret=normalized_api_key)
        outcome = result.outcome
        if outcome in {
            SubmissionOutcome.HTTP_ACCEPTED,
            SubmissionOutcome.RECEIPT_CONFIRMED,
        }:
            first_confirmation = result.tick not in confirmed_ticks
            mark_terminal(result.tick)
            mark_confirmed(result.tick)
            if result.tick == latest_planned_tick:
                memory_store.save(tactic.memory, tick=result.tick)
                last_checkpoint_tick = result.tick
            heartbeat.mark("HEALTHY", tick=result.tick)
            if first_confirmation:
                source = (
                    "receipt"
                    if outcome is SubmissionOutcome.RECEIPT_CONFIRMED
                    else "http"
                )
                print(
                    f"tick={result.tick} accepted=True confirmed_by={source}",
                    flush=True,
                )
            return
        if outcome is SubmissionOutcome.HTTP_OUTCOME_UNKNOWN:
            heartbeat.mark("SUBMISSION_OUTCOME_UNKNOWN", tick=result.tick)
            print(
                f"tick={result.tick} submission=outcome_unknown",
                flush=True,
            )
            return
        if outcome is SubmissionOutcome.RECOVERABLE_REJECTED:
            mark_terminal(result.tick)
            heartbeat.mark("RECOVERABLE_SKIP", tick=result.tick)
            print(
                f"tick={result.tick} skipped={(result.error_code or 'rejected').lower()}",
                flush=True,
            )
            return
        if outcome is SubmissionOutcome.SUBMITTER_SATURATED:
            heartbeat.mark("SUBMITTER_SATURATED", tick=result.tick)
            return
        if result.event == "SLOW_PENDING":
            heartbeat.mark("SLOW_PENDING", tick=result.tick)

    def drain_submission_results(*, raise_fatal: bool = True) -> None:
        assert coordinator is not None
        for result in coordinator.poll():
            handle_submission_result(result)
        fatal_error = coordinator.take_fatal_error()
        if fatal_error is not None and raise_fatal:
            raise fatal_error

    try:
        event_client = ArenaHeroClient(
            api_key=normalized_api_key,
            request_timeout=request_timeout,
            request_retries=request_retries,
        )
        with event_client as game:
            coordinator = SubmissionCoordinator(
                lambda: ArenaHeroClient(
                    api_key=normalized_api_key,
                    request_timeout=request_timeout,
                    request_retries=request_retries,
                ),
                max_inflight_submissions=max_inflight_submissions,
                soft_deadline=submission_soft_deadline,
            )
            stage = "waiting_for_turn"
            for event in game.events():
                drain_submission_results()
                if isinstance(event, Received):
                    heartbeat.mark("RECEIPT", tick=event.tick)
                    replay.record_receipt(event)
                    confirmation = coordinator.observe_receipt(event)
                    if confirmation is not None:
                        handle_submission_result(confirmation)
                    receipt_observer = getattr(tactic, "observe_receipt", None)
                    if receipt_observer is not None:
                        receipt_observer(event)
                    continue
                if not isinstance(event, Turn):
                    continue

                turn = event
                last_tick = turn.tick
                heartbeat.mark("PLANNING", tick=turn.tick)
                turn_failure_logged = False
                submission_state = coordinator.state_for_tick(turn.tick)
                if turn.tick in terminal_ticks:
                    replay.record_turn_skip(
                        tick=turn.tick,
                        reason="DUPLICATE_TERMINAL_TICK",
                    )
                    print(
                        f"tick={turn.tick} skipped=duplicate_terminal_tick",
                        flush=True,
                    )
                    continue
                if submission_state == "PENDING":
                    replay.record_turn_skip(
                        tick=turn.tick,
                        reason="DUPLICATE_PENDING_TICK",
                    )
                    print(
                        f"tick={turn.tick} skipped=duplicate_pending_tick",
                        flush=True,
                    )
                    continue
                if submission_state == "HTTP_OUTCOME_UNKNOWN":
                    recovered = coordinator.recover(turn.tick)
                    if recovered is None:
                        saturated = coordinator.saturated_result(turn.tick)
                        handle_submission_result(saturated)
                        replay.record_turn_skip(
                            tick=turn.tick,
                            reason="SUBMITTER_SATURATED_RECOVERY",
                        )
                    else:
                        heartbeat.mark("RECOVERY_QUEUED", tick=turn.tick)
                        drain_submission_results()
                    continue
                if submission_state is not None:
                    replay.record_turn_skip(
                        tick=turn.tick,
                        reason=f"DUPLICATE_{submission_state}",
                    )
                    continue
                if coordinator.saturated:
                    saturated = coordinator.saturated_result(turn.tick)
                    handle_submission_result(saturated)
                    replay.record_turn_skip(
                        tick=turn.tick,
                        reason="SUBMITTER_SATURATED",
                    )
                    print(
                        f"tick={turn.tick} skipped=submitter_saturated",
                        flush=True,
                    )
                    continue

                stage = "planning"
                decision_started = perf_counter()
                try:
                    tactic.choose_actions(turn)
                except Exception as error:
                    decision_ms = (perf_counter() - decision_started) * 1_000
                    turn_failure_logged = True
                    replay.record_turn(
                        turn,
                        decision_ms=decision_ms,
                        submission_ms=None,
                        error=error,
                        failure_stage=stage,
                        secret=normalized_api_key,
                        strategy=tactic.last_decision_trace,
                    )
                    raise
                decision_ms = (perf_counter() - decision_started) * 1_000
                latest_planned_tick = turn.tick
                stage = "queueing_submission"
                pending = coordinator.enqueue(
                    turn.plan,
                    idempotency_key=f"arena-balanced-tactic-{turn.tick}",
                )
                if pending is None:
                    raise RuntimeError(
                        "submission lane disappeared after capacity check"
                    )
                replay.record_turn(
                    turn,
                    decision_ms=decision_ms,
                    submission_ms=None,
                    pending=pending,
                    strategy=tactic.last_decision_trace,
                )
                heartbeat.mark("SUBMISSION_QUEUED", tick=turn.tick)
                print(
                    f"tick={turn.tick} submission=queued lane={pending.lane_id}",
                    flush=True,
                )
                stage = "waiting_for_turn"
                drain_submission_results()
        coordinator.close(timeout=30.0)
        drain_submission_results()
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as error:
        status = "error"
        if not turn_failure_logged:
            replay.record_error(
                stage=stage,
                tick=last_tick,
                error=error,
                secret=normalized_api_key,
            )
        raise
    finally:
        if coordinator is not None:
            coordinator.close(timeout=30.0)
            drain_submission_results(raise_fatal=False)
        if last_checkpoint_tick is not None and last_checkpoint_tick == latest_planned_tick:
            memory_store.save(
                tactic.memory,
                tick=last_checkpoint_tick,
                force=True,
            )
        heartbeat.mark(status.upper(), tick=last_tick)
        replay.close(status=status, last_tick=last_tick)


def main() -> None:
    previous_sigterm = None
    if hasattr(signal, "SIGTERM"):
        try:
            previous_sigterm = signal.getsignal(signal.SIGTERM)

            def stop_gracefully(_signum, _frame) -> None:
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, stop_gracefully)
        except (OSError, ValueError):
            # Signal registration is only legal in the main interpreter
            # thread. Embedded callers still retain the normal play() API.
            previous_sigterm = None
    api_key = os.environ.get("ARENA_HERO_API_KEY") or getpass(
        "Arena Hero API key: "
    )
    try:
        play(api_key, os.environ.get("ARENA_HERO_LOG_DIR"))
    except InstanceAlreadyRunning as error:
        print(f"Arena Hero tactic was not started: {error}")
    except KeyboardInterrupt:
        print("\nArena Hero tactic stopped.")
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()


__all__ = (
    "ArenaHeroClient",
    "BalancedTactic",
    "DEFAULT_CONFIG",
    "TacticConfig",
    "TacticMemory",
    "WorkerPatrolMode",
    "choose_actions",
    "main",
    "play",
)
