from __future__ import annotations

import os
import signal
from getpass import getpass
from pathlib import Path
from time import perf_counter

from arena_hero import APIError, ArenaHeroClient, Received, TransportError, Turn

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
        )


def _play_locked(
    normalized_api_key: str,
    log_root: Path,
    *,
    request_timeout: float,
    request_retries: int,
) -> None:
    memory_store = ExplorationMemoryStore(log_root)
    tactic = BalancedTactic(memory=memory_store.load())
    replay = ReplayLogger(
        log_root,
        request_timeout=request_timeout,
        request_retries=request_retries,
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
    last_accepted_tick: int | None = None
    turn_failure_logged = False
    terminal_ticks: dict[int, None] = {}

    def mark_terminal(tick: int) -> None:
        terminal_ticks[tick] = None
        while len(terminal_ticks) > 256:
            terminal_ticks.pop(next(iter(terminal_ticks)))

    try:
        with ArenaHeroClient(
            api_key=normalized_api_key,
            request_timeout=request_timeout,
            request_retries=request_retries,
        ) as game:
            stage = "waiting_for_turn"
            for event in game.events():
                if isinstance(event, Received):
                    heartbeat.mark("RECEIPT", tick=event.tick)
                    replay.record_receipt(event)
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

                stage = "submitting"
                submission_started = perf_counter()
                try:
                    accepted = turn.submit(
                        idempotency_key=f"arena-balanced-tactic-{turn.tick}"
                    )
                except APIError as error:
                    submission_ms = (perf_counter() - submission_started) * 1_000
                    if (
                        error.status_code == 409
                        and error.error
                        in {"COMMAND_WINDOW_CLOSED", "TICK_MISMATCH"}
                    ):
                        turn_failure_logged = True
                        replay.record_turn(
                            turn,
                            decision_ms=decision_ms,
                            submission_ms=submission_ms,
                            error=error,
                            failure_stage=stage,
                            secret=normalized_api_key,
                            strategy=tactic.last_decision_trace,
                            recoverable=True,
                        )
                        mark_terminal(turn.tick)
                        heartbeat.mark("RECOVERABLE_SKIP", tick=turn.tick)
                        print(
                            f"tick={turn.tick} skipped={error.error.lower()}",
                            flush=True,
                        )
                        stage = "waiting_for_turn"
                        turn_failure_logged = False
                        continue
                    turn_failure_logged = True
                    replay.record_turn(
                        turn,
                        decision_ms=decision_ms,
                        submission_ms=submission_ms,
                        error=error,
                        failure_stage=stage,
                        secret=normalized_api_key,
                        strategy=tactic.last_decision_trace,
                    )
                    raise
                except TransportError as error:
                    # The request may have reached the server even though the
                    # response was lost.  Replanning or changing the
                    # idempotency key would risk submitting two plans for the
                    # same Tick, so mark the Tick terminal locally and wait for
                    # the next authoritative Turn/receipt.
                    submission_ms = (perf_counter() - submission_started) * 1_000
                    turn_failure_logged = True
                    replay.record_turn(
                        turn,
                        decision_ms=decision_ms,
                        submission_ms=submission_ms,
                        error=error,
                        failure_stage=stage,
                        secret=normalized_api_key,
                        strategy=tactic.last_decision_trace,
                        recoverable=True,
                    )
                    mark_terminal(turn.tick)
                    heartbeat.mark("SUBMISSION_OUTCOME_UNKNOWN", tick=turn.tick)
                    print(
                        f"tick={turn.tick} submission=outcome_unknown",
                        flush=True,
                    )
                    stage = "waiting_for_turn"
                    turn_failure_logged = False
                    continue
                except Exception as error:
                    submission_ms = (perf_counter() - submission_started) * 1_000
                    turn_failure_logged = True
                    replay.record_turn(
                        turn,
                        decision_ms=decision_ms,
                        submission_ms=submission_ms,
                        error=error,
                        failure_stage=stage,
                        secret=normalized_api_key,
                        strategy=tactic.last_decision_trace,
                    )
                    raise
                submission_ms = (perf_counter() - submission_started) * 1_000
                mark_terminal(turn.tick)
                last_accepted_tick = turn.tick

                # Persist after submission so logging does not consume the command window.
                replay.record_turn(
                    turn,
                    decision_ms=decision_ms,
                    submission_ms=submission_ms,
                    accepted=accepted,
                    strategy=tactic.last_decision_trace,
                )
                memory_store.save(tactic.memory, tick=turn.tick)
                heartbeat.mark("HEALTHY", tick=turn.tick)
                print(
                    f"tick={accepted.tick} accepted={accepted.accepted}",
                    flush=True,
                )
                stage = "waiting_for_turn"
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
        if (
            last_accepted_tick is not None
            and (last_tick is None or last_tick == last_accepted_tick)
        ):
            memory_store.save(
                tactic.memory,
                tick=last_accepted_tick,
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
