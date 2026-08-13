from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable

from arena_hero import APIError, Accepted, CommandPlan, CommandSource, Received, TransportError


class SubmissionOutcome(str, Enum):
    HTTP_ACCEPTED = "HTTP_ACCEPTED"
    RECEIPT_CONFIRMED = "RECEIPT_CONFIRMED"
    HTTP_OUTCOME_UNKNOWN = "HTTP_OUTCOME_UNKNOWN"
    RECOVERABLE_REJECTED = "RECOVERABLE_REJECTED"
    FATAL_REJECTED = "FATAL_REJECTED"
    SUBMITTER_SATURATED = "SUBMITTER_SATURATED"


@dataclass(frozen=True)
class PendingSubmission:
    tick: int
    plan: CommandPlan
    idempotency_key: str
    plan_hash: str
    lane_id: int
    queued_at: datetime
    queued_clock: float
    concurrent_count: int
    reused_request: bool = False
    attempt_number: int = 1


@dataclass(frozen=True)
class SubmissionResult:
    tick: int
    event: str
    outcome: SubmissionOutcome | None
    lane_id: int | None
    plan_hash: str | None
    recorded_at: datetime
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    queue_ms: float | None = None
    http_ms: float | None = None
    receipt_ms: float | None = None
    concurrent_count: int | None = None
    slow_pending: bool = False
    receipt_confirmed: bool = False
    late_result: bool = False
    reused_request: bool = False
    attempt_number: int = 1
    accepted_tick: int | None = None
    accepted_at: datetime | None = None
    error_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class _TrackedSubmission:
    pending: PendingSubmission
    started_at: datetime | None = None
    started_clock: float | None = None
    slow_reported: bool = False
    in_flight: bool = True
    receipt_confirmed: bool = False
    authoritative_outcome: SubmissionOutcome | None = None


@dataclass
class _Lane:
    lane_id: int
    client: Any
    inbox: queue.Queue[PendingSubmission | None]
    thread: threading.Thread | None = None
    busy_tick: int | None = None


def plan_fingerprint(plan: CommandPlan) -> str:
    payload = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class SubmissionCoordinator:
    """Run exact SDK submissions off the WebSocket event-consumer thread."""

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        max_inflight_submissions: int = 3,
        soft_deadline: float = 5.0,
    ) -> None:
        if not 1 <= max_inflight_submissions <= 3:
            raise ValueError("max_inflight_submissions must be between 1 and 3")
        if soft_deadline <= 0:
            raise ValueError("submission soft deadline must be positive")

        self.max_inflight_submissions = max_inflight_submissions
        self.soft_deadline = soft_deadline
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._results: queue.SimpleQueue[SubmissionResult] = queue.SimpleQueue()
        self._tracked: dict[int, _TrackedSubmission] = {}
        self._fatal_errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()
        self._accepting = True
        self._closed = False
        self._lanes: list[_Lane] = []

        try:
            for lane_id in range(max_inflight_submissions):
                self._lanes.append(
                    _Lane(
                        lane_id=lane_id,
                        client=client_factory(),
                        inbox=queue.Queue(maxsize=1),
                    )
                )
        except Exception:
            for lane in self._lanes:
                lane.client.close()
            raise

        for lane in self._lanes:
            lane.thread = threading.Thread(
                target=self._run_lane,
                args=(lane,),
                name=f"arena-submit-{lane.lane_id}",
                daemon=True,
            )
            lane.thread.start()

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return sum(lane.busy_tick is not None for lane in self._lanes)

    @property
    def saturated(self) -> bool:
        return self.inflight_count >= self.max_inflight_submissions

    def state_for_tick(self, tick: int) -> str | None:
        with self._lock:
            tracked = self._tracked.get(tick)
            if tracked is None:
                return None
            if tracked.in_flight:
                return "PENDING"
            if tracked.authoritative_outcome is None:
                return "HTTP_OUTCOME_UNKNOWN"
            return tracked.authoritative_outcome.value

    def enqueue(
        self,
        plan: CommandPlan,
        *,
        idempotency_key: str,
        reused_request: bool = False,
        attempt_number: int = 1,
    ) -> PendingSubmission | None:
        detached = plan.model_copy(deep=True)
        with self._lock:
            if not self._accepting:
                return None
            existing = self._tracked.get(detached.tick)
            if existing is not None and not reused_request:
                return None
            lane = next((item for item in self._lanes if item.busy_tick is None), None)
            if lane is None:
                return None
            concurrent_count = self.inflight_count + 1
            pending = PendingSubmission(
                tick=detached.tick,
                plan=detached,
                idempotency_key=idempotency_key,
                plan_hash=plan_fingerprint(detached),
                lane_id=lane.lane_id,
                queued_at=datetime.now().astimezone(),
                queued_clock=perf_counter(),
                concurrent_count=concurrent_count,
                reused_request=reused_request,
                attempt_number=attempt_number,
            )
            self._tracked[detached.tick] = _TrackedSubmission(pending=pending)
            self._trim_history_locked()
            lane.busy_tick = detached.tick
            lane.inbox.put_nowait(pending)
            self._results.put(
                SubmissionResult(
                    tick=pending.tick,
                    event="RECOVERY_QUEUED" if reused_request else "QUEUED",
                    outcome=None,
                    lane_id=pending.lane_id,
                    plan_hash=pending.plan_hash,
                    recorded_at=pending.queued_at,
                    queued_at=pending.queued_at,
                    concurrent_count=pending.concurrent_count,
                    reused_request=pending.reused_request,
                    attempt_number=pending.attempt_number,
                )
            )
            return pending

    def recover(self, tick: int) -> PendingSubmission | None:
        with self._lock:
            tracked = self._tracked.get(tick)
            if (
                tracked is None
                or tracked.in_flight
                or tracked.authoritative_outcome
                is not SubmissionOutcome.HTTP_OUTCOME_UNKNOWN
            ):
                return None
            plan = tracked.pending.plan
            key = tracked.pending.idempotency_key
            attempt_number = tracked.pending.attempt_number + 1
        return self.enqueue(
            plan,
            idempotency_key=key,
            reused_request=True,
            attempt_number=attempt_number,
        )

    def observe_receipt(self, receipt: Received) -> SubmissionResult | None:
        if receipt.source is not CommandSource.AGENT:
            return None
        receipt_hash = plan_fingerprint(receipt.plan)
        now = datetime.now().astimezone()
        now_clock = perf_counter()
        with self._lock:
            tracked = self._tracked.get(receipt.tick)
            if tracked is None or tracked.pending.plan_hash != receipt_hash:
                return None
            if tracked.receipt_confirmed:
                return None
            tracked.receipt_confirmed = True
            if tracked.authoritative_outcome is not SubmissionOutcome.HTTP_ACCEPTED:
                tracked.authoritative_outcome = SubmissionOutcome.RECEIPT_CONFIRMED
            pending = tracked.pending
            result = SubmissionResult(
                tick=receipt.tick,
                event="RECEIPT_CONFIRMED",
                outcome=SubmissionOutcome.RECEIPT_CONFIRMED,
                lane_id=pending.lane_id,
                plan_hash=pending.plan_hash,
                recorded_at=now,
                queued_at=pending.queued_at,
                started_at=tracked.started_at,
                receipt_ms=round((now_clock - pending.queued_clock) * 1_000, 3),
                concurrent_count=pending.concurrent_count,
                slow_pending=tracked.slow_reported,
                receipt_confirmed=True,
                reused_request=pending.reused_request,
                attempt_number=pending.attempt_number,
                accepted_tick=receipt.tick,
                accepted_at=receipt.received_at,
            )
            return result

    def poll(self) -> tuple[SubmissionResult, ...]:
        now = datetime.now().astimezone()
        now_clock = perf_counter()
        slow_results: list[SubmissionResult] = []
        with self._lock:
            for tracked in self._tracked.values():
                if (
                    not tracked.in_flight
                    or tracked.started_clock is None
                    or tracked.slow_reported
                    or now_clock - tracked.started_clock < self.soft_deadline
                ):
                    continue
                tracked.slow_reported = True
                pending = tracked.pending
                slow_results.append(
                    SubmissionResult(
                        tick=pending.tick,
                        event="SLOW_PENDING",
                        outcome=None,
                        lane_id=pending.lane_id,
                        plan_hash=pending.plan_hash,
                        recorded_at=now,
                        queued_at=pending.queued_at,
                        started_at=tracked.started_at,
                        http_ms=round(
                            (now_clock - tracked.started_clock) * 1_000,
                            3,
                        ),
                        concurrent_count=pending.concurrent_count,
                        slow_pending=True,
                        receipt_confirmed=tracked.receipt_confirmed,
                        reused_request=pending.reused_request,
                        attempt_number=pending.attempt_number,
                    )
                )
        completed: list[SubmissionResult] = []
        while True:
            try:
                completed.append(self._results.get_nowait())
            except queue.Empty:
                break
        return tuple(slow_results + completed)

    def take_fatal_error(self) -> BaseException | None:
        try:
            return self._fatal_errors.get_nowait()
        except queue.Empty:
            return None

    def saturated_result(self, tick: int) -> SubmissionResult:
        return SubmissionResult(
            tick=tick,
            event="SUBMITTER_SATURATED",
            outcome=SubmissionOutcome.SUBMITTER_SATURATED,
            lane_id=None,
            plan_hash=None,
            recorded_at=datetime.now().astimezone(),
            concurrent_count=self.inflight_count,
        )

    def close(self, *, timeout: float = 30.0) -> None:
        if timeout < 0:
            raise ValueError("submission shutdown timeout cannot be negative")
        with self._condition:
            if self._closed:
                return
            self._accepting = False
            deadline = perf_counter() + timeout
            while any(lane.busy_tick is not None for lane in self._lanes):
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            self._closed = True

        for lane in self._lanes:
            try:
                lane.inbox.put_nowait(None)
            except queue.Full:
                pass
        for lane in self._lanes:
            remaining = max(0.0, deadline - perf_counter())
            if lane.thread is not None and remaining:
                lane.thread.join(timeout=remaining)
            if lane.thread is not None and lane.thread.is_alive():
                try:
                    lane.inbox.put_nowait(None)
                except queue.Full:
                    pass
                lane.thread.join(timeout=0.05)
            lane.client.close()

    def _run_lane(self, lane: _Lane) -> None:
        while True:
            pending = lane.inbox.get()
            if pending is None:
                return
            self._mark_started(pending)
            try:
                accepted = lane.client.submit(
                    pending.plan,
                    idempotency_key=pending.idempotency_key,
                )
            except APIError as error:
                self._mark_api_error(pending, error)
            except TransportError as error:
                self._mark_completed(
                    pending,
                    SubmissionOutcome.HTTP_OUTCOME_UNKNOWN,
                    error=error,
                )
            except BaseException as error:
                self._mark_completed(
                    pending,
                    SubmissionOutcome.FATAL_REJECTED,
                    error=error,
                )
                self._fatal_errors.put(error)
            else:
                self._mark_completed(
                    pending,
                    SubmissionOutcome.HTTP_ACCEPTED,
                    accepted=accepted,
                )

    def _mark_started(self, pending: PendingSubmission) -> None:
        now = datetime.now().astimezone()
        now_clock = perf_counter()
        with self._lock:
            tracked = self._tracked.get(pending.tick)
            if tracked is None or tracked.pending is not pending:
                return
            tracked.started_at = now
            tracked.started_clock = now_clock
        self._results.put(
            SubmissionResult(
                tick=pending.tick,
                event="HTTP_STARTED",
                outcome=None,
                lane_id=pending.lane_id,
                plan_hash=pending.plan_hash,
                recorded_at=now,
                queued_at=pending.queued_at,
                started_at=now,
                queue_ms=round((now_clock - pending.queued_clock) * 1_000, 3),
                concurrent_count=pending.concurrent_count,
                reused_request=pending.reused_request,
                attempt_number=pending.attempt_number,
            )
        )

    def _mark_api_error(self, pending: PendingSubmission, error: APIError) -> None:
        if error.status_code == 409 and error.error in {
            "COMMAND_WINDOW_CLOSED",
            "TICK_MISMATCH",
        }:
            outcome = SubmissionOutcome.RECOVERABLE_REJECTED
        else:
            outcome = SubmissionOutcome.FATAL_REJECTED
        self._mark_completed(pending, outcome, error=error)
        if outcome is SubmissionOutcome.FATAL_REJECTED:
            self._fatal_errors.put(error)

    def _mark_completed(
        self,
        pending: PendingSubmission,
        outcome: SubmissionOutcome,
        *,
        accepted: Accepted | None = None,
        error: BaseException | None = None,
    ) -> None:
        now = datetime.now().astimezone()
        now_clock = perf_counter()
        with self._condition:
            tracked = self._tracked.get(pending.tick)
            if tracked is None or tracked.pending is not pending:
                return
            tracked.in_flight = False
            late_result = tracked.receipt_confirmed
            if outcome is SubmissionOutcome.HTTP_ACCEPTED:
                if not tracked.receipt_confirmed:
                    tracked.authoritative_outcome = outcome
            elif outcome is SubmissionOutcome.HTTP_OUTCOME_UNKNOWN:
                if not tracked.receipt_confirmed:
                    tracked.authoritative_outcome = outcome
            elif not tracked.receipt_confirmed:
                tracked.authoritative_outcome = outcome
            lane = self._lanes[pending.lane_id]
            if lane.busy_tick == pending.tick:
                lane.busy_tick = None
            started_clock = tracked.started_clock or pending.queued_clock
            crossed_soft_deadline = (
                now_clock - started_clock >= self.soft_deadline
            )
            if crossed_soft_deadline and not tracked.slow_reported:
                tracked.slow_reported = True
                self._results.put(
                    SubmissionResult(
                        tick=pending.tick,
                        event="SLOW_PENDING",
                        outcome=None,
                        lane_id=pending.lane_id,
                        plan_hash=pending.plan_hash,
                        recorded_at=now,
                        queued_at=pending.queued_at,
                        started_at=tracked.started_at,
                        http_ms=round(self.soft_deadline * 1_000, 3),
                        concurrent_count=pending.concurrent_count,
                        slow_pending=True,
                        receipt_confirmed=tracked.receipt_confirmed,
                        reused_request=pending.reused_request,
                        attempt_number=pending.attempt_number,
                    )
                )
            result = SubmissionResult(
                tick=pending.tick,
                event="HTTP_COMPLETED",
                outcome=outcome,
                lane_id=pending.lane_id,
                plan_hash=pending.plan_hash,
                recorded_at=now,
                queued_at=pending.queued_at,
                started_at=tracked.started_at,
                completed_at=now,
                queue_ms=(
                    None
                    if tracked.started_clock is None
                    else round(
                        (tracked.started_clock - pending.queued_clock) * 1_000,
                        3,
                    )
                ),
                http_ms=round((now_clock - started_clock) * 1_000, 3),
                concurrent_count=pending.concurrent_count,
                slow_pending=(
                    tracked.slow_reported
                    or crossed_soft_deadline
                ),
                receipt_confirmed=tracked.receipt_confirmed,
                late_result=late_result,
                reused_request=pending.reused_request,
                attempt_number=pending.attempt_number,
                accepted_tick=None if accepted is None else accepted.tick,
                accepted_at=None if accepted is None else accepted.received_at,
                error_type=None if error is None else type(error).__name__,
                error_code=None if not isinstance(error, APIError) else error.error,
                error_message=None if error is None else str(error),
            )
            self._results.put(result)
            self._condition.notify_all()

    def _trim_history_locked(self) -> None:
        if len(self._tracked) <= 256:
            return
        for tick in tuple(sorted(self._tracked)):
            tracked = self._tracked[tick]
            if tracked.in_flight:
                continue
            del self._tracked[tick]
            if len(self._tracked) <= 256:
                return


__all__ = (
    "PendingSubmission",
    "SubmissionCoordinator",
    "SubmissionOutcome",
    "SubmissionResult",
    "plan_fingerprint",
)
