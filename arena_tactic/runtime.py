from __future__ import annotations

import json
import os
import threading
import warnings
from datetime import datetime
from pathlib import Path


class InstanceAlreadyRunning(RuntimeError):
    """Raised when another tactic process owns the workspace lock."""


_LOCAL_LOCKS: set[Path] = set()
_LOCAL_LOCKS_GUARD = threading.Lock()


class SingleInstanceLock:
    """Cross-process, crash-safe lock backed by the operating system.

    The small marker file may remain after a crash, but the byte-range/file
    lock is released by the OS when the owning process exits.  That avoids the
    stale-PID race of an ``O_EXCL`` sentinel while keeping the implementation
    dependency-free on Windows and POSIX.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._stream = None
        self._owned = False

    def acquire(self) -> None:
        if self._owned:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCAL_LOCKS_GUARD:
            if self.path in _LOCAL_LOCKS:
                raise InstanceAlreadyRunning(
                    f"Arena Hero tactic is already running for {self.path.parent}"
                )
        # Keep byte zero alive for the whole lifetime of the Windows range
        # lock.  Truncating a locked file makes a later ``LK_UNLCK`` fail on
        # Windows, while an append-mode stream ignores seeks when updating the
        # diagnostic PID marker.
        self.path.touch(exist_ok=True)
        stream = self.path.open("r+b")
        try:
            self._lock_stream(stream)
        except OSError as error:
            stream.close()
            raise InstanceAlreadyRunning(
                f"Arena Hero tactic is already running for {self.path.parent}"
            ) from error
        with _LOCAL_LOCKS_GUARD:
            if self.path in _LOCAL_LOCKS:
                self._unlock_stream(stream)
                stream.close()
                raise InstanceAlreadyRunning(
                    f"Arena Hero tactic is already running for {self.path.parent}"
                )
            _LOCAL_LOCKS.add(self.path)
        stream.seek(0)
        marker = f"{os.getpid():<31}\n".encode("ascii")
        stream.write(marker)
        stream.flush()
        self._stream = stream
        self._owned = True

    def release(self) -> None:
        if not self._owned or self._stream is None:
            return
        stream = self._stream
        self._stream = None
        self._owned = False
        try:
            self._unlock_stream(stream)
        finally:
            stream.close()
            with _LOCAL_LOCKS_GUARD:
                _LOCAL_LOCKS.discard(self.path)

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(self, *_args) -> None:
        self.release()

    @staticmethod
    def _lock_stream(stream) -> None:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if stream.read(1) == b"":
                stream.seek(0)
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock_stream(stream) -> None:
        if os.name == "nt":
            # Closing the CRT descriptor releases all of its byte-range locks.
            # Explicit ``LK_UNLCK`` is unreliable after a failed competing
            # non-blocking acquisition on some Windows CRT versions.
            return
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class HeartbeatWriter:
    """Best-effort atomic liveness signal for the on-demand watchdog."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._disabled = False

    def mark(self, status: str, *, tick: int | None = None) -> None:
        if self._disabled:
            return
        payload = {
            "process_id": os.getpid(),
            "status": status,
            "tick": tick,
            "recorded_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError as error:
            self._disabled = True
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            warnings.warn(
                f"Tactic heartbeat disabled after an I/O error: {error}",
                RuntimeWarning,
                stacklevel=2,
            )


__all__ = (
    "HeartbeatWriter",
    "InstanceAlreadyRunning",
    "SingleInstanceLock",
)
