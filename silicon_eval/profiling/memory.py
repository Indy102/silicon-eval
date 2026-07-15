"""Process-level memory sampling via psutil.

Complements the runtime-reported Metal allocator peak. Caveat: on macOS, RSS
reliably captures CPU-side allocations (tokenizer, Python overhead, mmapped
weights that have been touched) but may undercount GPU/Metal wired memory —
treat the runtime-reported Metal peak as the authoritative accelerator
number and RSS as the host-side complement.
"""

from __future__ import annotations

import threading
from types import TracebackType

import psutil

from silicon_eval.exceptions import InvalidStateError


class RssSampler:
    """Samples this process's RSS on a background thread; single-use context manager.

    Peak is available via :attr:`peak_rss_bytes` during and after the
    ``with`` block. Samples once synchronously on enter and exit so even very
    short sections get a reading.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self._interval_s = interval_s
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._samples = 0

    def __enter__(self) -> RssSampler:
        if self._thread is not None:
            raise InvalidStateError("RssSampler is single-use; create a new instance")
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()

    @property
    def peak_rss_bytes(self) -> int:
        """Largest RSS observed so far (valid during and after sampling)."""
        return self._peak

    @property
    def samples_taken(self) -> int:
        """Number of RSS samples collected (enter + loop ticks + exit)."""
        return self._samples

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def _sample(self) -> None:
        self._peak = max(self._peak, int(self._process.memory_info().rss))
        self._samples += 1
