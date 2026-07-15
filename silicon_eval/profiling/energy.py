"""Energy sampling via macOS ``powermetrics``.

``powermetrics`` needs root, so the sampler launches it through ``sudo -n``
(non-interactive). Without a cached credential or a passwordless sudoers rule
it degrades gracefully: ``available`` is False, a human-readable ``reason``
says why, and no energy figures are reported.

Caveat recorded wherever the numbers surface: powermetrics reports
*system-wide* CPU+GPU+ANE power, so readings include the machine's baseline
load, not just the model. Keep the machine otherwise idle while sampling.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from silicon_eval.exceptions import InvalidStateError

_COMBINED_POWER = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\): (?P<mw>\d+(?:\.\d+)?) mW")

_POWERMETRICS_CMD = [
    "sudo",
    "-n",
    "powermetrics",
    "--samplers",
    "cpu_power",
    "-i",
    "200",
]


def parse_combined_power_mw(text: str) -> list[float]:
    """Extract each sample's combined CPU+GPU+ANE power draw in milliwatts."""
    return [float(match.group("mw")) for match in _COMBINED_POWER.finditer(text)]


@dataclass(frozen=True, slots=True)
class EnergyReading:
    """Aggregate of one sampling session (power in mW, energy in mJ = mW·s)."""

    mean_power_mw: float
    energy_mj: float
    samples: int
    duration_s: float


class PowerMetricsSampler:
    """Single-use context manager sampling system power for the ``with`` block.

    Check :attr:`available` inside the block — when False (no sudo, not
    macOS, powermetrics missing), skip the workload you meant to measure and
    :meth:`reading` returns ``None`` with :attr:`unavailable_reason` set.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_path: Path | None = None
        self._started_at = 0.0
        self._reading: EnergyReading | None = None
        self._used = False
        self.available = False
        self.unavailable_reason: str | None = None

    def __enter__(self) -> PowerMetricsSampler:
        if self._used:
            raise InvalidStateError("PowerMetricsSampler is single-use; create a new instance")
        self._used = True
        try:
            self._process = self._launch()
        except OSError as exc:
            self.unavailable_reason = f"could not launch powermetrics: {exc}"
            return self
        # sudo -n fails within moments when no credential is cached.
        time.sleep(0.3)
        if self._process.poll() is not None:
            stderr = self._read_stderr()
            self.unavailable_reason = "powermetrics requires passwordless sudo" + (
                f" ({stderr})" if stderr else ""
            )
            self._cleanup()
            return self
        self.available = True
        self._started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._process is None:
            return
        duration = time.perf_counter() - self._started_at
        self._stop_process()
        self._reading = _aggregate(parse_combined_power_mw(self._read_stdout()), duration)
        self._cleanup()

    def reading(self) -> EnergyReading | None:
        """The session's aggregate, or None if sampling was unavailable/empty."""
        return self._reading

    def _stop_process(self) -> None:
        if self._process is None:
            return
        self._signal_process("TERM")
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._signal_process("KILL")
            self._process.wait()

    def _signal_process(self, signal_name: str) -> None:
        """Signal powermetrics, escalating through sudo when it runs as root.

        The sampled process is root-owned (launched via sudo), so an
        unprivileged SIGTERM raises PermissionError — exactly on the machines
        where sampling works. sudo -n kill succeeds there by construction:
        sampling being available means passwordless sudo is configured.
        """
        if self._process is None:
            return
        try:
            if signal_name == "KILL":
                self._process.kill()
            else:
                self._process.terminate()
        except PermissionError:
            subprocess.run(
                ["sudo", "-n", "kill", f"-{signal_name}", str(self._process.pid)],
                check=False,
                capture_output=True,
            )

    def _launch(self) -> subprocess.Popen[bytes]:
        # stdout goes to a temp file, not a pipe: powermetrics emits a steady
        # stream, and an unread 64 KB pipe would fill within seconds and stall
        # the sampler mid-session.
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="silicon-eval-powermetrics-", suffix=".txt", delete=False
        ) as stdout_file:
            self._stdout_path = Path(stdout_file.name)
            return subprocess.Popen(
                _POWERMETRICS_CMD,
                stdout=stdout_file,
                stderr=subprocess.PIPE,
            )

    def _read_stdout(self) -> str:
        if self._stdout_path is None:
            return ""
        try:
            return self._stdout_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _read_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        data: bytes = self._process.stderr.read()
        return data.decode(errors="replace").strip()

    def _cleanup(self) -> None:
        if self._stdout_path is not None:
            self._stdout_path.unlink(missing_ok=True)
            self._stdout_path = None
        self._process = None


def _aggregate(samples_mw: list[float], duration_s: float) -> EnergyReading | None:
    if not samples_mw or duration_s <= 0:
        return None
    mean_mw = sum(samples_mw) / len(samples_mw)
    return EnergyReading(
        mean_power_mw=mean_mw,
        energy_mj=mean_mw * duration_s,
        samples=len(samples_mw),
        duration_s=duration_s,
    )
