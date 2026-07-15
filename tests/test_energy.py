"""Tests for powermetrics parsing and the energy sampler (process mocked)."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from silicon_eval.exceptions import InvalidStateError
from silicon_eval.profiling.energy import (
    PowerMetricsSampler,
    _aggregate,
    parse_combined_power_mw,
)

SAMPLE_OUTPUT = """\
*** Sampled system activity (Tue Jul 14 10:00:00 2026 -0700) (201.5ms elapsed) ***

CPU Power: 450 mW
GPU Power: 120 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 570 mW

*** Sampled system activity (Tue Jul 14 10:00:00 2026 -0700) (200.1ms elapsed) ***

CPU Power: 900 mW
GPU Power: 2100 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 3000 mW

*** Sampled system activity (Tue Jul 14 10:00:01 2026 -0700) (200.0ms elapsed) ***

Combined Power (CPU + GPU + ANE): 1430.5 mW
"""


class TestParsing:
    def test_extracts_all_combined_power_samples(self) -> None:
        assert parse_combined_power_mw(SAMPLE_OUTPUT) == [570.0, 3000.0, 1430.5]

    def test_no_samples_in_unrelated_text(self) -> None:
        assert parse_combined_power_mw("CPU Power: 450 mW\n") == []

    def test_aggregate_math(self) -> None:
        reading = _aggregate([1000.0, 2000.0], duration_s=3.0)
        assert reading is not None
        assert reading.mean_power_mw == pytest.approx(1500.0)
        assert reading.energy_mj == pytest.approx(4500.0)  # mW * s = mJ
        assert reading.samples == 2
        assert reading.duration_s == 3.0

    def test_aggregate_empty_is_none(self) -> None:
        assert _aggregate([], duration_s=1.0) is None
        assert _aggregate([100.0], duration_s=0.0) is None


class FakeProcess:
    """Stands in for the sudo powermetrics subprocess."""

    def __init__(self, *, exits_immediately: bool, stderr: str = "") -> None:
        self._exits_immediately = exits_immediately
        self.stderr = io.BytesIO(stderr.encode())
        self.terminated = False

    def poll(self) -> int | None:
        return 1 if self._exits_immediately else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0


def make_sampler(
    monkeypatch: pytest.MonkeyPatch,
    process: FakeProcess,
    tmp_path: Path,
    stdout_text: str = "",
) -> PowerMetricsSampler:
    """Wire a fake process in, with the captured stdout already on disk."""

    def fake_launch(self: PowerMetricsSampler) -> FakeProcess:
        stdout_file = tmp_path / "powermetrics-stdout.txt"
        stdout_file.write_text(stdout_text)
        self._stdout_path = stdout_file
        return process

    monkeypatch.setattr(PowerMetricsSampler, "_launch", fake_launch)
    monkeypatch.setattr("silicon_eval.profiling.energy.time.sleep", lambda s: None)
    return PowerMetricsSampler()


class TestPowerMetricsSampler:
    def test_no_passwordless_sudo_degrades_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = FakeProcess(exits_immediately=True, stderr="sudo: a password is required")
        with make_sampler(monkeypatch, process, tmp_path) as sampler:
            assert not sampler.available
            assert sampler.unavailable_reason is not None
            assert "passwordless sudo" in sampler.unavailable_reason
            assert "a password is required" in sampler.unavailable_reason
        assert sampler.reading() is None

    def test_launch_failure_degrades_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_launch(self: PowerMetricsSampler) -> subprocess.Popen[bytes]:
            raise FileNotFoundError("no such file: sudo")

        monkeypatch.setattr(PowerMetricsSampler, "_launch", broken_launch)
        with PowerMetricsSampler() as sampler:
            assert not sampler.available
            assert sampler.unavailable_reason is not None
            assert "could not launch" in sampler.unavailable_reason

    def test_available_path_collects_reading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = FakeProcess(exits_immediately=False)
        with make_sampler(monkeypatch, process, tmp_path, stdout_text=SAMPLE_OUTPUT) as sampler:
            assert sampler.available
        assert process.terminated
        reading = sampler.reading()
        assert reading is not None
        assert reading.samples == 3
        assert reading.mean_power_mw == pytest.approx((570 + 3000 + 1430.5) / 3)
        assert reading.duration_s > 0

    def test_stdout_temp_file_removed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = FakeProcess(exits_immediately=False)
        with make_sampler(monkeypatch, process, tmp_path, stdout_text=SAMPLE_OUTPUT):
            pass
        assert not (tmp_path / "powermetrics-stdout.txt").exists()

    def test_root_owned_process_terminated_via_sudo_kill(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # powermetrics runs as root: unprivileged SIGTERM raises PermissionError
        # on exactly the machines where sampling works.
        class RootProcess(FakeProcess):
            pid = 4242

            def terminate(self) -> None:
                raise PermissionError("Operation not permitted")

        sudo_kills: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> None:
            sudo_kills.append(cmd)

        monkeypatch.setattr("silicon_eval.profiling.energy.subprocess.run", fake_run)
        process = RootProcess(exits_immediately=False)
        with make_sampler(monkeypatch, process, tmp_path, stdout_text=SAMPLE_OUTPUT) as sampler:
            assert sampler.available
        assert sudo_kills == [["sudo", "-n", "kill", "-TERM", "4242"]]
        reading = sampler.reading()
        assert reading is not None  # session still parsed after escalated stop
        assert reading.samples == 3

    def test_all_kill_paths_denied_never_hangs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # sudoers rule covers powermetrics but not kill: direct signals raise
        # PermissionError AND sudo -n kill is denied, so waits keep expiring.
        class UnkillableProcess(FakeProcess):
            pid = 4242

            def terminate(self) -> None:
                raise PermissionError("Operation not permitted")

            def kill(self) -> None:
                raise PermissionError("Operation not permitted")

            def wait(self, timeout: float | None = None) -> int:
                raise subprocess.TimeoutExpired(cmd="powermetrics", timeout=timeout or 0)

        monkeypatch.setattr("silicon_eval.profiling.energy.subprocess.run", lambda *a, **kw: None)
        process = UnkillableProcess(exits_immediately=False)
        with make_sampler(monkeypatch, process, tmp_path, stdout_text=SAMPLE_OUTPUT) as sampler:
            assert sampler.available
        # __exit__ returned instead of blocking forever; samples still parsed.
        reading = sampler.reading()
        assert reading is not None
        assert reading.samples == 3

    def test_single_use(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        process = FakeProcess(exits_immediately=True)
        sampler = make_sampler(monkeypatch, process, tmp_path)
        with sampler:
            pass
        with pytest.raises(InvalidStateError, match="single-use"), sampler:
            pass
