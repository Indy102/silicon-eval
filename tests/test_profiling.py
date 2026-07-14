"""Tests for generation profiling, stats, and the RSS sampler."""

from __future__ import annotations

import dataclasses
import time

import pytest

from silicon_eval.exceptions import InvalidStateError
from silicon_eval.profiling.generation import profile_generation
from silicon_eval.profiling.memory import RssSampler
from silicon_eval.profiling.stats import Stat
from silicon_eval.runtimes.base import GenerationMetrics, GenerationResult
from tests.conftest import CANNED_METRICS, FakeRuntime


class SequenceRuntime(FakeRuntime):
    """Returns a scripted sequence of metrics, one per generate() call."""

    def __init__(self, metrics: list[GenerationMetrics]) -> None:
        super().__init__()
        self._metrics = iter(metrics)

    def generate(self, prompt: str, *, max_tokens: int = 128) -> GenerationResult:
        self.prompts.append(prompt)
        return GenerationResult(text="x", metrics=next(self._metrics))


def metrics_with(ttft: float, peak: int | None) -> GenerationMetrics:
    return dataclasses.replace(CANNED_METRICS, time_to_first_token_s=ttft, peak_memory_bytes=peak)


class TestStat:
    def test_from_samples(self) -> None:
        stat = Stat.from_samples([0.1, 0.2, 0.3])
        assert stat.mean == pytest.approx(0.2)
        assert stat.min == 0.1
        assert stat.max == 0.3

    def test_empty_samples_raise(self) -> None:
        with pytest.raises(ValueError, match="no samples"):
            Stat.from_samples([])


class TestProfileGeneration:
    def test_warmup_excluded_from_stats(self) -> None:
        runtime = SequenceRuntime(
            [
                metrics_with(ttft=9.9, peak=None),  # warmup — must not count
                metrics_with(ttft=0.1, peak=100),
                metrics_with(ttft=0.2, peak=300),
                metrics_with(ttft=0.3, peak=None),
            ]
        )
        profile = profile_generation(runtime, "p", max_tokens=8, runs=3, warmup=1)

        assert len(runtime.prompts) == 4
        assert profile.ttft_s.mean == pytest.approx(0.2)
        assert profile.ttft_s.max == pytest.approx(0.3)
        assert profile.peak_metal_bytes == 300
        assert profile.runs == 3
        assert profile.warmup_runs == 1

    def test_all_peaks_missing_gives_none(self) -> None:
        runtime = SequenceRuntime([metrics_with(ttft=0.1, peak=None)])
        profile = profile_generation(runtime, "p", runs=1, warmup=0)
        assert profile.peak_metal_bytes is None

    def test_invalid_runs_raise(self, fake_runtime: FakeRuntime) -> None:
        with pytest.raises(ValueError, match="runs"):
            profile_generation(fake_runtime, "p", runs=0)
        with pytest.raises(ValueError, match="warmup"):
            profile_generation(fake_runtime, "p", warmup=-1)


class TestRssSampler:
    def test_captures_positive_peak(self) -> None:
        with RssSampler(interval_s=0.01) as sampler:
            _ = [b"x" * 1024 for _ in range(100)]
        assert sampler.peak_rss_bytes > 0

    def test_background_loop_actually_samples(self) -> None:
        with RssSampler(interval_s=0.005) as sampler:
            time.sleep(0.1)
        # enter + exit account for 2; anything above proves the thread ran
        assert sampler.samples_taken > 2

    def test_single_use(self) -> None:
        sampler = RssSampler()
        with sampler:
            pass
        with pytest.raises(InvalidStateError, match="single-use"), sampler:
            pass
