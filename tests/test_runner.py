"""Tests for the variant runner orchestration."""

from __future__ import annotations

import pytest

from silicon_eval import __version__
from silicon_eval.evals.base import EvalResult
from silicon_eval.profiling.energy import EnergyReading
from silicon_eval.runner import build_report, run_variant
from silicon_eval.runtimes.base import ModelSpec, Quantization, Runtime
from tests.conftest import FakeRuntime, UnavailableEnergySampler

SPEC = ModelSpec(model_id="some/model", quantization=Quantization.Q8)


class FakeEvaluator:
    name: str = "fake-eval"

    def run(self, runtime: Runtime) -> EvalResult:
        return EvalResult(name=self.name, metrics={"score": 1.0}, duration_s=0.1)


class ExplodingEvaluator:
    name: str = "exploding-eval"

    def run(self, runtime: Runtime) -> EvalResult:
        raise ValueError("boom")


def test_run_variant_assembles_result(fake_runtime: FakeRuntime) -> None:
    result = run_variant(
        fake_runtime,
        SPEC,
        prompt="p",
        max_tokens=8,
        runs=2,
        warmup=1,
        evaluators=[FakeEvaluator()],
    )

    assert fake_runtime.loaded_specs == [SPEC]
    assert fake_runtime.unload_count == 1
    assert len(fake_runtime.prompts) == 3  # 1 warmup + 2 measured
    assert result.model_id == "some/model"
    assert result.quantization is Quantization.Q8
    assert result.runtime == "fake"
    assert result.generation.runs == 2
    assert [e.name for e in result.evals] == ["fake-eval"]
    assert result.peak_rss_bytes is not None
    assert result.peak_rss_bytes > 0
    assert result.backend_versions == {}  # "fake" runtime has no registered dists


def test_backend_versions_registry() -> None:
    from silicon_eval.runtimes import backend_versions

    mlx_versions = backend_versions("mlx")
    assert set(mlx_versions) == {"mlx", "mlx-lm"}
    assert all(isinstance(v, str) and v for v in mlx_versions.values())
    assert backend_versions("unknown-runtime") == {}


def test_run_variant_unloads_when_eval_fails(fake_runtime: FakeRuntime) -> None:
    with pytest.raises(ValueError, match="boom"):
        run_variant(fake_runtime, SPEC, prompt="p", evaluators=[ExplodingEvaluator()])
    assert fake_runtime.unload_count == 1


class AvailableEnergySampler:
    """Fake sampler that pretends powermetrics collected a clean session."""

    available = True
    unavailable_reason = None

    def __enter__(self) -> AvailableEnergySampler:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def reading(self) -> EnergyReading:
        return EnergyReading(mean_power_mw=1000.0, energy_mj=6400.0, samples=5, duration_s=6.4)


class TestEnergyMeasurement:
    def test_disabled_by_default(self, fake_runtime: FakeRuntime) -> None:
        result = run_variant(fake_runtime, SPEC, prompt="p", runs=1, warmup=0)
        assert result.energy is None
        assert result.energy_unavailable_reason is None

    def test_unavailable_reason_recorded(
        self, fake_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("silicon_eval.runner.PowerMetricsSampler", UnavailableEnergySampler)
        result = run_variant(fake_runtime, SPEC, prompt="p", runs=1, warmup=0, measure_energy=True)
        assert result.energy is None
        assert result.energy_unavailable_reason is not None
        assert "sudo" in result.energy_unavailable_reason

    def test_profile_computed_over_dedicated_runs(
        self, fake_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("silicon_eval.runner.PowerMetricsSampler", AvailableEnergySampler)
        result = run_variant(fake_runtime, SPEC, prompt="p", runs=2, warmup=0, measure_energy=True)
        assert len(fake_runtime.prompts) == 4  # 2 profiling + 2 energy runs
        assert result.energy is not None
        assert result.energy.generated_tokens == 128  # 2 runs x 64 canned tokens
        assert result.energy.energy_per_generated_token_mj == pytest.approx(6400.0 / 128)
        assert result.energy.mean_power_mw == 1000.0
        assert result.energy_unavailable_reason is None


def test_build_report_stamps_metadata(fake_runtime: FakeRuntime) -> None:
    variant = run_variant(fake_runtime, SPEC, prompt="p", runs=1, warmup=0)
    report = build_report([variant])

    assert report.schema_version == 2
    assert report.silicon_eval_version == __version__
    assert report.created_at.endswith("+00:00")
    assert report.machine.memory_bytes > 0
    assert report.variants == [variant]
