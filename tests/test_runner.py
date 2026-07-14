"""Tests for the variant runner orchestration."""

from __future__ import annotations

import pytest

from silicon_eval import __version__
from silicon_eval.evals.base import EvalResult
from silicon_eval.runner import build_report, run_variant
from silicon_eval.runtimes.base import ModelSpec, Quantization, Runtime
from tests.conftest import FakeRuntime

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


def test_run_variant_unloads_when_eval_fails(fake_runtime: FakeRuntime) -> None:
    with pytest.raises(ValueError, match="boom"):
        run_variant(fake_runtime, SPEC, prompt="p", evaluators=[ExplodingEvaluator()])
    assert fake_runtime.unload_count == 1


def test_build_report_stamps_metadata(fake_runtime: FakeRuntime) -> None:
    variant = run_variant(fake_runtime, SPEC, prompt="p", runs=1, warmup=0)
    report = build_report([variant])

    assert report.schema_version == 1
    assert report.silicon_eval_version == __version__
    assert report.created_at.endswith("+00:00")
    assert report.machine.memory_bytes > 0
    assert report.variants == [variant]
