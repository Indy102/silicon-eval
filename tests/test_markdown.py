"""Tests for the Markdown report renderer."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from silicon_eval.evals.base import EvalResult
from silicon_eval.report.markdown import render_markdown, write_report_markdown
from silicon_eval.report.schema import EnergyProfile, VariantResult
from silicon_eval.runner import build_report, run_variant
from silicon_eval.runtimes.base import ModelSpec, Quantization, Runtime
from tests.conftest import FakeRuntime


class CannedEvaluator:
    def __init__(self, name: str, metrics: dict[str, float | int]) -> None:
        self.name = name
        self._metrics = metrics

    def run(self, runtime: Runtime) -> EvalResult:
        return EvalResult(name=self.name, metrics=self._metrics, duration_s=0.1)


def make_variants(fake_runtime: FakeRuntime) -> list[VariantResult]:
    evaluator = [
        CannedEvaluator(
            "perplexity:wikitext2", {"perplexity": 26.16, "scored_tokens": 5120, "windows": 10}
        ),
        CannedEvaluator("hellaswag", {"accuracy_norm": 0.312, "accuracy": 0.29, "items": 100}),
    ]
    q4 = run_variant(
        fake_runtime,
        ModelSpec(model_id="some/model", quantization=Quantization.Q4),
        prompt="p",
        runs=1,
        warmup=0,
        evaluators=evaluator,
    )
    q8 = run_variant(
        fake_runtime,
        ModelSpec(model_id="some/model", quantization=Quantization.Q8),
        prompt="p",
        runs=1,
        warmup=0,
        evaluators=evaluator,
    )
    q4 = dataclasses.replace(
        q4,
        energy=EnergyProfile(
            mean_power_mw=2500.0,
            energy_per_generated_token_mj=39.1,
            generated_tokens=128,
            samples=10,
            duration_s=2.0,
        ),
    )
    q8 = dataclasses.replace(
        q8, energy_unavailable_reason="powermetrics requires passwordless sudo"
    )
    return [q4, q8]


def test_renders_comparison_table(fake_runtime: FakeRuntime) -> None:
    report = build_report(make_variants(fake_runtime))
    text = render_markdown(report)

    assert "# silicon-eval report" in text
    assert "**Model:** some/model" in text
    assert "| quant" in text
    assert "26.16" in text
    assert "0.312" in text
    assert "39.1" in text  # energy for q4
    assert "n/a" in text  # energy for q8
    assert "42.5" in text  # gen tok/s from CANNED_METRICS


def test_notes_cover_methodology_and_energy(fake_runtime: FakeRuntime) -> None:
    report = build_report(make_variants(fake_runtime))
    text = render_markdown(report)

    assert "## Notes" in text
    assert "first 5120 tokens of WikiText-2" in text
    assert "first 100 validation items" in text
    assert "Energy sampling unavailable: powermetrics requires passwordless sudo." in text
    assert "system-wide CPU+GPU+ANE" in text


def test_empty_report_renders(fake_runtime: FakeRuntime) -> None:
    text = render_markdown(build_report([]))
    assert "# silicon-eval report" in text


def test_write_report_markdown(fake_runtime: FakeRuntime, tmp_path: Path) -> None:
    report = build_report(make_variants(fake_runtime))
    out = tmp_path / "report.md"
    write_report_markdown(report, out)
    assert out.read_text().startswith("# silicon-eval report")
