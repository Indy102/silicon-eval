"""Integration tests against a real tiny model.

Apple Silicon only; excluded from the default run and from CI.
Run locally with: pytest -m slow --no-cov
"""

from __future__ import annotations

import math

import pytest

from silicon_eval.evals.perplexity import PerplexityConfig, PerplexityEvaluator
from silicon_eval.runner import build_report, run_variant
from silicon_eval.runtimes.base import ModelSpec, Quantization
from silicon_eval.runtimes.mlx_runtime import MLXRuntime

pytestmark = pytest.mark.slow

TINY_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
TINY_SPEC = ModelSpec(model_id=TINY_MODEL, quantization=Quantization.Q4)

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. Language models assign "
    "probabilities to sequences of tokens, and perplexity summarizes how "
    "well a model predicts a corpus: lower is better."
)


@pytest.fixture(scope="module")
def loaded_runtime() -> MLXRuntime:
    pytest.importorskip("mlx_lm")
    runtime = MLXRuntime()
    runtime.load(TINY_SPEC)
    return runtime


def test_real_generation_produces_text_and_metrics(loaded_runtime: MLXRuntime) -> None:
    result = loaded_runtime.generate("The capital of France is", max_tokens=16)

    assert result.text.strip()
    assert result.metrics.prompt_tokens > 0
    assert 0 < result.metrics.generation_tokens <= 16
    assert result.metrics.time_to_first_token_s > 0
    assert result.metrics.generation_tps > 0
    assert result.metrics.peak_memory_bytes is not None
    assert result.metrics.peak_memory_bytes > 0


def test_real_score_gives_sane_perplexity(loaded_runtime: MLXRuntime) -> None:
    score = loaded_runtime.score(SAMPLE_TEXT, max_context_tokens=64)

    assert score.scored_tokens > 20
    assert score.windows >= 1
    perplexity = math.exp(score.negative_log_likelihood / score.scored_tokens)
    assert 1.0 < perplexity < 1000.0  # coherent English on a real LM


def test_real_score_completion_prefers_true_continuation(
    loaded_runtime: MLXRuntime,
) -> None:
    context = "The capital of France is"
    paris = loaded_runtime.score_completion(context, " Paris")
    zebra = loaded_runtime.score_completion(context, " zebra")

    assert paris.scored_tokens >= 1
    assert zebra.scored_tokens >= 1
    paris_nll = paris.negative_log_likelihood / paris.scored_tokens
    zebra_nll = zebra.negative_log_likelihood / zebra.scored_tokens
    assert paris_nll < zebra_nll  # a real LM must prefer " Paris" here


def test_real_hellaswag_subset(loaded_runtime: MLXRuntime) -> None:
    from silicon_eval.evals.hellaswag import HellaSwagConfig, HellaSwagEvaluator

    evaluator = HellaSwagEvaluator(HellaSwagConfig(max_items=4))
    result = evaluator.run(loaded_runtime)

    assert result.metrics["items"] == 4
    assert 0.0 <= float(result.metrics["accuracy"]) <= 1.0
    assert 0.0 <= float(result.metrics["accuracy_norm"]) <= 1.0


def test_real_pipeline_end_to_end() -> None:
    pytest.importorskip("mlx_lm")
    from silicon_eval.report.json_io import variant_from_dict, variant_to_dict
    from silicon_eval.report.markdown import render_markdown

    evaluator = PerplexityEvaluator(
        PerplexityConfig(max_context_tokens=64, max_windows=2),
        text_loader=lambda: SAMPLE_TEXT,
    )
    variant = run_variant(
        MLXRuntime(),
        TINY_SPEC,
        prompt="Count to five:",
        max_tokens=8,
        runs=1,
        warmup=1,
        evaluators=[evaluator],
        measure_energy=True,
    )
    report = build_report([variant])

    assert report.machine.chip is not None  # we are on Apple Silicon here
    assert variant.generation.generation_tps.mean > 0
    assert variant.peak_rss_bytes is not None
    assert variant.peak_rss_bytes > 0
    ppl = variant.evals[0].metrics["perplexity"]
    assert isinstance(ppl, float)
    assert 1.0 < ppl < 1000.0
    # Energy either measured or degraded with a reason — never silently absent.
    assert (variant.energy is None) == (variant.energy_unavailable_reason is not None)

    markdown = render_markdown(report)
    assert "| 4bit" in markdown
    assert variant_from_dict(variant_to_dict(variant)) == variant  # cache round trip
