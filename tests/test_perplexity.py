"""Tests for the perplexity evaluator (runtime mocked)."""

from __future__ import annotations

import math

import pytest

from silicon_eval.evals.base import Evaluator
from silicon_eval.evals.perplexity import PerplexityConfig, PerplexityEvaluator
from silicon_eval.runtimes.base import ScoreResult
from tests.conftest import FakeRuntime


def make_evaluator(config: PerplexityConfig | None = None) -> PerplexityEvaluator:
    return PerplexityEvaluator(config, text_loader=lambda: "corpus text")


def test_satisfies_evaluator_protocol() -> None:
    assert isinstance(make_evaluator(), Evaluator)


def test_perplexity_is_exp_of_mean_nll(fake_runtime: FakeRuntime) -> None:
    fake_runtime.score_result = ScoreResult(
        negative_log_likelihood=20.0, scored_tokens=10, windows=2
    )
    result = make_evaluator().run(fake_runtime)

    assert result.name == "perplexity:wikitext2"
    assert result.metrics["perplexity"] == pytest.approx(math.exp(2.0))
    assert result.metrics["scored_tokens"] == 10
    assert result.metrics["windows"] == 2
    assert result.duration_s >= 0


def test_config_is_forwarded_to_runtime(fake_runtime: FakeRuntime) -> None:
    config = PerplexityConfig(max_context_tokens=128, max_windows=7)
    make_evaluator(config).run(fake_runtime)
    assert fake_runtime.scored_texts == ["corpus text"]
    assert fake_runtime.score_args == [(128, 7)]


def test_default_config_caps_windows(fake_runtime: FakeRuntime) -> None:
    make_evaluator().run(fake_runtime)
    assert fake_runtime.score_args == [(512, 50)]


def test_corpus_loaded_once_across_sweep(fake_runtime: FakeRuntime) -> None:
    loads = 0

    def counting_loader() -> str:
        nonlocal loads
        loads += 1
        return "corpus text"

    evaluator = PerplexityEvaluator(text_loader=counting_loader)
    evaluator.run(fake_runtime)
    evaluator.run(fake_runtime)  # second variant in a quantization sweep
    assert loads == 1
