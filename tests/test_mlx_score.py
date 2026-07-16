"""Tests for MLXRuntime.score() against a numpy-backed fake mlx.core.

The fake implements exactly the array ops score() uses, so the windowing
and NLL math run for real — only the framework is substituted. With uniform
logits over a vocab of V, every token's logprob is -log(V), which gives
closed-form expected values.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from silicon_eval.exceptions import InvalidStateError
from silicon_eval.runtimes import mlx_runtime
from silicon_eval.runtimes.base import ModelSpec, Quantization
from silicon_eval.runtimes.mlx_runtime import MLXRuntime

VOCAB = 7
SPEC = ModelSpec(model_id="mlx-community/TestModel", quantization=Quantization.Q4)


def _logsumexp(x: Any, axis: int = -1, keepdims: bool = False) -> Any:
    peak = np.max(x, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(x - peak), axis=axis, keepdims=True)) + peak
    return out if keepdims else np.squeeze(out, axis=axis)


FAKE_MX = SimpleNamespace(
    array=np.asarray,
    float32=np.float32,
    logsumexp=_logsumexp,
    take_along_axis=np.take_along_axis,
    eval=lambda *_: None,
    reset_peak_memory=lambda: None,
    get_peak_memory=lambda: 0,
    clear_cache=lambda: None,
)


class UniformLogitsModel:
    """Returns all-zero logits: every token has probability 1/VOCAB."""

    def __call__(self, inputs: Any) -> Any:
        batch, length = inputs.shape
        return np.zeros((batch, length, VOCAB), dtype=np.float16)


class NextTokenOracleModel:
    """Puts ~all probability on ``(input_token + 1) % VOCAB`` at every position.

    Fed consecutive token ids, a correctly aligned scorer sees NLL ≈ 0; any
    input/target misalignment gathers a near-zero-probability entry and blows
    the NLL up — something uniform logits can never detect.
    """

    def __call__(self, inputs: Any) -> Any:
        batch, length = inputs.shape
        logits = np.zeros((batch, length, VOCAB), dtype=np.float16)
        for position in range(length):
            expected_next = (int(inputs[0, position]) + 1) % VOCAB
            logits[0, position, expected_next] = 25.0
        return logits


class FixedTokenizer:
    def __init__(self, tokens: list[int]) -> None:
        self._tokens = tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False  # scoring must not add BOS/specials
        return self._tokens


class MappingTokenizer:
    """Maps exact strings to token id lists — models tokenizer boundary effects."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self._mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False  # scoring must not add BOS/specials
        return self._mapping[text]


def make_runtime(
    monkeypatch: pytest.MonkeyPatch, tokens: list[int], model: Any = None
) -> MLXRuntime:
    loaded_model = model if model is not None else UniformLogitsModel()
    fake_mlx_lm = SimpleNamespace(load=lambda repo: (loaded_model, FixedTokenizer(tokens)))
    monkeypatch.setattr(mlx_runtime, "_import_mlx_lm", lambda: fake_mlx_lm)
    monkeypatch.setattr(mlx_runtime, "_import_mx", lambda: FAKE_MX)
    runtime = MLXRuntime()
    runtime.load(SPEC)
    return runtime


def test_uniform_logits_give_log_vocab_nll(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = make_runtime(monkeypatch, tokens=[i % VOCAB for i in range(11)])
    result = runtime.score("text", max_context_tokens=5)

    assert result.scored_tokens == 10  # every token but the first
    assert result.windows == 2
    assert result.negative_log_likelihood == pytest.approx(10 * math.log(VOCAB), rel=1e-5)


def test_single_window_when_context_covers_text(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = make_runtime(monkeypatch, tokens=[i % VOCAB for i in range(11)])
    result = runtime.score("text", max_context_tokens=1024)
    assert result.windows == 1
    assert result.scored_tokens == 10


def test_max_windows_caps_work(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = make_runtime(monkeypatch, tokens=[i % VOCAB for i in range(11)])
    result = runtime.score("text", max_context_tokens=5, max_windows=1)
    assert result.windows == 1
    assert result.scored_tokens == 5
    assert result.negative_log_likelihood == pytest.approx(5 * math.log(VOCAB), rel=1e-5)


def test_ragged_final_window_scores_remaining_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 12 tokens, window 5: windows of 5, 5, and 1 targets — nothing dropped.
    runtime = make_runtime(monkeypatch, tokens=[i % VOCAB for i in range(12)])
    result = runtime.score("text", max_context_tokens=5)
    assert result.windows == 3
    assert result.scored_tokens == 11
    assert result.negative_log_likelihood == pytest.approx(11 * math.log(VOCAB), rel=1e-5)


def test_input_target_alignment_via_oracle_model(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = [i % VOCAB for i in range(9)]  # consecutive ids: next = current + 1 (mod V)
    runtime = make_runtime(monkeypatch, tokens=tokens, model=NextTokenOracleModel())
    result = runtime.score("text", max_context_tokens=4)
    # Correct alignment scores the oracle's predicted token: NLL per token ≈ 0.
    # An off-by-one in inputs/targets would score near-zero-probability tokens.
    assert result.negative_log_likelihood / result.scored_tokens < 0.01


def test_zero_max_windows_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = make_runtime(monkeypatch, tokens=[1, 2, 3])
    with pytest.raises(ValueError, match="max_windows"):
        runtime.score("text", max_windows=0)


def make_completion_runtime(
    monkeypatch: pytest.MonkeyPatch, mapping: dict[str, list[int]], model: Any = None
) -> MLXRuntime:
    loaded_model = model if model is not None else UniformLogitsModel()
    fake_mlx_lm = SimpleNamespace(load=lambda repo: (loaded_model, MappingTokenizer(mapping)))
    monkeypatch.setattr(mlx_runtime, "_import_mlx_lm", lambda: fake_mlx_lm)
    monkeypatch.setattr(mlx_runtime, "_import_mx", lambda: FAKE_MX)
    runtime = MLXRuntime()
    runtime.load(SPEC)
    return runtime


class TestScoreCompletion:
    def test_scores_only_continuation_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = make_completion_runtime(
            monkeypatch, {"ctx": [1, 2, 3], "ctxcont": [1, 2, 3, 4, 5]}
        )
        result = runtime.score_completion("ctx", "cont")
        assert result.scored_tokens == 2
        assert result.windows == 1
        assert result.negative_log_likelihood == pytest.approx(2 * math.log(VOCAB), rel=1e-5)

    def test_alignment_via_oracle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Consecutive ids: the oracle puts ~all mass on (token + 1) % VOCAB, so a
        # correctly aligned conditional score of the last 2 tokens is ~0 NLL.
        runtime = make_completion_runtime(
            monkeypatch,
            {"ctx": [0, 1, 2, 3, 4], "ctxcont": [0, 1, 2, 3, 4, 5, 6]},
            model=NextTokenOracleModel(),
        )
        result = runtime.score_completion("ctx", "cont")
        assert result.scored_tokens == 2
        assert result.negative_log_likelihood / result.scored_tokens < 0.01

    def test_boundary_merge_shrinks_continuation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The tokenizer merges the boundary: full is only 1 token longer than ctx.
        runtime = make_completion_runtime(monkeypatch, {"ctx": [1, 2, 3], "ctxcont": [1, 2, 6, 5]})
        result = runtime.score_completion("ctx", "cont")
        assert result.scored_tokens == 1

    def test_long_context_left_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = make_completion_runtime(
            monkeypatch, {"ctx": [1, 2, 3, 4, 5, 6], "ctxcont": [1, 2, 3, 4, 5, 6, 0, 1]}
        )
        result = runtime.score_completion("ctx", "cont", max_context_tokens=4)
        assert result.scored_tokens == 2  # continuation fully scored despite truncation
        assert result.negative_log_likelihood == pytest.approx(2 * math.log(VOCAB), rel=1e-5)

    def test_empty_continuation_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = make_completion_runtime(monkeypatch, {"ctx": [1, 2, 3], "ctxcont": [1, 2, 3]})
        with pytest.raises(ValueError, match="adds no tokens"):
            runtime.score_completion("ctx", "cont")

    def test_empty_context_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Zero context tokens would silently drop the first continuation
        # token from the NLL while still counting it in scored_tokens.
        runtime = make_completion_runtime(monkeypatch, {"": [], "cont": [4, 5]})
        with pytest.raises(ValueError, match="context must contribute"):
            runtime.score_completion("", "cont")

    def test_scores_the_right_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Context breaks the oracle's consecutive pattern; only the
        # continuation follows it. A wrong-span slice ([:n] instead of [-n:])
        # scores ~25 nats/token here instead of ~0 — kills the slice mutant.
        runtime = make_completion_runtime(
            monkeypatch,
            {"ctx": [5, 0, 3], "ctxcont": [5, 0, 3, 4, 5]},
            model=NextTokenOracleModel(),
        )
        result = runtime.score_completion("ctx", "cont")
        assert result.scored_tokens == 2
        assert result.negative_log_likelihood / result.scored_tokens < 0.01

    def test_oversized_continuation_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = make_completion_runtime(monkeypatch, {"ctx": [1], "ctxcont": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="does not fit"):
            runtime.score_completion("ctx", "cont", max_context_tokens=4)

    def test_before_load_raises(self) -> None:
        with pytest.raises(InvalidStateError, match="no model loaded"):
            MLXRuntime().score_completion("ctx", "cont")


def test_score_before_load_raises() -> None:
    with pytest.raises(InvalidStateError, match="no model loaded"):
        MLXRuntime().score("text")


def test_too_short_text_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = make_runtime(monkeypatch, tokens=[3])
    with pytest.raises(ValueError, match="fewer than 2 tokens"):
        runtime.score("x")


def test_invalid_context_size_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = make_runtime(monkeypatch, tokens=[1, 2, 3])
    with pytest.raises(ValueError, match="max_context_tokens"):
        runtime.score("text", max_context_tokens=0)
