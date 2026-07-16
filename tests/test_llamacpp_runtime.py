"""Tests for LlamaCppRuntime with llama_cpp mocked by a numpy-backed fake.

FakeLlama mirrors the real 0.3.x surface closely: ``scores`` is a persistent
preallocated (n_ctx, vocab) buffer filled with garbage, ``eval`` writes rows
at ``n_tokens`` and advances it, and ``reset`` only rewinds ``n_tokens`` —
so a missing reset or a wrong slice corrupts the NLL math in these tests the
same way it would against real llama.cpp.
"""

from __future__ import annotations

import math
import typing
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from silicon_eval.exceptions import (
    InvalidStateError,
    ModelLoadError,
    RuntimeUnavailableError,
)
from silicon_eval.runtimes import llamacpp_runtime
from silicon_eval.runtimes.base import ModelSpec, Quantization, Runtime
from silicon_eval.runtimes.llamacpp_runtime import LlamaCppRuntime, resolve_gguf_file

VOCAB = 7
SPEC = ModelSpec(model_id="Qwen/TestModel-GGUF", quantization=Quantization.Q4)

GGUF_FILES = [
    "README.md",
    "testmodel-q4_k_m.gguf",
    "testmodel-q4_0.gguf",
    "testmodel-q8_0.gguf",
    "testmodel-f16.gguf",
]


class FakeLlama:
    """llama_cpp.Llama stand-in faithful to its buffer and stream semantics."""

    def __init__(self, token_map: dict[str, list[int]], oracle: bool = False) -> None:
        self._token_map = token_map
        self._oracle = oracle
        self.n_tokens = 0
        # Real Llama preallocates an uninitialized (n_ctx, vocab) buffer;
        # garbage rows make wrong slices and missing resets loudly wrong.
        self.scores = np.full((64, VOCAB), 1e30, dtype=np.float32)
        self.reset_calls = 0
        self.closed = False
        self.eval_lengths: list[int] = []
        self.tokenize_calls: list[dict[str, bool]] = []
        self.completion_requests: list[dict[str, Any]] = []

    def tokenize(self, data: bytes, add_bos: bool = True, special: bool = True) -> list[int]:
        self.tokenize_calls.append({"add_bos": add_bos, "special": special})
        return list(self._token_map[data.decode("utf-8")])

    def reset(self) -> None:
        self.reset_calls += 1
        self.n_tokens = 0  # rows persist, exactly like the real buffer

    def eval(self, tokens: list[int]) -> None:
        self.eval_lengths.append(len(tokens))
        for offset, token in enumerate(tokens):
            row = np.zeros(VOCAB, dtype=np.float32)
            if self._oracle:
                row[(token + 1) % VOCAB] = 25.0
            self.scores[self.n_tokens + offset] = row
        self.n_tokens += len(tokens)

    def create_completion(self, prompt: str, **kwargs: Any) -> Any:
        self.completion_requests.append({"prompt": prompt, **kwargs})
        for text in ["Hel", "lo", "!"]:
            yield {"choices": [{"text": text, "finish_reason": None}]}
        # Real llama-cpp-python ends the stream with a tokenless finish chunk.
        yield {"choices": [{"text": "", "finish_reason": "length"}]}

    def close(self) -> None:
        self.closed = True


def make_runtime(
    monkeypatch: pytest.MonkeyPatch,
    token_map: dict[str, list[int]] | None = None,
    oracle: bool = False,
    n_ctx: int = 1024,
) -> tuple[LlamaCppRuntime, FakeLlama]:
    fake = FakeLlama(token_map or {}, oracle=oracle)
    fake_module = SimpleNamespace(Llama=lambda **kwargs: fake)
    monkeypatch.setattr(llamacpp_runtime, "_import_llama_cpp", lambda: fake_module)
    monkeypatch.setattr(LlamaCppRuntime, "_download_gguf", lambda self, spec: "/fake/model.gguf")
    runtime = LlamaCppRuntime(n_ctx=n_ctx)
    runtime.load(SPEC)
    return runtime, fake


class TestResolveGgufFile:
    def test_picks_preferred_pattern(self) -> None:
        assert resolve_gguf_file(SPEC, GGUF_FILES) == "testmodel-q4_k_m.gguf"

    def test_falls_back_to_secondary_pattern(self) -> None:
        files = ["testmodel-q4_0.gguf", "testmodel-q8_0.gguf"]
        assert resolve_gguf_file(SPEC, files) == "testmodel-q4_0.gguf"

    def test_q8_and_fp16_patterns(self) -> None:
        q8 = ModelSpec(model_id="m", quantization=Quantization.Q8)
        fp16 = ModelSpec(model_id="m", quantization=Quantization.FP16)
        assert resolve_gguf_file(q8, GGUF_FILES) == "testmodel-q8_0.gguf"
        assert resolve_gguf_file(fp16, GGUF_FILES) == "testmodel-f16.gguf"

    def test_f16_never_matches_bf16(self) -> None:
        both = ["model-bf16.gguf", "model-f16.gguf"]
        fp16 = ModelSpec(model_id="m", quantization=Quantization.FP16)
        bf16 = ModelSpec(model_id="m", quantization=Quantization.BF16)
        assert resolve_gguf_file(fp16, both) == "model-f16.gguf"
        assert resolve_gguf_file(bf16, both) == "model-bf16.gguf"
        with pytest.raises(ModelLoadError, match="no GGUF matching fp16"):
            resolve_gguf_file(fp16, ["model-bf16.gguf"])

    def test_file_override_wins(self) -> None:
        spec = ModelSpec(model_id="m", quantization=Quantization.Q4, file_override="custom.gguf")
        assert resolve_gguf_file(spec, []) == "custom.gguf"

    def test_empty_repo_raises_with_guidance(self) -> None:
        with pytest.raises(ModelLoadError, match=r"among 0 \.gguf files"):
            resolve_gguf_file(SPEC, [])

    def test_no_match_raises_with_guidance(self) -> None:
        spec = ModelSpec(model_id="m", quantization=Quantization.BF16)
        with pytest.raises(ModelLoadError, match="file_override"):
            resolve_gguf_file(spec, GGUF_FILES)

    def test_multipart_gguf_rejected(self) -> None:
        files = ["big-q4_k_m-00001-of-00002.gguf", "big-q4_k_m-00002-of-00002.gguf"]
        with pytest.raises(ModelLoadError, match="multi-part"):
            resolve_gguf_file(SPEC, files)


class TestGenerate:
    TOKEN_MAP: typing.ClassVar[dict[str, list[int]]] = {"hi": [1, 2, 3], "Hello!": [4, 5]}

    def test_streams_resets_and_counts_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, fake = make_runtime(monkeypatch, token_map=self.TOKEN_MAP)
        result = runtime.generate("hi", max_tokens=8)

        assert result.text == "Hello!"
        assert result.metrics.prompt_tokens == 3
        # Token count comes from retokenizing the output, not chunk counting.
        assert result.metrics.generation_tokens == 2
        assert result.metrics.time_to_first_token_s > 0
        assert result.metrics.generation_tps > 0
        assert result.metrics.peak_memory_bytes is None  # llama.cpp reports none
        assert fake.reset_calls == 1  # KV cache cleared: no prefix-cache TTFT
        request = fake.completion_requests[0]
        assert request["max_tokens"] == 8
        assert request["temperature"] == 0.0
        assert request["stream"] is True

    def test_prompt_counted_with_model_native_specials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime, fake = make_runtime(monkeypatch, token_map=self.TOKEN_MAP)
        runtime.generate("hi", max_tokens=8)
        # First tokenize call counts the prompt as create_completion evaluates
        # it (special=True); the output recount is a raw scoring tokenize.
        assert fake.tokenize_calls[0]["special"] is True
        assert fake.tokenize_calls[-1] == {"add_bos": False, "special": False}

    def test_clamped_to_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, fake = make_runtime(monkeypatch, token_map=self.TOKEN_MAP, n_ctx=5)
        runtime.generate("hi", max_tokens=64)
        assert fake.completion_requests[0]["max_tokens"] == 2  # 5 - 3 prompt tokens

    def test_nonpositive_max_tokens_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # llama.cpp treats max_tokens <= 0 as "fill the context" — must not pass through.
        runtime, _ = make_runtime(monkeypatch, token_map=self.TOKEN_MAP)
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            runtime.generate("hi", max_tokens=0)

    def test_prompt_filling_context_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(monkeypatch, token_map=self.TOKEN_MAP, n_ctx=3)
        with pytest.raises(ValueError, match="fills the whole context"):
            runtime.generate("hi", max_tokens=8)


class TestScoring:
    def test_score_uniform_logits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tokens = [i % VOCAB for i in range(11)]
        runtime, fake = make_runtime(monkeypatch, token_map={"text": tokens})
        result = runtime.score("text", max_context_tokens=5)

        assert result.scored_tokens == 10
        assert result.windows == 2
        assert result.negative_log_likelihood == pytest.approx(10 * math.log(VOCAB), rel=1e-5)
        assert fake.reset_calls == 2  # context cleared between windows

    def test_score_alignment_via_oracle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tokens = [i % VOCAB for i in range(9)]
        runtime, _ = make_runtime(monkeypatch, token_map={"text": tokens}, oracle=True)
        result = runtime.score("text", max_context_tokens=4)
        assert result.negative_log_likelihood / result.scored_tokens < 0.01

    def test_scoring_tokenizes_raw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tokens = [1, 2, 3]
        runtime, fake = make_runtime(monkeypatch, token_map={"text": tokens})
        runtime.score("text")
        assert fake.tokenize_calls[0] == {"add_bos": False, "special": False}

    def test_score_window_clamp_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tokens = [i % VOCAB for i in range(11)]
        runtime, _ = make_runtime(monkeypatch, token_map={"text": tokens}, n_ctx=6)
        with pytest.warns(UserWarning, match="clamped from 512 to 5"):
            result = runtime.score("text", max_context_tokens=512)
        assert result.windows == 2

    def test_completion_scores_only_continuation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(
            monkeypatch, token_map={"ctx": [1, 2, 3], "ctxcont": [1, 2, 3, 4, 5]}
        )
        result = runtime.score_completion("ctx", "cont")
        assert result.scored_tokens == 2
        assert result.windows == 1
        assert result.negative_log_likelihood == pytest.approx(2 * math.log(VOCAB), rel=1e-5)

    def test_completion_scores_the_right_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Context breaks the oracle's consecutive pattern; the continuation
        # follows it. Scoring the wrong span (context instead of continuation)
        # yields ~25 nats/token instead of ~0 — this kills the slice mutant.
        runtime, _ = make_runtime(
            monkeypatch,
            token_map={"ctx": [5, 0, 3], "ctxcont": [5, 0, 3, 4, 5]},
            oracle=True,
        )
        result = runtime.score_completion("ctx", "cont")
        assert result.scored_tokens == 2
        assert result.negative_log_likelihood / result.scored_tokens < 0.01

    def test_completion_long_context_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, fake = make_runtime(
            monkeypatch,
            token_map={"ctx": [1, 2, 3, 4, 5, 6], "ctxcont": [1, 2, 3, 4, 5, 6, 0, 1]},
        )
        result = runtime.score_completion("ctx", "cont", max_context_tokens=4)
        assert result.scored_tokens == 2  # continuation fully scored
        assert fake.eval_lengths == [4]  # window truncated to the limit
        assert result.negative_log_likelihood == pytest.approx(2 * math.log(VOCAB), rel=1e-5)

    def test_completion_oversized_continuation_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime, _ = make_runtime(monkeypatch, token_map={"ctx": [1], "ctxcont": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="does not fit"):
            runtime.score_completion("ctx", "cont", max_context_tokens=4)

    def test_completion_empty_context_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(monkeypatch, token_map={"": [], "cont": [4, 5]})
        with pytest.raises(ValueError, match="context must contribute"):
            runtime.score_completion("", "cont")

    def test_empty_continuation_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(monkeypatch, token_map={"ctx": [1, 2, 3], "ctxcont": [1, 2, 3]})
        with pytest.raises(ValueError, match="adds no tokens"):
            runtime.score_completion("ctx", "cont")


class TestLifecycle:
    def test_satisfies_runtime_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(monkeypatch)
        assert isinstance(runtime, Runtime)

    def test_unload_closes_model_and_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, fake = make_runtime(monkeypatch)
        runtime.unload()
        runtime.unload()  # second call must be a no-op, not an error
        assert fake.closed
        with pytest.raises(InvalidStateError, match="no model loaded"):
            runtime.generate("hi")

    def test_methods_require_load(self) -> None:
        runtime = LlamaCppRuntime()
        with pytest.raises(InvalidStateError):
            runtime.generate("hi")
        with pytest.raises(InvalidStateError):
            runtime.score("text")
        with pytest.raises(InvalidStateError):
            runtime.score_completion("a", "b")

    def test_missing_llama_cpp_raises_runtime_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        with pytest.raises(RuntimeUnavailableError, match="llama-cpp-python is not installed"):
            LlamaCppRuntime().load(SPEC)


def test_registry_exposes_llamacpp() -> None:
    from silicon_eval.runtimes import available_runtimes, backend_versions, get_runtime

    assert "llama.cpp" in available_runtimes()
    assert isinstance(get_runtime("llama.cpp"), LlamaCppRuntime)
    assert set(backend_versions("llama.cpp")) == {"llama-cpp-python"}
