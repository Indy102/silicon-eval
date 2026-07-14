"""Tests for MLXRuntime with mlx-lm and mlx.core mocked out."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from silicon_eval.exceptions import (
    InvalidStateError,
    ModelLoadError,
    RuntimeUnavailableError,
)
from silicon_eval.runtimes import mlx_runtime
from silicon_eval.runtimes.base import ModelSpec, Quantization
from silicon_eval.runtimes.mlx_runtime import MLXRuntime, resolve_model_repo

SPEC = ModelSpec(model_id="mlx-community/TestModel", quantization=Quantization.Q4)
PEAK_BYTES = 512 * 1024 * 1024


@dataclass
class FakeResponse:
    text: str
    prompt_tokens: int = 12
    prompt_tps: float = 200.0
    generation_tokens: int = 3
    generation_tps: float = 40.0


class FakeMx:
    """Stands in for mlx.core; records the memory-API calls MLXRuntime makes."""

    def __init__(self, peak_bytes: int = PEAK_BYTES) -> None:
        self.peak_bytes = peak_bytes
        self.reset_calls = 0
        self.clear_cache_calls = 0

    def reset_peak_memory(self) -> None:
        self.reset_calls += 1

    def get_peak_memory(self) -> int:
        return self.peak_bytes

    def clear_cache(self) -> None:
        self.clear_cache_calls += 1


def make_fake_mlx_lm(responses: list[FakeResponse]) -> SimpleNamespace:
    def load(repo: str) -> tuple[str, str]:
        return (f"model:{repo}", f"tokenizer:{repo}")

    def stream_generate(model: Any, tokenizer: Any, *, prompt: str, max_tokens: int) -> Any:
        yield from responses

    return SimpleNamespace(load=load, stream_generate=stream_generate)


def make_runtime(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[FakeResponse] | None = None,
    fake_mx: FakeMx | None = None,
) -> tuple[MLXRuntime, FakeMx]:
    if responses is None:
        responses = [FakeResponse("Hello"), FakeResponse(" world")]
    mx = fake_mx if fake_mx is not None else FakeMx()
    fake = make_fake_mlx_lm(responses)
    monkeypatch.setattr(mlx_runtime, "_import_mlx_lm", lambda: fake)
    monkeypatch.setattr(mlx_runtime, "_import_mx", lambda: mx)
    return MLXRuntime(), mx


class TestResolveModelRepo:
    def test_appends_suffix_to_base_id(self) -> None:
        assert resolve_model_repo(SPEC) == "mlx-community/TestModel-4bit"

    def test_matching_suffix_used_verbatim(self) -> None:
        spec = ModelSpec(model_id="mlx-community/TestModel-8bit", quantization=Quantization.Q8)
        assert resolve_model_repo(spec) == "mlx-community/TestModel-8bit"

    def test_mismatched_suffix_raises(self) -> None:
        spec = ModelSpec(model_id="mlx-community/TestModel-4bit", quantization=Quantization.Q8)
        with pytest.raises(ValueError, match="is a 4bit repo but 8bit was requested"):
            resolve_model_repo(spec)

    def test_repo_override_wins(self) -> None:
        spec = ModelSpec(
            model_id="anything",
            quantization=Quantization.FP16,
            repo_override="me/custom-repo",
        )
        assert resolve_model_repo(spec) == "me/custom-repo"


class TestMLXRuntime:
    def test_generate_concatenates_stream_and_maps_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime, _ = make_runtime(monkeypatch)
        runtime.load(SPEC)
        result = runtime.generate("hi", max_tokens=8)

        assert result.text == "Hello world"
        assert result.metrics.prompt_tokens == 12
        assert result.metrics.generation_tokens == 3
        assert result.metrics.prompt_tps == 200.0
        assert result.metrics.generation_tps == 40.0
        assert result.metrics.peak_memory_bytes == PEAK_BYTES  # exact bytes, no unit round-trip
        assert result.metrics.time_to_first_token_s > 0

    def test_load_resets_peak_memory_per_variant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, mx = make_runtime(monkeypatch)
        runtime.load(SPEC)
        assert mx.reset_calls == 1
        runtime.load(ModelSpec(model_id="other/Model", quantization=Quantization.Q8))
        assert mx.reset_calls == 2  # sweep variants cannot inherit earlier peaks

    def test_unload_clears_mlx_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, mx = make_runtime(monkeypatch)
        runtime.load(SPEC)
        runtime.unload()
        assert mx.clear_cache_calls == 1

    def test_zero_peak_memory_reported_as_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(
            monkeypatch, responses=[FakeResponse("x")], fake_mx=FakeMx(peak_bytes=0)
        )
        runtime.load(SPEC)
        assert runtime.generate("hi").metrics.peak_memory_bytes is None

    def test_generate_before_load_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(monkeypatch)
        with pytest.raises(InvalidStateError, match="no model loaded"):
            runtime.generate("hi")

    def test_generate_after_unload_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(monkeypatch)
        runtime.load(SPEC)
        runtime.unload()
        with pytest.raises(InvalidStateError, match="no model loaded"):
            runtime.generate("hi")

    def test_empty_stream_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime, _ = make_runtime(monkeypatch, responses=[])
        runtime.load(SPEC)
        with pytest.raises(InvalidStateError, match="no tokens"):
            runtime.generate("hi")

    def test_load_failure_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_load(repo: str) -> tuple[str, str]:
            raise OSError("repo not found")

        runtime, _ = make_runtime(monkeypatch)
        fake = make_fake_mlx_lm([])
        fake.load = broken_load
        monkeypatch.setattr(mlx_runtime, "_import_mlx_lm", lambda: fake)
        with pytest.raises(ModelLoadError, match="repo not found"):
            runtime.load(SPEC)

    def test_missing_mlx_lm_raises_runtime_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "mlx_lm", None)
        with pytest.raises(RuntimeUnavailableError, match="mlx-lm is not installed"):
            MLXRuntime().load(SPEC)
