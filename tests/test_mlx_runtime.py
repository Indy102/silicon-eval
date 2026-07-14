"""Tests for MLXRuntime with mlx-lm mocked out."""

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


@dataclass
class FakeResponse:
    text: str
    prompt_tokens: int = 12
    prompt_tps: float = 200.0
    generation_tokens: int = 3
    generation_tps: float = 40.0
    peak_memory: float = 0.5  # GB, matching mlx-lm's GenerationResponse


def make_fake_mlx_lm(responses: list[FakeResponse]) -> SimpleNamespace:
    def load(repo: str) -> tuple[str, str]:
        return (f"model:{repo}", f"tokenizer:{repo}")

    def stream_generate(model: Any, tokenizer: Any, *, prompt: str, max_tokens: int) -> Any:
        yield from responses

    return SimpleNamespace(load=load, stream_generate=stream_generate)


@pytest.fixture
def patched_runtime(monkeypatch: pytest.MonkeyPatch) -> MLXRuntime:
    fake = make_fake_mlx_lm([FakeResponse("Hello"), FakeResponse(" world")])
    monkeypatch.setattr(mlx_runtime, "_import_mlx_lm", lambda: fake)
    return MLXRuntime()


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
        self, patched_runtime: MLXRuntime
    ) -> None:
        patched_runtime.load(SPEC)
        result = patched_runtime.generate("hi", max_tokens=8)

        assert result.text == "Hello world"
        assert result.metrics.prompt_tokens == 12
        assert result.metrics.generation_tokens == 3
        assert result.metrics.prompt_tps == 200.0
        assert result.metrics.generation_tps == 40.0
        assert result.metrics.peak_memory_bytes == int(0.5 * 1024**3)
        assert result.metrics.time_to_first_token_s > 0

    def test_generate_before_load_raises(self, patched_runtime: MLXRuntime) -> None:
        with pytest.raises(InvalidStateError, match="no model loaded"):
            patched_runtime.generate("hi")

    def test_generate_after_unload_raises(self, patched_runtime: MLXRuntime) -> None:
        patched_runtime.load(SPEC)
        patched_runtime.unload()
        with pytest.raises(InvalidStateError, match="no model loaded"):
            patched_runtime.generate("hi")

    def test_empty_stream_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = make_fake_mlx_lm([])
        monkeypatch.setattr(mlx_runtime, "_import_mlx_lm", lambda: fake)
        runtime = MLXRuntime()
        runtime.load(SPEC)
        with pytest.raises(InvalidStateError, match="no tokens"):
            runtime.generate("hi")

    def test_load_failure_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_load(repo: str) -> tuple[str, str]:
            raise OSError("repo not found")

        fake = make_fake_mlx_lm([])
        fake.load = broken_load
        monkeypatch.setattr(mlx_runtime, "_import_mlx_lm", lambda: fake)
        with pytest.raises(ModelLoadError, match="repo not found"):
            MLXRuntime().load(SPEC)

    def test_missing_mlx_lm_raises_runtime_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "mlx_lm", None)
        with pytest.raises(RuntimeUnavailableError, match="mlx-lm is not installed"):
            MLXRuntime().load(SPEC)

    def test_zero_peak_memory_reported_as_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = make_fake_mlx_lm([FakeResponse("x", peak_memory=0.0)])
        monkeypatch.setattr(mlx_runtime, "_import_mlx_lm", lambda: fake)
        runtime = MLXRuntime()
        runtime.load(SPEC)
        assert runtime.generate("hi").metrics.peak_memory_bytes is None
