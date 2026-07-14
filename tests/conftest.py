"""Shared test doubles."""

from __future__ import annotations

import pytest

from silicon_eval.runtimes.base import GenerationMetrics, GenerationResult, ModelSpec

CANNED_METRICS = GenerationMetrics(
    prompt_tokens=12,
    generation_tokens=64,
    time_to_first_token_s=0.35,
    prompt_tps=210.0,
    generation_tps=42.5,
    peak_memory_bytes=512 * 1024 * 1024,
)


class FakeRuntime:
    """In-memory Runtime implementation that records its calls."""

    name: str = "fake"

    def __init__(self) -> None:
        self.loaded_specs: list[ModelSpec] = []
        self.prompts: list[str] = []
        self.unload_count = 0

    def load(self, spec: ModelSpec) -> None:
        self.loaded_specs.append(spec)

    def generate(self, prompt: str, *, max_tokens: int = 128) -> GenerationResult:
        self.prompts.append(prompt)
        return GenerationResult(text=f"fake output ({max_tokens} max)", metrics=CANNED_METRICS)

    def unload(self) -> None:
        self.unload_count += 1


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()
