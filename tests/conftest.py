"""Shared test doubles."""

from __future__ import annotations

import pytest

from silicon_eval.runtimes.base import (
    GenerationMetrics,
    GenerationResult,
    ModelSpec,
    ScoreResult,
)

CANNED_METRICS = GenerationMetrics(
    prompt_tokens=12,
    generation_tokens=64,
    time_to_first_token_s=0.35,
    prompt_tps=210.0,
    generation_tps=42.5,
    peak_memory_bytes=512 * 1024 * 1024,
)

# exp(20/10) = e^2 ≈ 7.389 perplexity
CANNED_SCORE = ScoreResult(negative_log_likelihood=20.0, scored_tokens=10, windows=2)


class FakeRuntime:
    """In-memory Runtime implementation that records its calls."""

    name: str = "fake"

    def __init__(self) -> None:
        self.loaded_specs: list[ModelSpec] = []
        self.prompts: list[str] = []
        self.scored_texts: list[str] = []
        self.score_args: list[tuple[int, int | None]] = []
        self.unload_count = 0
        self.score_result = CANNED_SCORE

    def load(self, spec: ModelSpec) -> None:
        self.loaded_specs.append(spec)

    def generate(self, prompt: str, *, max_tokens: int = 128) -> GenerationResult:
        self.prompts.append(prompt)
        return GenerationResult(text=f"fake output ({max_tokens} max)", metrics=CANNED_METRICS)

    def score(
        self,
        text: str,
        *,
        max_context_tokens: int = 512,
        max_windows: int | None = None,
    ) -> ScoreResult:
        self.scored_texts.append(text)
        self.score_args.append((max_context_tokens, max_windows))
        return self.score_result

    def unload(self) -> None:
        self.unload_count += 1


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()
