"""Integration test against a real tiny model.

Apple Silicon only; excluded from the default run and from CI.
Run locally with: pytest -m slow --no-cov
"""

from __future__ import annotations

import pytest

from silicon_eval.runtimes.base import ModelSpec, Quantization
from silicon_eval.runtimes.mlx_runtime import MLXRuntime

pytestmark = pytest.mark.slow

TINY_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def test_real_generation_produces_text_and_metrics() -> None:
    pytest.importorskip("mlx_lm")
    runtime = MLXRuntime()
    runtime.load(ModelSpec(model_id=TINY_MODEL, quantization=Quantization.Q4))
    try:
        result = runtime.generate("The capital of France is", max_tokens=16)
    finally:
        runtime.unload()

    assert result.text.strip()
    assert result.metrics.prompt_tokens > 0
    assert 0 < result.metrics.generation_tokens <= 16
    assert result.metrics.time_to_first_token_s > 0
    assert result.metrics.generation_tps > 0
    assert result.metrics.peak_memory_bytes is not None
    assert result.metrics.peak_memory_bytes > 0
