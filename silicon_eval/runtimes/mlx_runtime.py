"""MLX runtime: generation via mlx-lm with streaming metrics.

mlx-lm is imported lazily so this module (and the CLI) stays importable on
machines without Apple Silicon — e.g. Linux CI runners.
"""

from __future__ import annotations

import time
from typing import Any

from silicon_eval.exceptions import (
    InvalidStateError,
    ModelLoadError,
    RuntimeUnavailableError,
)
from silicon_eval.runtimes.base import (
    GenerationMetrics,
    GenerationResult,
    ModelSpec,
    Quantization,
)

_QUANT_SUFFIXES: dict[Quantization, str] = {
    Quantization.Q4: "-4bit",
    Quantization.Q8: "-8bit",
    Quantization.FP16: "-fp16",
}

_BYTES_PER_GB = 1024**3


def _import_mlx_lm() -> Any:  # untyped third-party module
    try:
        import mlx_lm
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "mlx-lm is not installed; install it with: pip install 'silicon-eval[mlx]'"
        ) from exc
    return mlx_lm


def resolve_model_repo(spec: ModelSpec) -> str:
    """Map a model spec to a Hugging Face repo id using mlx-community naming.

    ``mlx-community`` publishes one repo per quantization, suffixed with
    ``-4bit``/``-8bit``/``-fp16``. A ``model_id`` already carrying the matching
    suffix is used verbatim; a mismatched suffix is an error rather than a
    silently wrong download.
    """
    if spec.repo_override is not None:
        return spec.repo_override
    wanted = _QUANT_SUFFIXES[spec.quantization]
    if spec.model_id.endswith(wanted):
        return spec.model_id
    for quant, suffix in _QUANT_SUFFIXES.items():
        if spec.model_id.endswith(suffix):
            raise ValueError(
                f"model id {spec.model_id!r} is a {quant.value} repo but "
                f"{spec.quantization.value} was requested; pass the base model id "
                "or set repo_override"
            )
    return f"{spec.model_id}{wanted}"


class MLXRuntime:
    """Runs models through mlx-lm on Apple Silicon."""

    name: str = "mlx"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._spec: ModelSpec | None = None

    def load(self, spec: ModelSpec) -> None:
        mlx_lm = _import_mlx_lm()
        repo = resolve_model_repo(spec)
        try:
            self._model, self._tokenizer = mlx_lm.load(repo)
        except Exception as exc:
            raise ModelLoadError(f"failed to load {repo!r}: {exc}") from exc
        self._spec = spec

    def generate(self, prompt: str, *, max_tokens: int = 128) -> GenerationResult:
        if self._model is None:
            raise InvalidStateError("no model loaded; call load() first")
        mlx_lm = _import_mlx_lm()

        start = time.perf_counter()
        time_to_first_token = 0.0
        chunks: list[str] = []
        last: Any = None
        for response in mlx_lm.stream_generate(
            self._model, self._tokenizer, prompt=prompt, max_tokens=max_tokens
        ):
            if last is None:
                time_to_first_token = time.perf_counter() - start
            chunks.append(response.text)
            last = response

        if last is None:
            raise InvalidStateError("model produced no tokens")
        return GenerationResult(text="".join(chunks), metrics=_metrics(last, time_to_first_token))

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._spec = None


def _metrics(response: Any, time_to_first_token: float) -> GenerationMetrics:
    """Build metrics from the final mlx-lm ``GenerationResponse`` of a stream."""
    peak_gb = float(getattr(response, "peak_memory", 0.0))
    return GenerationMetrics(
        prompt_tokens=int(response.prompt_tokens),
        generation_tokens=int(response.generation_tokens),
        time_to_first_token_s=time_to_first_token,
        prompt_tps=float(response.prompt_tps),
        generation_tps=float(response.generation_tps),
        peak_memory_bytes=int(peak_gb * _BYTES_PER_GB) if peak_gb > 0 else None,
    )
