"""MLX runtime: generation via mlx-lm with streaming metrics.

mlx-lm is imported lazily so this module (and the CLI) stays importable on
machines without Apple Silicon — e.g. Linux CI runners.
"""

from __future__ import annotations

import logging
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
    ScoreResult,
)

_QUANT_SUFFIXES: dict[Quantization, str] = {
    Quantization.Q4: "-4bit",
    Quantization.Q8: "-8bit",
    Quantization.FP16: "-fp16",
    Quantization.BF16: "-bf16",
}


def _import_mlx_lm() -> Any:  # untyped third-party module
    try:
        import mlx_lm
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "mlx-lm is not installed; install it with: pip install 'silicon-eval[mlx]'"
        ) from exc
    return mlx_lm


def _import_mx() -> Any:  # untyped third-party module
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "mlx is not installed; install it with: pip install 'silicon-eval[mlx]'"
        ) from exc
    return mx


def resolve_model_repo(spec: ModelSpec) -> str:
    """Map a model spec to a Hugging Face repo id using mlx-community naming.

    ``mlx-community`` publishes one repo per quantization, suffixed with
    ``-4bit``/``-8bit``/``-fp16``/``-bf16``. A ``model_id`` already carrying
    the matching suffix is used verbatim; a mismatched suffix is an error
    rather than a silently wrong download.
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
        # Peak memory is a process-lifetime counter; reset it so this variant's
        # readings can't be contaminated by previously loaded variants.
        _import_mx().reset_peak_memory()
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
        # Read the peak directly in bytes rather than round-tripping through
        # mlx-lm's decimal-GB float; covers everything since this variant's load.
        peak_bytes = int(_import_mx().get_peak_memory())
        return GenerationResult(
            text="".join(chunks),
            metrics=_metrics(last, time_to_first_token, peak_bytes),
        )

    def score(
        self,
        text: str,
        *,
        max_context_tokens: int = 512,
        max_windows: int | None = None,
    ) -> ScoreResult:
        if self._model is None:
            raise InvalidStateError("no model loaded; call load() first")
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1")
        if max_windows is not None and max_windows < 1:
            raise ValueError("max_windows must be >= 1, or None to score everything")
        mx = _import_mx()

        token_ids = self._encode(text)
        if len(token_ids) < 2:
            raise ValueError("text yields fewer than 2 tokens; nothing to score")

        total_nll = 0.0
        scored = 0
        windows = 0
        for start in range(0, len(token_ids) - 1, max_context_tokens):
            if max_windows is not None and windows >= max_windows:
                break
            # Window k predicts tokens[start+1 .. start+W] from tokens[start .. start+W-1];
            # its last token seeds window k+1, so each token is scored exactly once.
            window = token_ids[start : start + max_context_tokens + 1]
            window_nll = -self._token_logprobs(mx, window).sum()
            mx.eval(window_nll)
            total_nll += float(window_nll)
            scored += len(window) - 1
            windows += 1
        return ScoreResult(negative_log_likelihood=total_nll, scored_tokens=scored, windows=windows)

    def score_completion(
        self,
        context: str,
        continuation: str,
        *,
        max_context_tokens: int = 2048,
    ) -> ScoreResult:
        if self._model is None:
            raise InvalidStateError("no model loaded; call load() first")
        mx = _import_mx()

        context_ids = self._encode(context)
        full_ids = self._encode(context + continuation)
        n_continuation = len(full_ids) - len(context_ids)
        if n_continuation < 1:
            raise ValueError("continuation adds no tokens to the context")
        if n_continuation + 1 > max_context_tokens:
            raise ValueError(
                f"continuation ({n_continuation} tokens) does not fit in "
                f"max_context_tokens={max_context_tokens}"
            )
        # Left-truncate long contexts; the continuation is always fully scored.
        full_ids = full_ids[-max_context_tokens:]

        token_logprobs = self._token_logprobs(mx, full_ids)
        continuation_nll = -token_logprobs[-n_continuation:].sum()
        mx.eval(continuation_nll)
        return ScoreResult(
            negative_log_likelihood=float(continuation_nll),
            scored_tokens=n_continuation,
            windows=1,
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._spec = None
        try:
            mx = _import_mx()
        except RuntimeUnavailableError:
            return
        # Dropping references returns buffers to MLX's cache, not to the OS;
        # without this, a multi-variant sweep accumulates cached Metal memory.
        mx.clear_cache()

    def _encode(self, text: str) -> list[int]:
        """Tokenize without transformers' sequence-length warning.

        That warning assumes the sequence reaches the model whole; scoring
        windows it first, so corpus-length inputs are fine here.
        """
        logger = logging.getLogger("transformers.tokenization_utils_base")
        original_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            return list(self._tokenizer.encode(text))
        finally:
            logger.setLevel(original_level)

    def _token_logprobs(self, mx: Any, token_ids: list[int]) -> Any:
        """Per-token logprobs of ``token_ids[1:]`` given their prefixes; shape [T-1, 1]."""
        inputs = mx.array(token_ids[:-1])[None]
        targets = mx.array(token_ids[1:])
        logits = self._model(inputs)[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return mx.take_along_axis(logprobs, targets[:, None], axis=-1)


def _metrics(
    response: Any, time_to_first_token: float, peak_memory_bytes: int
) -> GenerationMetrics:
    """Build metrics from the final mlx-lm ``GenerationResponse`` of a stream."""
    return GenerationMetrics(
        prompt_tokens=int(response.prompt_tokens),
        generation_tokens=int(response.generation_tokens),
        time_to_first_token_s=time_to_first_token,
        prompt_tps=float(response.prompt_tps),
        generation_tps=float(response.generation_tps),
        peak_memory_bytes=peak_memory_bytes if peak_memory_bytes > 0 else None,
    )
