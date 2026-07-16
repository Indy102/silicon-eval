"""llama.cpp runtime via llama-cpp-python, loading GGUF files from the Hub.

llama-cpp-python is imported lazily so this module stays importable where it
isn't installed (it is an optional extra: ``pip install 'silicon-eval[llamacpp]'``).

Measurement notes, per ADR-004:

- Timing is measured by silicon-eval around the streaming API (llama.cpp's
  internal timings are not exposed portably), so TTFT includes prompt
  evaluation plus the first decode step, and prompt tok/s is derived from it.
- llama.cpp does not report an accelerator memory peak; ``peak_memory_bytes``
  is ``None`` and reports show n/a for these rows.
- GGUF quantization schemes (Q4_K_M, Q8_0, …) differ from MLX's group-wise
  affine quantization at the same nominal bit width: cross-runtime rows
  compare what each runtime actually ships at that level, not identical
  weights.
"""

from __future__ import annotations

import re
import time
import warnings
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

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
    iter_score_windows,
)

# Filename fragments per level, in preference order. Q4_K_M is the community
# standard "4-bit"; Q4_0 is the fallback for older conversions. Fragments are
# matched as delimited segments, so "f16" can never match a "bf16" file.
_QUANT_PATTERNS: dict[Quantization, tuple[str, ...]] = {
    Quantization.Q4: ("q4_k_m", "q4_0"),
    Quantization.Q8: ("q8_0",),
    Quantization.FP16: ("fp16", "f16"),
    Quantization.BF16: ("bf16",),
}


def _pattern_matches(pattern: str, name: str) -> bool:
    return re.search(rf"(^|[-_.]){re.escape(pattern)}([-_.]|$)", name.lower()) is not None


def _import_llama_cpp() -> Any:  # untyped third-party module
    try:
        import llama_cpp
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "llama-cpp-python is not installed; "
            "install it with: pip install 'silicon-eval[llamacpp]'"
        ) from exc
    return llama_cpp


def resolve_gguf_file(spec: ModelSpec, repo_files: list[str]) -> str:
    """Pick the repo's GGUF file matching the spec's quantization level.

    ``file_override`` wins outright. Otherwise filenames are matched against
    the level's patterns in preference order; multi-part GGUFs (``-of-``)
    are rejected explicitly rather than half-loaded.
    """
    if spec.file_override is not None:
        return spec.file_override
    ggufs = [name for name in repo_files if name.lower().endswith(".gguf")]
    for pattern in _QUANT_PATTERNS[spec.quantization]:
        matches = sorted(name for name in ggufs if _pattern_matches(pattern, name))
        if not matches:
            continue
        chosen = matches[0]
        if "-of-" in chosen.lower():
            raise ModelLoadError(
                f"{chosen!r} is a multi-part GGUF, which silicon-eval does not "
                "load; pass file_override for a single-file artifact"
            )
        return chosen
    patterns = ", ".join(_QUANT_PATTERNS[spec.quantization])
    raise ModelLoadError(
        f"no GGUF matching {spec.quantization.value} (patterns: {patterns}) "
        f"among {len(ggufs)} .gguf files in the repo; pass file_override"
    )


class LlamaCppRuntime:
    """Runs GGUF models through llama-cpp-python (Metal-accelerated on macOS).

    ``model_id`` is a GGUF repository id (e.g. ``Qwen/Qwen2.5-0.5B-Instruct-GGUF``).
    ``n_ctx`` bounds both scoring windows and generation length; it is kept
    modest because scoring needs ``logits_all``, whose buffer costs
    ``n_ctx * vocab * 4`` bytes (~0.6 GB at 1024 for a 152k vocab).
    """

    name: str = "llama.cpp"

    def __init__(self, n_ctx: int = 1024) -> None:
        self._n_ctx = n_ctx
        self._llama: Any = None
        self._spec: ModelSpec | None = None

    def load(self, spec: ModelSpec) -> None:
        llama_cpp = _import_llama_cpp()
        path = self._download_gguf(spec)
        try:
            self._llama = llama_cpp.Llama(
                model_path=path,
                n_ctx=self._n_ctx,
                logits_all=True,  # required for score()/score_completion()
                n_gpu_layers=-1,  # full Metal offload
                verbose=False,
            )
        except Exception as exc:
            raise ModelLoadError(f"failed to load {path!r}: {exc}") from exc
        self._spec = spec

    def generate(self, prompt: str, *, max_tokens: int = 128) -> GenerationResult:
        if self._llama is None:
            raise InvalidStateError("no model loaded; call load() first")
        # Count prompt tokens the way create_completion will evaluate them
        # (model-native BOS/special handling), so metrics and the context
        # clamp agree with what the model actually processes.
        prompt_tokens = len(self._llama.tokenize(prompt.encode("utf-8"), special=True))
        max_tokens = self._clamp_generation(prompt_tokens, max_tokens)
        # Clear the KV cache: llama.cpp reuses matching prompt prefixes across
        # calls, which would turn every post-warmup TTFT into a cache hit and
        # inflate prompt tok/s unboundedly with prompt length.
        self._llama.reset()

        start = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        for chunk in self._llama.create_completion(
            prompt, max_tokens=max_tokens, temperature=0.0, stream=True
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(chunk["choices"][0]["text"])
        end = time.perf_counter()

        if first_token_at is None:
            raise InvalidStateError("model produced no tokens")
        text = "".join(chunks)
        # Streamed chunks can carry several tokens each (multi-byte buffering),
        # so token counts come from retokenizing the emitted text (ADR-004).
        generated = len(self._tokenize(text)) if text else 1
        return GenerationResult(
            text=text,
            metrics=_timed_metrics(
                prompt_tokens=prompt_tokens,
                generation_tokens=max(generated, 1),
                start=start,
                first_token_at=first_token_at,
                end=end,
            ),
        )

    def score(
        self,
        text: str,
        *,
        max_context_tokens: int = 512,
        max_windows: int | None = None,
    ) -> ScoreResult:
        if self._llama is None:
            raise InvalidStateError("no model loaded; call load() first")
        max_context_tokens = self._clamp_window(max_context_tokens)

        token_ids = self._tokenize(text)
        total_nll = 0.0
        scored = 0
        windows = 0
        for window in iter_score_windows(
            token_ids, max_context_tokens=max_context_tokens, max_windows=max_windows
        ):
            total_nll += self._window_nll(list(window))
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
        if self._llama is None:
            raise InvalidStateError("no model loaded; call load() first")
        max_context_tokens = self._clamp_window(max_context_tokens)

        context_ids = self._tokenize(context)
        full_ids = self._tokenize(context + continuation)
        n_continuation = len(full_ids) - len(context_ids)
        if n_continuation < 1:
            raise ValueError("continuation adds no tokens to the context")
        if n_continuation >= len(full_ids):
            raise ValueError("context must contribute at least one token")
        if n_continuation + 1 > max_context_tokens:
            raise ValueError(
                f"continuation ({n_continuation} tokens) does not fit in "
                f"max_context_tokens={max_context_tokens}"
            )
        # Left-truncate long contexts; the continuation is always fully scored.
        full_ids = full_ids[-max_context_tokens:]
        nll = self._window_nll(full_ids, last_n_targets=n_continuation)
        return ScoreResult(negative_log_likelihood=nll, scored_tokens=n_continuation, windows=1)

    def unload(self) -> None:
        if self._llama is not None and hasattr(self._llama, "close"):
            self._llama.close()
        self._llama = None
        self._spec = None

    def _download_gguf(self, spec: ModelSpec) -> str:
        repo = spec.repo_override if spec.repo_override is not None else spec.model_id
        try:
            if spec.file_override is not None:
                filename = spec.file_override
            else:
                files = HfApi().list_repo_files(repo)
                filename = resolve_gguf_file(spec, files)
            return hf_hub_download(repo, filename)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"failed to fetch a GGUF from {repo!r}: {exc}") from exc

    def _tokenize(self, text: str) -> list[int]:
        # No BOS / special tokens: scoring treats text as a raw continuation
        # stream. MLXRuntime._encode passes add_special_tokens=False for the
        # same reason — the two must stay in lockstep (ADR-004).
        return list(self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def _window_nll(self, window: list[int], last_n_targets: int | None = None) -> float:
        """Negative log-likelihood of a window's targets via logits_all scores."""
        import numpy as np  # llama-cpp-python already depends on numpy

        self._llama.reset()
        self._llama.eval(window)
        n = len(window)
        logits = np.asarray(self._llama.scores[: n - 1], dtype=np.float64)
        targets = np.asarray(window[1:])
        peak = logits.max(axis=-1, keepdims=True)
        logsumexp = peak + np.log(np.exp(logits - peak).sum(axis=-1, keepdims=True))
        token_logprobs = np.take_along_axis(logits, targets[:, None], axis=-1) - logsumexp
        if last_n_targets is not None:
            if last_n_targets > token_logprobs.shape[0]:
                raise ValueError(
                    f"cannot score {last_n_targets} targets from "
                    f"{token_logprobs.shape[0]} predictions"
                )
            token_logprobs = token_logprobs[-last_n_targets:]
        return float(-token_logprobs.sum())

    def _clamp_window(self, max_context_tokens: int) -> int:
        limit = self._n_ctx - 1
        if max_context_tokens > limit:
            warnings.warn(
                f"scoring window clamped from {max_context_tokens} to {limit} "
                f"tokens by llama.cpp's context (n_ctx={self._n_ctx}); other "
                "runtimes may score larger windows for the same arguments",
                stacklevel=3,
            )
            return limit
        return max_context_tokens

    def _clamp_generation(self, prompt_tokens: int, max_tokens: int) -> int:
        if max_tokens < 1:
            # llama.cpp reinterprets max_tokens <= 0 as "fill the context".
            raise ValueError("max_tokens must be >= 1")
        available = self._n_ctx - prompt_tokens
        if available < 1:
            raise ValueError(
                f"prompt ({prompt_tokens} tokens) fills the whole context "
                f"(n_ctx={self._n_ctx}); nothing left to generate"
            )
        return min(max_tokens, available)


def _timed_metrics(
    *,
    prompt_tokens: int,
    generation_tokens: int,
    start: float,
    first_token_at: float,
    end: float,
) -> GenerationMetrics:
    """Wall-clock metrics around the streaming API (see module docstring)."""
    ttft = first_token_at - start
    decode_time = end - first_token_at
    if generation_tokens > 1 and decode_time > 0:
        generation_tps = (generation_tokens - 1) / decode_time
    else:
        generation_tps = generation_tokens / max(end - start, 1e-9)
    return GenerationMetrics(
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        time_to_first_token_s=ttft,
        prompt_tps=prompt_tokens / max(ttft, 1e-9),
        generation_tps=generation_tps,
        peak_memory_bytes=None,  # llama.cpp exposes no accelerator peak
    )
