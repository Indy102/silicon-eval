"""Runtime abstraction: the protocol every inference backend must satisfy.

Evals and profiling depend only on this module, never on a concrete backend,
so adding a runtime (llama.cpp, etc.) means implementing ``Runtime`` and
registering it — nothing upstream changes. See docs/adr/001 for rationale.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Quantization(StrEnum):
    """Quantization levels a model variant can be evaluated at.

    ``fp16`` and ``bf16`` are distinct half-precision formats; mlx-community
    publishes one or the other depending on the model — pick the one that
    exists for yours.
    """

    Q4 = "4bit"
    Q8 = "8bit"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model variant to evaluate: a base model at a quantization level.

    ``repo_override`` pins an exact repo id when the runtime's naming
    convention doesn't apply (e.g. a private fork or unusual repo name).
    ``file_override`` additionally pins a file within the repo for runtimes
    whose repos hold one artifact per quantization (e.g. GGUF files for
    llama.cpp); runtimes that load whole repos ignore it.
    """

    model_id: str
    quantization: Quantization
    repo_override: str | None = None
    file_override: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Performance counters from a single generation call."""

    prompt_tokens: int
    generation_tokens: int
    time_to_first_token_s: float
    prompt_tps: float
    generation_tps: float
    peak_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Generated text plus the metrics observed while producing it."""

    text: str
    metrics: GenerationMetrics


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Aggregate negative log-likelihood of a text under the model.

    ``negative_log_likelihood`` is a sum in nats over ``scored_tokens`` tokens;
    perplexity is ``exp(nll / scored_tokens)``.
    """

    negative_log_likelihood: float
    scored_tokens: int
    windows: int


@runtime_checkable
class Runtime(Protocol):
    """Inference backend contract.

    Lifecycle: ``load(spec)`` → one or more ``generate(...)`` calls → ``unload()``.
    Implementations raise :class:`~silicon_eval.exceptions.RuntimeUnavailableError`
    when their backing library is missing, so callers can degrade gracefully.
    """

    name: str

    def load(self, spec: ModelSpec) -> None:
        """Download (if needed) and load the model variant into memory."""
        ...

    def generate(self, prompt: str, *, max_tokens: int = 128) -> GenerationResult:
        """Generate a completion for ``prompt`` and measure it."""
        ...

    def score(
        self,
        text: str,
        *,
        max_context_tokens: int = 512,
        max_windows: int | None = None,
    ) -> ScoreResult:
        """Sum the negative log-likelihood of ``text`` under the model.

        The runtime tokenizes ``text`` and scores it in consecutive
        non-overlapping windows of at most ``max_context_tokens`` targets
        (each window's last token seeds the next window's context, so every
        token except the first is scored exactly once). ``max_windows``
        caps the work for long corpora; ``None`` scores everything.
        """
        ...

    def score_completion(
        self,
        context: str,
        continuation: str,
        *,
        max_context_tokens: int = 2048,
    ) -> ScoreResult:
        """NLL of ``continuation`` conditioned on ``context`` (one forward pass).

        Continuation length is measured as ``len(tokens(context + continuation))
        - len(tokens(context))`` so tokenizer merges at the boundary are handled.
        If the combined sequence exceeds ``max_context_tokens``, context is
        truncated from the left; the continuation is always fully scored.
        """
        ...

    def unload(self) -> None:
        """Release the loaded model's memory."""
        ...


def iter_score_windows(
    token_ids: Sequence[int],
    *,
    max_context_tokens: int,
    max_windows: int | None,
) -> Iterator[Sequence[int]]:
    """Yield the consecutive scoring windows ``Runtime.score`` requires.

    Window k covers ``token_ids[k*W : k*W + W + 1]``: it predicts tokens
    ``k*W+1 .. k*W+W`` from their prefixes, and its last token seeds window
    k+1's context — so every token except the first is scored exactly once.
    Shared by runtime implementations so their windowing cannot diverge.
    """
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be >= 1")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be >= 1, or None to score everything")
    if len(token_ids) < 2:
        raise ValueError("text yields fewer than 2 tokens; nothing to score")
    for index, start in enumerate(range(0, len(token_ids) - 1, max_context_tokens)):
        if max_windows is not None and index >= max_windows:
            return
        yield token_ids[start : start + max_context_tokens + 1]


def parse_quant_list(raw: str) -> list[Quantization]:
    """Parse a comma-separated CLI string like ``"4bit,8bit"`` into levels.

    Deduplicates while preserving order; raises ``ValueError`` on unknown names.
    """
    quants: list[Quantization] = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        try:
            quants.append(Quantization(name))
        except ValueError:
            valid = ", ".join(q.value for q in Quantization)
            raise ValueError(f"unknown quantization {name!r} (expected one of: {valid})") from None
    if not quants:
        raise ValueError("no quantization levels given")
    return list(dict.fromkeys(quants))
