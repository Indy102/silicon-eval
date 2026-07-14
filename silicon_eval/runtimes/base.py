"""Runtime abstraction: the protocol every inference backend must satisfy.

Evals and profiling depend only on this module, never on a concrete backend,
so adding a runtime (llama.cpp, etc.) means implementing ``Runtime`` and
registering it — nothing upstream changes. See docs/adr/001 for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Quantization(StrEnum):
    """Quantization levels a model variant can be evaluated at."""

    Q4 = "4bit"
    Q8 = "8bit"
    FP16 = "fp16"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model variant to evaluate: a base model at a quantization level.

    ``repo_override`` pins an exact repo id when the runtime's naming
    convention doesn't apply (e.g. a private fork or unusual repo name).
    """

    model_id: str
    quantization: Quantization
    repo_override: str | None = None


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

    def unload(self) -> None:
        """Release the loaded model's memory."""
        ...


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
