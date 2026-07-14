"""Runtime registry: look up inference backends by name."""

from __future__ import annotations

from collections.abc import Callable

from silicon_eval.runtimes.base import (
    GenerationMetrics,
    GenerationResult,
    ModelSpec,
    Quantization,
    Runtime,
    parse_quant_list,
)
from silicon_eval.runtimes.mlx_runtime import MLXRuntime

_REGISTRY: dict[str, Callable[[], Runtime]] = {
    "mlx": MLXRuntime,
}


def get_runtime(name: str) -> Runtime:
    """Instantiate the runtime registered under ``name``."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown runtime {name!r} (available: {valid})") from None
    return factory()


def available_runtimes() -> list[str]:
    """Names of all registered runtimes."""
    return sorted(_REGISTRY)


__all__ = [
    "GenerationMetrics",
    "GenerationResult",
    "MLXRuntime",
    "ModelSpec",
    "Quantization",
    "Runtime",
    "available_runtimes",
    "get_runtime",
    "parse_quant_list",
]
