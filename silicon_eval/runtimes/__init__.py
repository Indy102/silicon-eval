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

# Distributions whose versions determine a runtime's measured numbers.
_BACKEND_DISTS: dict[str, tuple[str, ...]] = {
    "mlx": ("mlx", "mlx-lm"),
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


def backend_versions(name: str) -> dict[str, str]:
    """Installed versions of the packages backing a runtime.

    Recorded in reports and cache keys: a backend upgrade changes every
    number, so it must be attributable and must invalidate cached results.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for dist in _BACKEND_DISTS.get(name, ()):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            versions[dist] = "not-installed"
    return versions


__all__ = [
    "GenerationMetrics",
    "GenerationResult",
    "MLXRuntime",
    "ModelSpec",
    "Quantization",
    "Runtime",
    "available_runtimes",
    "backend_versions",
    "get_runtime",
    "parse_quant_list",
]
