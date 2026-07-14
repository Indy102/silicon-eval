"""Generation performance profiling: repeated timed runs on a loaded runtime."""

from __future__ import annotations

from dataclasses import dataclass

from silicon_eval.profiling.stats import Stat
from silicon_eval.runtimes.base import Runtime


@dataclass(frozen=True, slots=True)
class GenerationProfile:
    """Latency/throughput statistics over ``runs`` measured generations.

    ``peak_metal_bytes`` is the largest runtime-reported peak unified-memory
    figure observed across the measured runs; runtimes report the peak since
    the variant's model load, so it includes resident weights plus inference
    buffers (``None`` if unreported).
    """

    runs: int
    warmup_runs: int
    max_tokens: int
    ttft_s: Stat
    prompt_tps: Stat
    generation_tps: Stat
    peak_metal_bytes: int | None


def profile_generation(
    runtime: Runtime,
    prompt: str,
    *,
    max_tokens: int = 64,
    runs: int = 3,
    warmup: int = 1,
) -> GenerationProfile:
    """Run ``warmup`` unmeasured then ``runs`` measured generations and aggregate.

    Warmup absorbs one-time costs (Metal kernel compilation, cache priming) so
    the measured runs reflect steady-state performance.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        runtime.generate(prompt, max_tokens=max_tokens)
    measured = [runtime.generate(prompt, max_tokens=max_tokens).metrics for _ in range(runs)]

    peaks = [m.peak_memory_bytes for m in measured if m.peak_memory_bytes is not None]
    return GenerationProfile(
        runs=runs,
        warmup_runs=warmup,
        max_tokens=max_tokens,
        ttft_s=Stat.from_samples([m.time_to_first_token_s for m in measured]),
        prompt_tps=Stat.from_samples([m.prompt_tps for m in measured]),
        generation_tps=Stat.from_samples([m.generation_tps for m in measured]),
        peak_metal_bytes=max(peaks) if peaks else None,
    )
