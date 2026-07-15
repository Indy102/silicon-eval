"""Orchestrates one variant: load → profile → evaluate → assemble a result."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from silicon_eval import __version__
from silicon_eval.evals.base import Evaluator
from silicon_eval.profiling.energy import PowerMetricsSampler
from silicon_eval.profiling.generation import profile_generation
from silicon_eval.profiling.memory import RssSampler
from silicon_eval.report.machine import collect_machine_info
from silicon_eval.report.schema import (
    SCHEMA_VERSION,
    EnergyProfile,
    Report,
    VariantResult,
)
from silicon_eval.runtimes import backend_versions
from silicon_eval.runtimes.base import ModelSpec, Runtime


def run_variant(
    runtime: Runtime,
    spec: ModelSpec,
    *,
    prompt: str,
    max_tokens: int = 64,
    runs: int = 3,
    warmup: int = 1,
    evaluators: Sequence[Evaluator] = (),
    measure_energy: bool = False,
) -> VariantResult:
    """Measure one model variant end to end; always unloads the model."""
    energy: EnergyProfile | None = None
    energy_reason: str | None = None
    with RssSampler() as rss:
        runtime.load(spec)
        try:
            generation = profile_generation(
                runtime, prompt, max_tokens=max_tokens, runs=runs, warmup=warmup
            )
            if measure_energy:
                energy, energy_reason = _measure_energy(runtime, prompt, max_tokens, runs)
            eval_results = [evaluator.run(runtime) for evaluator in evaluators]
        finally:
            runtime.unload()
    return VariantResult(
        model_id=spec.model_id,
        quantization=spec.quantization,
        runtime=runtime.name,
        generation=generation,
        evals=eval_results,
        peak_rss_bytes=rss.peak_rss_bytes,
        energy=energy,
        energy_unavailable_reason=energy_reason,
        backend_versions=backend_versions(runtime.name),
    )


def _measure_energy(
    runtime: Runtime, prompt: str, max_tokens: int, runs: int
) -> tuple[EnergyProfile | None, str | None]:
    """Sample system power over dedicated generation runs (model already warm)."""
    generated_tokens = 0
    with PowerMetricsSampler() as sampler:
        if not sampler.available:
            return None, sampler.unavailable_reason
        for _ in range(runs):
            result = runtime.generate(prompt, max_tokens=max_tokens)
            generated_tokens += result.metrics.generation_tokens
    reading = sampler.reading()
    if reading is None or generated_tokens == 0:
        return None, "powermetrics produced no samples"
    return (
        EnergyProfile(
            mean_power_mw=reading.mean_power_mw,
            energy_per_generated_token_mj=reading.energy_mj / generated_tokens,
            generated_tokens=generated_tokens,
            samples=reading.samples,
            duration_s=reading.duration_s,
        ),
        None,
    )


def build_report(variants: Sequence[VariantResult]) -> Report:
    """Stamp a set of variant results with version, time, and machine context."""
    return Report(
        schema_version=SCHEMA_VERSION,
        silicon_eval_version=__version__,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        machine=collect_machine_info(),
        variants=list(variants),
    )
