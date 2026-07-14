"""Orchestrates one variant: load → profile → evaluate → assemble a result."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from silicon_eval import __version__
from silicon_eval.evals.base import Evaluator
from silicon_eval.profiling.generation import profile_generation
from silicon_eval.profiling.memory import RssSampler
from silicon_eval.report.machine import collect_machine_info
from silicon_eval.report.schema import SCHEMA_VERSION, Report, VariantResult
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
) -> VariantResult:
    """Measure one model variant end to end; always unloads the model."""
    with RssSampler() as rss:
        runtime.load(spec)
        try:
            generation = profile_generation(
                runtime, prompt, max_tokens=max_tokens, runs=runs, warmup=warmup
            )
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
