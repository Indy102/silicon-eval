"""Typer CLI: `silicon-eval run --model X --quant 4bit,8bit`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from silicon_eval import __version__
from silicon_eval.cache import ResultCache, cache_key
from silicon_eval.evals.base import Evaluator
from silicon_eval.evals.hellaswag import HellaSwagConfig, HellaSwagEvaluator
from silicon_eval.evals.perplexity import PerplexityConfig, PerplexityEvaluator
from silicon_eval.report.json_io import variant_from_dict, variant_to_dict, write_report_json
from silicon_eval.report.machine import collect_machine_info
from silicon_eval.report.markdown import write_report_markdown
from silicon_eval.report.schema import SCHEMA_VERSION, VariantResult
from silicon_eval.runner import build_report, run_variant
from silicon_eval.runtimes import backend_versions, get_runtime
from silicon_eval.runtimes.base import ModelSpec, parse_quant_list

app = typer.Typer(
    name="silicon-eval",
    help="Evaluate and profile LLM quantization variants on Apple Silicon.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"silicon-eval {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """Evaluate and profile LLM quantization variants on Apple Silicon."""


@app.command()
def run(
    model: Annotated[str, typer.Option("--model", "-m", help="Model id, e.g. mlx-community/…")],
    quant: Annotated[
        str, typer.Option("--quant", "-q", help="Comma-separated levels: 4bit,8bit,fp16")
    ] = "4bit",
    runtime: Annotated[str, typer.Option("--runtime", "-r", help="Inference backend")] = "mlx",
    prompt: Annotated[
        str, typer.Option("--prompt", help="Prompt used for generation profiling")
    ] = "Explain quantization of neural networks in one sentence.",
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Tokens per profiled run")] = 64,
    runs: Annotated[int, typer.Option("--runs", help="Measured generation runs")] = 3,
    warmup: Annotated[int, typer.Option("--warmup", help="Unmeasured warmup runs")] = 1,
    perplexity: Annotated[
        bool,
        typer.Option("--perplexity/--no-perplexity", help="Score perplexity on WikiText-2"),
    ] = True,
    ppl_windows: Annotated[
        int,
        typer.Option("--ppl-windows", help="WikiText-2 windows to score (0 = full corpus)"),
    ] = 50,
    ppl_context: Annotated[
        int, typer.Option("--ppl-context", help="Perplexity window size in tokens")
    ] = 512,
    hellaswag: Annotated[
        bool,
        typer.Option("--hellaswag/--no-hellaswag", help="Run HellaSwag multiple-choice eval"),
    ] = True,
    hs_items: Annotated[
        int, typer.Option("--hs-items", help="HellaSwag validation items to score")
    ] = 100,
    energy: Annotated[
        bool,
        typer.Option(
            "--energy/--no-energy",
            help="Sample energy via powermetrics (needs passwordless sudo; degrades gracefully)",
        ),
    ] = True,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the JSON report here")
    ] = None,
    markdown: Annotated[
        Path | None, typer.Option("--markdown", help="Write the Markdown report here")
    ] = None,
    cache: Annotated[
        bool,
        typer.Option(
            "--cache/--no-cache",
            help="Reuse cached results for unchanged configs; "
            "--no-cache re-measures and refreshes the cache",
        ),
    ] = True,
) -> None:
    """Profile and evaluate each quantization variant, then report."""
    try:
        _validate_flags(
            runs=runs,
            warmup=warmup,
            max_tokens=max_tokens,
            ppl_windows=ppl_windows,
            ppl_context=ppl_context,
            hs_items=hs_items,
        )
        quants = parse_quant_list(quant)
        backend = get_runtime(runtime)
    except ValueError as exc:
        _fail(str(exc))

    evaluators = _build_evaluators(
        perplexity=perplexity,
        ppl_windows=ppl_windows,
        ppl_context=ppl_context,
        hellaswag=hellaswag,
        hs_items=hs_items,
    )
    result_cache = ResultCache()
    base_payload = _cache_payload_base(
        model=model,
        runtime=runtime,
        prompt=prompt,
        max_tokens=max_tokens,
        runs=runs,
        warmup=warmup,
        perplexity=perplexity,
        ppl_windows=ppl_windows,
        ppl_context=ppl_context,
        hellaswag=hellaswag,
        hs_items=hs_items,
        energy=energy,
    )

    variants: list[VariantResult] = []
    cached_quants: set[str] = set()
    failure: str | None = None
    for level in quants:
        key = cache_key({**base_payload, "quantization": level.value})
        if cache:
            cached = _cached_variant(result_cache, key)
            if cached is not None:
                typer.echo(f"[{backend.name}] {model} @ {level.value}: cached")
                variants.append(cached)
                cached_quants.add(level.value)
                continue
        spec = ModelSpec(model_id=model, quantization=level)
        typer.echo(f"[{backend.name}] {model} @ {level.value}: measuring…")
        try:
            variant = run_variant(
                backend,
                spec,
                prompt=prompt,
                max_tokens=max_tokens,
                runs=runs,
                warmup=warmup,
                evaluators=evaluators,
                measure_energy=energy,
            )
        except Exception as exc:  # CLI boundary: fail cleanly, keep finished variants
            failure = f"{model} @ {level.value}: {exc}"
            break
        variants.append(variant)
        # Written even with --no-cache (refresh semantics), and never allowed
        # to cost measured results: a cache is an optimization, not an output.
        try:
            result_cache.put(key, variant_to_dict(variant))
        except OSError as exc:
            typer.echo(f"note: could not write result cache: {exc}")

    if variants:
        report = build_report(variants)
        typer.echo(_render_table(report.variants))
        _echo_energy_notes(report.variants, cached_quants, energy_requested=energy)
        for path, writer in ((output, write_report_json), (markdown, write_report_markdown)):
            if path is not None:
                try:
                    writer(report, path)
                except OSError as exc:
                    _fail(f"could not write report to {path}: {exc}")
                typer.echo(f"report written to {path}")
    if failure is not None:
        _fail(failure)


def _build_evaluators(
    *, perplexity: bool, ppl_windows: int, ppl_context: int, hellaswag: bool, hs_items: int
) -> list[Evaluator]:
    evaluators: list[Evaluator] = []
    if perplexity:
        evaluators.append(
            PerplexityEvaluator(
                PerplexityConfig(
                    max_context_tokens=ppl_context,
                    max_windows=ppl_windows if ppl_windows > 0 else None,
                )
            )
        )
    if hellaswag:
        evaluators.append(HellaSwagEvaluator(HellaSwagConfig(max_items=hs_items)))
    return evaluators


def _cache_payload_base(**config: Any) -> dict[str, Any]:
    machine = collect_machine_info()
    return {
        # Versions are part of the key: a fix in silicon-eval, an upgraded
        # backend (new kernels change every number), or an OS update must
        # invalidate old entries rather than silently mix stale numbers into
        # freshly timestamped reports.
        "silicon_eval_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "backend_versions": backend_versions(str(config.get("runtime", ""))),
        "machine": {
            "chip": machine.chip,
            "memory_bytes": machine.memory_bytes,
            "os_version": machine.os_version,
        },
        **config,
    }


def _cached_variant(result_cache: ResultCache, key: str) -> VariantResult | None:
    data = result_cache.get(key)
    if data is None:
        return None
    try:
        return variant_from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None  # stale/corrupt entry: recompute


def _echo_energy_notes(
    variants: list[VariantResult], cached_quants: set[str], *, energy_requested: bool
) -> None:
    if not energy_requested:
        return
    reasons = {
        variant.energy_unavailable_reason
        for variant in variants
        if variant.energy is None and variant.energy_unavailable_reason
    }
    for reason in sorted(reasons):
        typer.echo(f"note: energy sampling unavailable: {reason}")
    stale_energy = any(
        variant.energy is None
        and variant.energy_unavailable_reason
        and variant.quantization.value in cached_quants
        for variant in variants
    )
    if stale_energy:
        typer.echo(
            "note: that reason is from a cached result — after fixing it, "
            "re-measure with --no-cache"
        )


def _validate_flags(
    *, runs: int, warmup: int, max_tokens: int, ppl_windows: int, ppl_context: int, hs_items: int
) -> None:
    if runs < 1:
        raise ValueError("--runs must be >= 1")
    if warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")  # mlx-lm treats -1 as unbounded
    if ppl_windows < 0:
        raise ValueError("--ppl-windows must be >= 0 (0 scores the full corpus)")
    if ppl_context < 1:
        raise ValueError("--ppl-context must be >= 1")
    if hs_items < 1:
        raise ValueError("--hs-items must be >= 1")


def _render_table(variants: list[VariantResult]) -> str:
    header = (
        f"{'quant':<6} {'ppl':>8} {'hswag':>7} {'ttft (s)':>9} {'gen t/s':>9} "
        f"{'peak metal':>11} {'mJ/tok':>8}"
    )
    lines = [header, "-" * len(header)]
    for variant in variants:
        ppl = _eval_metric(variant, "perplexity")
        accuracy = _eval_metric(variant, "accuracy_norm")
        generation = variant.generation
        energy = variant.energy
        lines.append(
            f"{variant.quantization.value:<6} "
            f"{f'{ppl:.2f}' if ppl is not None else 'n/a':>8} "
            f"{f'{accuracy:.3f}' if accuracy is not None else 'n/a':>7} "
            f"{generation.ttft_s.mean:>9.3f} {generation.generation_tps.mean:>9.1f} "
            f"{_format_mb(generation.peak_metal_bytes):>11} "
            f"{f'{energy.energy_per_generated_token_mj:.1f}' if energy else 'n/a':>8}"
        )
    return "\n".join(lines)


def _eval_metric(variant: VariantResult, key: str) -> float | None:
    for eval_result in variant.evals:
        value = eval_result.metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _format_mb(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "n/a"
    return f"{size_bytes / 1024**2:.0f} MB"


def _fail(message: str) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)
