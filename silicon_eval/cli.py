"""Typer CLI: `silicon-eval run --model X --quant 4bit,8bit`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from silicon_eval import __version__
from silicon_eval.evals.base import Evaluator
from silicon_eval.evals.perplexity import PerplexityConfig, PerplexityEvaluator
from silicon_eval.report.json_io import write_report_json
from silicon_eval.report.schema import VariantResult
from silicon_eval.runner import build_report, run_variant
from silicon_eval.runtimes import get_runtime
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
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the JSON report here")
    ] = None,
) -> None:
    """Profile and evaluate each quantization variant, then report."""
    try:
        _validate_flags(runs=runs, warmup=warmup, ppl_windows=ppl_windows, ppl_context=ppl_context)
        quants = parse_quant_list(quant)
        backend = get_runtime(runtime)
    except ValueError as exc:
        _fail(str(exc))

    evaluators: list[Evaluator] = []
    if perplexity:
        config = PerplexityConfig(
            max_context_tokens=ppl_context,
            max_windows=ppl_windows if ppl_windows > 0 else None,
        )
        evaluators.append(PerplexityEvaluator(config))

    variants: list[VariantResult] = []
    failure: str | None = None
    for level in quants:
        spec = ModelSpec(model_id=model, quantization=level)
        typer.echo(f"[{backend.name}] {model} @ {level.value}: measuring…")
        try:
            variants.append(
                run_variant(
                    backend,
                    spec,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    runs=runs,
                    warmup=warmup,
                    evaluators=evaluators,
                )
            )
        except Exception as exc:  # CLI boundary: fail cleanly, keep finished variants
            failure = f"{model} @ {level.value}: {exc}"
            break

    if variants:
        report = build_report(variants)
        typer.echo(_render_table(report.variants))
        if output is not None:
            try:
                write_report_json(report, output)
            except OSError as exc:
                _fail(f"could not write report to {output}: {exc}")
            typer.echo(f"report written to {output}")
    if failure is not None:
        _fail(failure)


def _validate_flags(*, runs: int, warmup: int, ppl_windows: int, ppl_context: int) -> None:
    if runs < 1:
        raise ValueError("--runs must be >= 1")
    if warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if ppl_windows < 0:
        raise ValueError("--ppl-windows must be >= 0 (0 scores the full corpus)")
    if ppl_context < 1:
        raise ValueError("--ppl-context must be >= 1")


def _render_table(variants: list[VariantResult]) -> str:
    header = (
        f"{'quant':<6} {'ppl':>8} {'ttft (s)':>9} {'gen t/s':>9} {'peak metal':>11} {'peak rss':>9}"
    )
    lines = [header, "-" * len(header)]
    for variant in variants:
        ppl = _eval_metric(variant, "perplexity")
        gen = variant.generation
        lines.append(
            f"{variant.quantization.value:<6} "
            f"{f'{ppl:.2f}' if ppl is not None else 'n/a':>8} "
            f"{gen.ttft_s.mean:>9.3f} {gen.generation_tps.mean:>9.1f} "
            f"{_format_mb(gen.peak_metal_bytes):>11} {_format_mb(variant.peak_rss_bytes):>9}"
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
