"""Typer CLI: `silicon-eval run --model X --quant 4bit,8bit`."""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer

from silicon_eval import __version__
from silicon_eval.exceptions import SiliconEvalError
from silicon_eval.runtimes import get_runtime
from silicon_eval.runtimes.base import (
    GenerationResult,
    ModelSpec,
    Quantization,
    Runtime,
    parse_quant_list,
)

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
        str, typer.Option("--prompt", help="Prompt used for the generation smoke test")
    ] = "Explain quantization of neural networks in one sentence.",
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Tokens to generate")] = 64,
    show_text: Annotated[
        bool, typer.Option("--show-text", help="Print each variant's generated text")
    ] = False,
) -> None:
    """Run each quantization variant and report generation metrics."""
    try:
        quants = parse_quant_list(quant)
        backend = get_runtime(runtime)
    except ValueError as exc:
        _fail(str(exc))

    results: list[tuple[Quantization, GenerationResult]] = []
    for level in quants:
        spec = ModelSpec(model_id=model, quantization=level)
        typer.echo(f"[{backend.name}] {model} @ {level.value}: loading…")
        try:
            results.append((level, _run_variant(backend, spec, prompt, max_tokens)))
        except (SiliconEvalError, ValueError) as exc:
            _fail(str(exc))
        if show_text:
            typer.echo(f"--- {level.value} output ---\n{results[-1][1].text}\n")

    typer.echo(_render_table(results))


def _run_variant(
    backend: Runtime, spec: ModelSpec, prompt: str, max_tokens: int
) -> GenerationResult:
    backend.load(spec)
    try:
        return backend.generate(prompt, max_tokens=max_tokens)
    finally:
        backend.unload()


def _render_table(results: list[tuple[Quantization, GenerationResult]]) -> str:
    header = (
        f"{'quant':<6} {'ttft (s)':>9} {'prompt t/s':>11} "
        f"{'gen t/s':>9} {'peak mem':>9} {'tokens':>7}"
    )
    lines = [header, "-" * len(header)]
    for level, result in results:
        m = result.metrics
        peak = f"{m.peak_memory_bytes / 1024**2:.0f} MB" if m.peak_memory_bytes else "n/a"
        lines.append(
            f"{level.value:<6} {m.time_to_first_token_s:>9.3f} {m.prompt_tps:>11.1f} "
            f"{m.generation_tps:>9.1f} {peak:>9} {m.generation_tokens:>7}"
        )
    return "\n".join(lines)


def _fail(message: str) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)
