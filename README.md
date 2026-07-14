# silicon-eval

Evaluation and profiling harness for LLMs on Apple Silicon. Measure the
quality / performance / efficiency tradeoffs of quantized models across
runtimes, and decide what to ship on-device with numbers instead of vibes.

> **Status: v0.1 — Phase 1 of 4.** Runtime abstraction, MLX generation with
> metrics, and the CLI skeleton are in place. Quality evals (perplexity,
> task benchmarks), energy sampling, reports, and result caching land in the
> next phases. Real benchmark numbers will appear here once the measurement
> pipeline is complete.

## Why

Picking a model for on-device inference means choosing a model **and** a
quantization level **and** a runtime — and the tradeoffs are coupled. 4-bit
halves your memory but what does it do to MMLU? Is MLX or llama.cpp faster
for *your* model on *your* chip? silicon-eval runs the matrix and gives you
one comparison table.

## Install

Requires Python 3.11+ and an Apple Silicon Mac for actual inference.

```sh
pip install "silicon-eval[mlx] @ git+https://github.com/indysingh/silicon-eval"
```

## Quickstart

```sh
silicon-eval run --model mlx-community/Qwen2.5-0.5B-Instruct --quant 4bit,8bit
```

Model ids follow mlx-community naming: pass the base id and silicon-eval
appends `-4bit` / `-8bit` / `-fp16` per quantization level, or pass the exact
repo id for a single level.

As a library:

```python
from silicon_eval.runtimes import get_runtime
from silicon_eval.runtimes.base import ModelSpec, Quantization

runtime = get_runtime("mlx")
runtime.load(ModelSpec("mlx-community/Qwen2.5-0.5B-Instruct", Quantization.Q4))
result = runtime.generate("Explain KV caching in one sentence.", max_tokens=64)
print(result.metrics.generation_tps, "tok/s")
```

## Architecture

```
silicon_eval/
  runtimes/        # Runtime protocol + MLXRuntime (llama.cpp planned)
  evals/           # (Phase 2/3) perplexity, task benchmarks
  profiling/       # (Phase 2/3) latency, memory, energy samplers
  report/          # (Phase 2/3) JSON schema + Markdown renderer
  cache/           # (Phase 3) content-addressed result cache
  cli.py           # Typer CLI
```

The load-bearing decision: evals and profiling depend on a `Runtime`
*protocol*, never on MLX directly, so new backends drop in without touching
measurement code. See [docs/adr/001-runtime-abstraction.md](docs/adr/001-runtime-abstraction.md).

## Development

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,mlx]"

ruff check . && ruff format --check .   # lint
mypy                                     # typecheck (strict)
pytest                                   # unit tests (no MLX needed)
pytest -m slow --no-cov                  # integration test: real 0.5B model, Apple Silicon only
```

CI (GitHub Actions) runs lint, typecheck, and unit tests on Linux — the
runtime layer is mocked there. Integration tests require Apple Silicon and
run locally only.
