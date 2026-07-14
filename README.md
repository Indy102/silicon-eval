# silicon-eval

Evaluation and profiling harness for LLMs on Apple Silicon. Measure the
quality / performance / efficiency tradeoffs of quantized models across
runtimes, and decide what to ship on-device with numbers instead of vibes.

> **Status: v0.2 — Phase 2 of 4.** Runtime abstraction, MLX generation,
> perplexity on WikiText-2, latency + memory profiling, and JSON reports are
> in place. Task benchmarks, energy sampling, Markdown reports, and result
> caching land in the next phases. Real benchmark numbers will appear here
> once the measurement pipeline is complete.

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
silicon-eval run --model mlx-community/Qwen2.5-0.5B-Instruct --quant 4bit,8bit -o report.json
```

Each variant gets generation profiling (time-to-first-token, prompt and
generation tok/s over repeated runs, peak memory) and perplexity on
WikiText-2, printed as a table and written as structured JSON with machine
context. Useful knobs:

- `--ppl-windows N` — how many 512-token windows of WikiText-2 to score
  (default 50 ≈ 25k tokens; `0` = full corpus). Scored token counts are
  recorded in the report, and all variants score the identical prefix. See
  [docs/adr/002](docs/adr/002-perplexity-methodology.md) for the methodology.
- `--runs / --warmup` — measured and unmeasured generation repetitions.
- `--no-perplexity` — profiling only.

Model ids follow mlx-community naming: pass the base id and silicon-eval
appends `-4bit` / `-8bit` / `-fp16` per quantization level, or pass the exact
repo id for a single level.

What the numbers mean:

- **ppl** — perplexity over a fixed WikiText-2 prefix; comparable across the
  variants in one report, not to published sliding-window results.
- **ttft / gen t/s** — steady-state stats over `--runs` generations, after
  `--warmup` unmeasured runs absorb kernel-compilation cost.
- **peak metal** — accelerator-side peak unified memory since the variant's
  model load (weights + KV cache + activations). This is the number that
  matters for "will it fit".
- **peak rss** — host-side process RSS. On macOS, Metal-backed model memory
  largely does **not** appear here; treat it as Python/tokenizer overhead,
  not total footprint.

As a library:

```python
from silicon_eval.evals import PerplexityConfig, PerplexityEvaluator
from silicon_eval.runner import build_report, run_variant
from silicon_eval.runtimes import get_runtime
from silicon_eval.runtimes.base import ModelSpec, Quantization

variant = run_variant(
    get_runtime("mlx"),
    ModelSpec("mlx-community/Qwen2.5-0.5B-Instruct", Quantization.Q4),
    prompt="Explain KV caching in one sentence.",
    evaluators=[PerplexityEvaluator(PerplexityConfig(max_windows=50))],
)
print(variant.generation.generation_tps.mean, "tok/s")
```

## Architecture

```
silicon_eval/
  runtimes/        # Runtime protocol + MLXRuntime (llama.cpp planned)
  evals/           # perplexity on WikiText-2; task benchmarks in Phase 3
  profiling/       # generation latency stats, RSS sampling; energy in Phase 3
  report/          # JSON schema + machine info; Markdown renderer in Phase 3
  cache/           # (Phase 3) content-addressed result cache
  runner.py        # per-variant orchestration: load → profile → evaluate
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
