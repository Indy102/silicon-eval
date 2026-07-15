# silicon-eval

Evaluation and profiling harness for LLMs on Apple Silicon. Measure the
quality / performance / efficiency tradeoffs of quantized models across
runtimes, and decide what to ship on-device with numbers instead of vibes.

> **Status: v0.3 — Phase 3 of 4.** Runtime abstraction, MLX generation,
> perplexity on WikiText-2, HellaSwag accuracy, latency + memory profiling,
> powermetrics energy sampling, JSON + Markdown reports, and a result cache
> are in place. Remaining: llama.cpp runtime, README benchmark tables from
> real runs, PyPI packaging polish.

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
silicon-eval run --model mlx-community/Qwen2.5-0.5B-Instruct --quant 4bit,8bit \
    -o report.json --markdown report.md
```

Each variant gets generation profiling (time-to-first-token, prompt and
generation tok/s over repeated runs, peak memory), perplexity on WikiText-2,
HellaSwag multiple-choice accuracy, and — where available — energy per token,
printed as a table and written as structured JSON and/or a Markdown
comparison. Results are cached per config, so re-runs only compute what
changed ([ADR-003](docs/adr/003-result-cache.md)). Useful knobs:

- `--ppl-windows N` — how many 512-token windows of WikiText-2 to score
  (default 50 ≈ 25k tokens; `0` = full corpus). Scored token counts are
  recorded in the report, and all variants score the identical prefix. See
  [docs/adr/002](docs/adr/002-perplexity-methodology.md) for the methodology.
- `--hs-items N` — HellaSwag validation items (default 100; small-sample
  noise is real, but variants are compared on identical items).
- `--runs / --warmup` — measured and unmeasured generation repetitions.
- `--no-perplexity / --no-hellaswag / --no-energy` — skip pieces.
- `--no-cache` — re-measure everything and refresh the cached entries.

Energy sampling uses `powermetrics`, which needs root. Grant passwordless
sudo for exactly that binary (`<user> ALL=(root) NOPASSWD:
/usr/bin/powermetrics` via `sudo visudo`) or run silicon-eval with sudo;
otherwise the run degrades gracefully and reports why. Note a cached variant
remembers that energy was unavailable — after enabling sudo, re-measure with
`--no-cache`.

Model ids follow mlx-community naming: pass the base id and silicon-eval
appends `-4bit` / `-8bit` / `-fp16` per quantization level, or pass the exact
repo id for a single level.

What the numbers mean:

- **ppl** — perplexity over a fixed WikiText-2 prefix; comparable across the
  variants in one report, not to published sliding-window results.
- **hswag** — HellaSwag accuracy (length-normalized log-likelihood scoring,
  lm-eval-style) over the first `--hs-items` validation items.
- **ttft / gen t/s** — steady-state stats over `--runs` generations, after
  `--warmup` unmeasured runs absorb kernel-compilation cost.
- **peak metal** — accelerator-side peak unified memory since the variant's
  model load (weights + KV cache + activations). This is the number that
  matters for "will it fit".
- **peak rss** — host-side process RSS. On macOS, Metal-backed model memory
  largely does **not** appear here; treat it as Python/tokenizer overhead,
  not total footprint.
- **mJ/tok** — system-wide CPU+GPU+ANE energy per generated token over
  dedicated generation runs; includes machine baseline load, so keep the
  machine otherwise idle.

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
  evals/           # perplexity on WikiText-2, HellaSwag multiple-choice
  profiling/       # generation latency stats, RSS sampling, powermetrics energy
  report/          # JSON schema + machine info + Markdown renderer
  cache/           # content-addressed result cache
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

Troubleshooting: if `silicon-eval` fails with `ModuleNotFoundError: No module
named 'silicon_eval'` after an editable install on macOS, check whether the
OS stamped the install's `.pth` file with the hidden flag — Python 3.13+
silently skips hidden `.pth` files, and some sandboxed/agentic environments
re-apply the flag after you clear it:

```sh
ls -lO .venv/lib/python3.13/site-packages/*.pth   # look for "hidden"
chflags nohidden .venv/lib/python3.13/site-packages/*.pth
```

If the flag keeps coming back, either run with the project root on
`PYTHONPATH` (`PYTHONPATH="$PWD" silicon-eval …`) or use a regular install
(`pip install .`), which doesn't rely on a `.pth` file.
