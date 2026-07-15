# silicon-eval

Evaluation and profiling harness for LLMs on Apple Silicon. Measure the
quality / performance / efficiency tradeoffs of quantized models across
runtimes, and decide what to ship on-device with numbers instead of vibes.

> **Status: v0.4.** Feature-complete for MLX: perplexity on WikiText-2,
> HellaSwag accuracy, latency + memory profiling, powermetrics energy
> sampling, JSON + Markdown reports, and a content-addressed result cache.
> Next up: a llama.cpp runtime for cross-runtime comparisons.

## Why

Picking a model for on-device inference means choosing a model **and** a
quantization level **and** a runtime — and the tradeoffs are coupled. 4-bit
halves your memory but what does it do to MMLU? Is MLX or llama.cpp faster
for *your* model on *your* chip? silicon-eval runs the matrix and gives you
one comparison table.

## Example results

Qwen2.5-0.5B-Instruct across quantization levels — one command, one table:

```sh
silicon-eval run --model mlx-community/Qwen2.5-0.5B-Instruct --quant 4bit,8bit,bf16 \
    -o report.json --markdown report.md
```

| quant | ppl (wikitext2) | hellaswag acc_norm | ttft (s) | gen tok/s | prompt tok/s | peak metal |
|-------|-----------------|--------------------|----------|-----------|--------------|------------|
| 4bit  | 21.82           | 0.490              | 0.122    | 139.8     | 334.0        | 287 MB     |
| 8bit  | 17.87           | 0.480              | 0.144    | 81.4      | 266.1        | 522 MB     |
| bf16  | 17.89           | 0.470              | 0.178    | 47.9      | 177.7        | 969 MB     |

*Measured 2026-07-15 on an Apple M1 (8 GB unified memory), macOS 26.5,
Python 3.13.0, mlx 0.32.0, mlx-lm 0.31.3. Perplexity over the first 25,600
tokens of WikiText-2 (raw, test); HellaSwag over the first 100 validation
items; latency is the mean of 3 runs of 64 tokens after 1 warmup.*

The table is the pitch: on this model, **8-bit matches bf16 quality (ppl
17.87 vs 17.89) at 70% higher throughput and half the memory**, while 4-bit
trades ~4 perplexity points for 3× bf16's speed at a third of the memory.
(At 100 items the HellaSwag differences are within sampling noise — the
perplexity column is the sensitive quality signal here.)
Raw report: [docs/benchmarks/qwen2.5-0.5b-m1-2026-07-15.json](docs/benchmarks/qwen2.5-0.5b-m1-2026-07-15.json).

## Install

Requires Python 3.11+ and an Apple Silicon Mac for actual inference. Until
the first PyPI release, install from a clone of this repository:

```sh
pip install ".[mlx]"
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
sudo for both the sampler and the signal used to stop it (via `sudo visudo`):

```
<user> ALL=(root) NOPASSWD: /usr/bin/powermetrics, /bin/kill
```

(`/bin/kill` is required too — powermetrics runs as root, so silicon-eval
stops it through `sudo -n kill`.) Alternatively run silicon-eval itself with
sudo. Without either, the run degrades gracefully and reports why. Note a
cached variant remembers that energy was unavailable — after enabling sudo,
re-measure with `--no-cache`.

Model ids follow mlx-community naming: pass the base id and silicon-eval
appends `-4bit` / `-8bit` / `-fp16` / `-bf16` per quantization level, or pass
the exact repo id for a single level. (fp16 and bf16 are distinct formats —
check which one mlx-community actually published for your model.)

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

```mermaid
flowchart LR
    CLI["cli.py<br/>Typer CLI"] --> CACHE["cache/<br/>content-addressed<br/>result cache"]
    CLI --> RUNNER["runner.py<br/>load → profile → evaluate"]
    RUNNER --> PROTO["runtimes/base.py<br/><b>Runtime protocol</b><br/>generate · score · score_completion"]
    PROTO --> MLX["MLXRuntime<br/>(mlx-lm)"]
    PROTO -.-> LLAMA["llama.cpp<br/>(planned)"]
    RUNNER --> EVALS["evals/<br/>perplexity · hellaswag"]
    RUNNER --> PROF["profiling/<br/>latency · memory · energy"]
    EVALS --> PROTO
    RUNNER --> REPORT["report/<br/>JSON · Markdown"]
```

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
