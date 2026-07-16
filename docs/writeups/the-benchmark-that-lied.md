# The benchmark that lied: a KV-cache story

*How my cross-runtime LLM benchmark reported a 9× time-to-first-token win
that didn't exist — and what caught it.*

I build [silicon-eval](https://github.com/Indy102/silicon-eval), an
open-source harness that benchmarks quantized LLMs on Apple Silicon:
perplexity, HellaSwag/MMLU accuracy, latency, memory, and energy, across
MLX and llama.cpp, from one command. The whole point of the project is
"numbers instead of vibes" — every table ships with the machine, versions,
and raw report that produced it.

Which made it interesting when my own first cross-runtime table was wrong.

## The suspicious table

First draft of the MLX vs llama.cpp comparison, Qwen2.5-0.5B-Instruct on an
M1 (8 GB):

| runtime   | quant | ttft (s) | gen tok/s | prompt tok/s |
|-----------|-------|----------|-----------|--------------|
| mlx       | 4bit  | 0.122    | 139.8     | 334.0        |
| llama.cpp | 4bit  | **0.014**| 76.0      | **805.8**    |

Reading: llama.cpp processes prompts 2.4× faster and reaches its first token
9× sooner. Plausible-sounding — llama.cpp's prefill kernels are famously
good. I nearly published it.

## The tell

Look at the llama.cpp row again. Generation runs at 76.0 tokens/second, so
one decode step takes 1/76.0 = 0.0132 seconds. The measured time-to-first-
token was 0.014 seconds.

**TTFT equaled exactly one decode period.** Which means the measurement
contained no prompt evaluation at all — statistically impossible for a real
prefill of a ~10-token prompt, and the "prompt tok/s" column was just
`prompt_tokens / one_decode_step`, a number that would scale with prompt
length rather than measure anything.

## The cause

`llama-cpp-python`'s `Llama.generate()` does something genuinely useful for
chat apps: it keeps the KV cache across calls and, when a new prompt's
tokens match the previous call's prefix, it skips re-evaluating them —
re-processing only the final token.

My profiler, like every careful benchmark harness, runs **warmup
generations** before the measured ones, using the same prompt. So:

1. Warmup call: full prompt eval, fills the KV cache. Unmeasured.
2. Every measured call: prompt matches the cached prefix → llama.cpp
   re-evaluates one token → "TTFT" is one decode step.

MLX (`mlx-lm`) keeps no cross-call prompt cache, so its rows paid full
prefill every run. The flagship comparison was apples-to-oranges — and the
warmup, which exists to make measurements *more* honest, is exactly what
poisoned them.

The fix is one line — reset the KV cache before each measured generation —
plus re-measuring and re-publishing:

| runtime   | quant | ttft (s) | gen tok/s | prompt tok/s |
|-----------|-------|----------|-----------|--------------|
| mlx       | 4bit  | 0.122    | 139.8     | 334.0        |
| llama.cpp | 4bit  | 0.035    | 77.7      | 318.7        |

Still a real TTFT win for llama.cpp (3.5×, from lower per-call overhead) —
but prompt throughput is actually *at parity* with MLX, not 2.4× faster.
The quality story survived untouched: GGUF's Q4_K_M holds near-8-bit
perplexity (18.87 vs 18.44) where MLX's 4-bit scheme trades ~4 points
(21.82) for ~80% more generation speed. That tradeoff — quality-per-bit vs
throughput — is the real headline, and it's only visible when the timing
columns aren't lying.

## What caught it

Not a failing test — every test passed, because the mocked llama.cpp in the
unit suite didn't model the prefix cache. What caught it was an adversarial
review step before publishing: checking each *claim in the methodology
docs* ("TTFT includes prompt evaluation plus the first decode step")
against the actual library source. The claim and
`llama-cpp-python`'s prefix-matching code couldn't both be true, and the
arithmetic (TTFT ≡ one decode period) confirmed which one was fiction.

## What I took away

1. **Cross-check derived quantities.** If one measurement equals a round
   function of another (TTFT = 1/gen_tps), suspect the measurement, not the
   hardware.
2. **Warmup interacts with caches.** Anything cached across calls turns
   "warm steady-state measurement" into "measuring the cache." Know what
   your runtime persists between calls.
3. **Mocks inherit your assumptions.** My fake llama.cpp streamed tokens
   exactly the way I believed the real one did — so unit tests could never
   catch the belief being wrong. The fix included making the fake model the
   real buffer semantics.
4. **Publish raw reports.** Both the wrong and corrected runs are archived
   in the repo (`docs/benchmarks/`), with versions and machine stamped. The
   correction is in the README, not hidden — a benchmark tool that quietly
   fixes its own numbers has no product left.

silicon-eval is MIT-licensed: `pip install` it, run your own chip, and — if
you have an M2/M3/M4 — I'd love a PR with your report so the benchmark
table can grow beyond one machine.
