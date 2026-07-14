# ADR-001: Runtime layer is a Protocol, not hardcoded MLX calls

Status: accepted · Date: 2026-07-13

## Context

silicon-eval measures quality/performance/efficiency tradeoffs of quantized
LLMs on Apple Silicon. v1 ships with MLX (via mlx-lm) as the only inference
backend, but the roadmap explicitly includes llama.cpp — and the most useful
comparisons for a shipping decision are *cross-runtime* (same model, same
quantization, MLX vs llama.cpp).

If evals and profilers called `mlx_lm.load()` / `mlx_lm.stream_generate()`
directly, adding llama.cpp would mean editing every evaluator and sampler, and
none of the measurement code could be unit-tested without an Apple Silicon
machine.

## Decision

Backends implement a structural `Runtime` protocol
(`silicon_eval/runtimes/base.py`):

```python
@runtime_checkable
class Runtime(Protocol):
    name: str
    def load(self, spec: ModelSpec) -> None: ...
    def generate(self, prompt: str, *, max_tokens: int = 128) -> GenerationResult: ...
    def unload(self) -> None: ...
```

- **Structural typing (`Protocol`), not an ABC.** Backends don't import or
  subclass anything from silicon-eval's core; `mypy --strict` verifies
  conformance at the point a backend is registered. A third-party runtime can
  live in a separate package.
- **Everything upstream depends only on the protocol.** Evals, profiling, and
  the CLI receive a `Runtime`; they never import `mlx_lm`. Concrete backends
  are looked up by name through a small registry
  (`silicon_eval/runtimes/get_runtime`).
- **Backend libraries are imported lazily** inside the concrete runtime, and a
  missing library raises `RuntimeUnavailableError`. The package stays
  importable (and unit-testable) on Linux CI runners with no MLX.
- **Measurements travel with results.** `generate()` returns a
  `GenerationResult` carrying `GenerationMetrics` (token counts, TTFT, tok/s,
  peak memory), so profilers consume runtime-reported numbers through one
  shape regardless of backend.

## Consequences

- llama.cpp support is one new module implementing three methods plus a
  registry entry; evals and profiling code are untouched.
- Unit tests substitute a `FakeRuntime` (`tests/conftest.py`) and run
  anywhere; only the `slow`-marked integration test needs real hardware.
- The protocol is the compatibility surface: extending it (e.g. a `logprobs`
  method for perplexity in Phase 2) touches every backend at once. We accept
  that cost — the protocol is small and backends are few.
- Runtime-reported metrics may be computed slightly differently per backend
  (e.g. what counts as prompt processing time). Cross-runtime comparisons must
  document this; where it matters we measure externally (wall clock, psutil)
  rather than trusting backend counters.
