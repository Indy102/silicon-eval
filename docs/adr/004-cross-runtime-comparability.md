# ADR-004: What cross-runtime comparisons do and don't hold constant

Status: accepted · Date: 2026-07-15

## Context

v0.5 adds a llama.cpp runtime beside MLX, enabling the tool's original
purpose: same model, same nominal quantization level, two runtimes, one
table. But "4bit on MLX" and "4bit on llama.cpp" are not the same weights,
and the two backends expose different instrumentation. Pretending otherwise
would produce dishonest tables.

## Decision

Cross-runtime rows compare **what each runtime actually ships at a level**,
with the differences stated rather than hidden:

1. **Quantization schemes differ.** MLX 4bit is group-wise affine int4 over
   mlx-community weights; llama.cpp 4bit resolves to GGUF `Q4_K_M`
   (`Q4_0` fallback). Same nominal width, different algorithms, different
   artifacts. That is the comparison an engineer shipping on-device actually
   faces — pick a runtime and you get its quantizer.
2. **Tokenization is model-native per runtime, with specials disabled on
   both sides.** Both use the model's own tokenizer (HF for MLX,
   GGUF-embedded for llama.cpp), and both scoring paths explicitly request
   raw tokens — MLX passes ``add_special_tokens=False``, llama.cpp passes
   ``add_bos=False, special=False`` — because HF tokenizers otherwise
   prepend BOS for some model families (Llama, Gemma), which would shift
   every scoring window on one runtime only. Enforced by unit tests on both
   runtimes.
3. **Timing methodology differs and is labeled.** MLX rows use mlx-lm's
   internal counters plus silicon-eval's TTFT clock. llama.cpp rows are
   wall-clock around the streaming API: TTFT includes prompt eval plus the
   first decode step, prompt tok/s is derived from TTFT, and generation
   tok/s excludes the first token. The KV cache is reset before every
   generation — llama.cpp otherwise reuses matching prompt prefixes across
   calls, which would turn post-warmup TTFT into a cache-hit artifact (this
   bit v0.5's first benchmark draft: measured TTFT collapsed to exactly one
   decode period). llama.cpp token counts come from retokenizing the emitted
   text (streamed chunks can carry several tokens), and multi-byte output
   buffering can add up to a few decode periods to its TTFT. Cross-runtime
   latency comparisons are sound at table granularity; microsecond-level
   readings are not.
4. **Memory reporting is asymmetric.** MLX reports a true Metal allocator
   peak; llama.cpp exposes no equivalent, so `peak metal` is n/a for its
   rows and only host RSS is recorded. An absent number is better than a
   fabricated one.
5. **Scoring math is shared, with one bounded divergence.** Both runtimes
   window corpora through the same `iter_score_windows` helper and score
   continuations with the same left-truncation contract, so quality metrics
   cannot diverge by implementation accident. The exception: llama.cpp needs
   `logits_all`, whose buffer costs `n_ctx * vocab * 4` bytes, so its
   context defaults to 1024 and scoring windows **clamp to n_ctx − 1 with a
   `UserWarning`** — a `--ppl-context` larger than that yields smaller
   effective windows on llama.cpp than on MLX (which has no clamp). Heed the
   warning when building cross-runtime tables with large windows; the
   defaults (512) are unaffected.

## Consequences

- A cross-runtime table answers "which stack should I ship at ~4 bits?" —
  not "which kernel multiplies matrices faster on identical weights?".
  The README says so wherever such tables appear.
- GGUF repos hold one file per quantization, so `ModelSpec` gained
  `file_override` and the llama.cpp resolver matches quant patterns against
  repo file lists (multi-part GGUFs are rejected explicitly).
- Energy sampling is runtime-agnostic (powermetrics measures the machine),
  so mJ/token remains comparable across runtimes where available.
