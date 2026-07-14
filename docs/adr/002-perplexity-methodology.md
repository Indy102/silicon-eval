# ADR-002: Perplexity via non-overlapping windows with a default budget

Status: accepted · Date: 2026-07-13

## Context

Perplexity on WikiText-2 is silicon-eval's first quality metric. Two
methodology choices matter for whether the numbers are honest and comparable:
how long documents are split to fit a context window, and how much of the
corpus is scored.

Full-corpus sliding-window scoring (stride < window, as in the Hugging Face
perplexity guide) gives the tightest estimate but costs one forward pass per
stride step — minutes to hours per variant on the small machines this tool
targets (reference machine: M1, 8 GB).

## Decision

1. **Non-overlapping windows.** The corpus is concatenated, tokenized once,
   and scored in consecutive windows of `max_context_tokens` targets. Each
   window's last token seeds the next window's context, so every token except
   the very first is scored exactly once. This is one forward pass per
   `max_context_tokens` tokens — the cheapest defensible scheme.
2. **Default budget of 50 windows** (`--ppl-windows`, `PerplexityConfig
   .max_windows`) ≈ 25k tokens with the default 512-token window. `0`/`None`
   scores the full corpus. The report always records `scored_tokens` and
   `windows`, so a subsampled number can never masquerade as a full-corpus one.
3. **Chunking lives in `Runtime.score()`**, not the evaluator, because it
   needs the runtime's tokenizer. The evaluator stays pure math
   (`exp(nll / tokens)`) and is runtime-agnostic per ADR-001.

## Consequences

- Absolute values run slightly higher than sliding-window numbers (early
  tokens in each window have short context). Comparisons **within** a
  silicon-eval report are apples-to-apples: every variant scores the identical
  token prefix with identical windowing.
- Numbers are not directly comparable to published sliding-window results;
  the README will say so wherever benchmark tables appear.
- A stride/overlap option can be added later as a pure extension
  (`PerplexityConfig` + one loop change) if tighter estimates are worth the
  compute.
