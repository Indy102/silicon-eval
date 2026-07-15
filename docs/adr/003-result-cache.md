# ADR-003: Content-addressed result cache keyed on full run config

Status: accepted · Date: 2026-07-14

## Context

A full sweep (3 quantizations × perplexity × HellaSwag × latency runs) takes
minutes to tens of minutes on the small machines this tool targets. Users
iterate: add a quantization level, tweak one flag, re-run. Recomputing
unchanged variants wastes most of that time.

## Decision

One cache entry per **variant result**, stored as JSON under
`~/.cache/silicon-eval/<key>.json` (override with `SILICON_EVAL_CACHE_DIR`).
The key is the SHA-256 of the canonical JSON (sorted keys, compact
separators) of everything that determines the numbers:

- model id, quantization, runtime name
- profiling config: prompt, max_tokens, runs, warmup
- eval configs: perplexity windows/context + enabled, hellaswag items + enabled
- energy on/off
- **machine**: chip name, total memory, and OS version — results are
  hardware- and driver-specific
- **silicon-eval version, schema version, and backend versions** (mlx,
  mlx-lm) — a measurement fix or a backend upgrade changes the numbers and
  must invalidate old entries rather than silently mix stale figures into
  freshly timestamped reports (this bit us in Phase 2: a units bug fix
  changed every memory figure; version-keyed caching makes such fixes safe)

Properties that follow:

- **Re-runs only compute what changed.** Adding `fp16` to a `4bit,8bit`
  sweep reuses the two finished variants; changing `--max-tokens` recomputes
  everything (correctly — throughput depends on it).
- **Corruption-safe**: unreadable/malformed entries are treated as misses and
  recomputed; writes go through a temp file + atomic rename.
- **Failure-safe**: entries are written only after a variant completes, so a
  mid-sweep crash never caches partial measurements.

## Consequences

- The version keys mean every silicon-eval or backend upgrade recomputes —
  conservative by design; correctness of published comparisons beats cache
  hit-rate.
- `--no-cache` skips reads but still **writes**: a forced re-measure
  refreshes the entry. This matters for energy: enabling passwordless sudo
  later doesn't change any key, so one `--no-cache` run is the documented
  way to replace a cached "energy unavailable" result — and the CLI labels
  such stale notes as cached.
- Cache writes are best-effort: an unwritable cache prints a note and never
  costs measured results (reads already treat errors as misses).
- The cache stores variant *results*, not report files — reports are
  re-assembled (with fresh machine info and timestamps) on every run.
