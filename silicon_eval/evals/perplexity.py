"""Perplexity on WikiText-2 (raw, test split).

Methodology (see docs/adr/002): the corpus is concatenated, tokenized by the
runtime, and scored in consecutive non-overlapping windows. By default only
the first ``max_windows`` windows are scored to keep runs fast on small
machines — the reported ``scored_tokens`` says exactly how much was measured,
and comparisons across quantization levels use the identical token prefix.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from silicon_eval.evals.base import EvalResult
from silicon_eval.evals.datasets import load_wikitext2_text
from silicon_eval.runtimes.base import Runtime


@dataclass(frozen=True, slots=True)
class PerplexityConfig:
    """Scoring budget: window size in tokens and how many windows to score."""

    max_context_tokens: int = 512
    max_windows: int | None = 50


class PerplexityEvaluator:
    """Computes perplexity of the loaded model over a text corpus."""

    name: str = "perplexity:wikitext2"

    def __init__(
        self,
        config: PerplexityConfig | None = None,
        text_loader: Callable[[], str] | None = None,
    ) -> None:
        self._config = config if config is not None else PerplexityConfig()
        self._text_loader = text_loader if text_loader is not None else load_wikitext2_text
        self._text: str | None = None

    def run(self, runtime: Runtime) -> EvalResult:
        text = self._corpus()
        start = time.perf_counter()
        score = runtime.score(
            text,
            max_context_tokens=self._config.max_context_tokens,
            max_windows=self._config.max_windows,
        )
        duration = time.perf_counter() - start
        perplexity = math.exp(score.negative_log_likelihood / score.scored_tokens)
        return EvalResult(
            name=self.name,
            metrics={
                "perplexity": perplexity,
                "scored_tokens": score.scored_tokens,
                "windows": score.windows,
            },
            duration_s=duration,
        )

    def _corpus(self) -> str:
        """Load the corpus once; a quantization sweep reuses one evaluator."""
        if self._text is None:
            self._text = self._text_loader()
        return self._text
