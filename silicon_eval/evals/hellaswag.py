"""HellaSwag: multiple-choice sentence completion, lm-eval-style scoring.

Each item's four endings are scored by conditional log-likelihood via
``Runtime.score_completion``; the prediction is the argmax. Two accuracies
are reported, following lm-eval conventions:

- ``accuracy`` — argmax of raw summed log-likelihood.
- ``accuracy_norm`` — argmax of log-likelihood divided by the ending's
  character length (compensates for longer endings accumulating more NLL;
  the headline HellaSwag number in most published results).

Preprocessing mirrors lm-eval's hellaswag task: bracketed artifacts like
``[header]`` are stripped and ``ctx_b`` is capitalized.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from silicon_eval.evals.base import EvalResult, MultipleChoiceItem
from silicon_eval.evals.datasets import load_hellaswag_records
from silicon_eval.exceptions import DatasetLoadError
from silicon_eval.runtimes.base import Runtime


@dataclass(frozen=True, slots=True)
class HellaSwagConfig:
    """Scoring budget: how many validation items and the context window."""

    max_items: int | None = 100
    max_context_tokens: int = 2048


def preprocess_text(text: str) -> str:
    """lm-eval's hellaswag text cleanup: strip bracketed WikiHow artifacts."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def record_to_item(record: dict[str, object]) -> MultipleChoiceItem:
    """Build a scored item from a raw HellaSwag record (lm-eval formatting)."""
    context = f"{record['activity_label']}: {record['ctx_a']} {str(record['ctx_b']).capitalize()}"
    endings = record["endings"]
    if not isinstance(endings, list) or len(endings) < 2:
        raise DatasetLoadError(f"malformed HellaSwag record: endings={endings!r}")
    return MultipleChoiceItem(
        context=preprocess_text(context),
        choices=tuple(preprocess_text(str(ending)) for ending in endings),
        answer_index=int(str(record["label"])),
    )


class HellaSwagEvaluator:
    """Multiple-choice accuracy of the loaded model on HellaSwag validation."""

    name: str = "hellaswag"

    def __init__(
        self,
        config: HellaSwagConfig | None = None,
        records_loader: Callable[[int | None], list[dict[str, object]]] | None = None,
    ) -> None:
        self._config = config if config is not None else HellaSwagConfig()
        self._records_loader = (
            records_loader
            if records_loader is not None
            else lambda max_items: load_hellaswag_records(max_items=max_items)
        )
        self._items: list[MultipleChoiceItem] | None = None

    def run(self, runtime: Runtime) -> EvalResult:
        items = self._load_items()
        start = time.perf_counter()
        correct = 0
        correct_norm = 0
        for item in items:
            log_likelihoods = [
                -runtime.score_completion(
                    item.context,
                    f" {choice}",
                    max_context_tokens=self._config.max_context_tokens,
                ).negative_log_likelihood
                for choice in item.choices
            ]
            normalized = [
                ll / max(len(choice), 1)
                for ll, choice in zip(log_likelihoods, item.choices, strict=True)
            ]
            correct += _argmax(log_likelihoods) == item.answer_index
            correct_norm += _argmax(normalized) == item.answer_index
        duration = time.perf_counter() - start
        return EvalResult(
            name=self.name,
            metrics={
                "accuracy": correct / len(items),
                "accuracy_norm": correct_norm / len(items),
                "items": len(items),
            },
            duration_s=duration,
        )

    def _load_items(self) -> list[MultipleChoiceItem]:
        """Load and preprocess once; a quantization sweep reuses one evaluator."""
        if self._items is None:
            records = self._records_loader(self._config.max_items)
            if not records:
                raise DatasetLoadError("HellaSwag loader returned no records")
            self._items = [record_to_item(record) for record in records]
        return self._items


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)
