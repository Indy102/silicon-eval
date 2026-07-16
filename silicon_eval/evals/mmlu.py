"""MMLU: multiple-choice knowledge questions, answer-letter scoring.

Zero-shot hendrycks-style prompting: the question and lettered options are
formatted into a prompt ending in ``Answer:``, and the four continuations
`` A``/`` B``/`` C``/`` D`` are ranked by conditional log-likelihood via
``Runtime.score_completion``. All continuations have equal length, so raw
accuracy is the only metric (length normalization would not change the
argmax).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from silicon_eval.evals.base import EvalResult, MultipleChoiceItem
from silicon_eval.evals.datasets import load_mmlu_records
from silicon_eval.exceptions import DatasetLoadError
from silicon_eval.runtimes.base import Runtime

_LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True, slots=True)
class MMLUConfig:
    """Scoring budget: how many test items and the context window."""

    max_items: int | None = 100
    max_context_tokens: int = 1024


def record_to_item(record: dict[str, object]) -> MultipleChoiceItem:
    """Format a raw MMLU record into a prompt + answer letters."""
    choices = record["choices"]
    if not isinstance(choices, list) or len(choices) != len(_LETTERS):
        raise DatasetLoadError(f"malformed MMLU record: choices={choices!r}")
    subject = str(record.get("subject", "")).replace("_", " ")
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(_LETTERS, choices, strict=True)
    )
    context = (
        f"The following are multiple choice questions (with answers) about {subject}.\n\n"
        f"{str(record['question']).strip()}\n{options}\nAnswer:"
    )
    return MultipleChoiceItem(
        context=context, choices=_LETTERS, answer_index=int(str(record["answer"]))
    )


class MMLUEvaluator:
    """Multiple-choice accuracy of the loaded model on the MMLU test split."""

    name: str = "mmlu"

    def __init__(
        self,
        config: MMLUConfig | None = None,
        records_loader: Callable[[int | None], list[dict[str, object]]] | None = None,
    ) -> None:
        self._config = config if config is not None else MMLUConfig()
        self._records_loader = (
            records_loader
            if records_loader is not None
            else lambda max_items: load_mmlu_records(max_items=max_items)
        )
        self._items: list[MultipleChoiceItem] | None = None

    def run(self, runtime: Runtime) -> EvalResult:
        items = self._load_items()
        start = time.perf_counter()
        correct = 0
        for item in items:
            log_likelihoods = [
                -runtime.score_completion(
                    item.context,
                    f" {letter}",
                    max_context_tokens=self._config.max_context_tokens,
                ).negative_log_likelihood
                for letter in item.choices
            ]
            correct += _argmax(log_likelihoods) == item.answer_index
        return EvalResult(
            name=self.name,
            metrics={"accuracy": correct / len(items), "items": len(items)},
            duration_s=time.perf_counter() - start,
        )

    def _load_items(self) -> list[MultipleChoiceItem]:
        """Load and format once; a quantization sweep reuses one evaluator."""
        if self._items is None:
            records = self._records_loader(self._config.max_items)
            if not records:
                raise DatasetLoadError("MMLU loader returned no records")
            self._items = [record_to_item(record) for record in records]
        return self._items


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)
