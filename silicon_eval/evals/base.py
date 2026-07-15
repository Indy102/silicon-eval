"""Evaluator abstraction: quality evals are small pluggable classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from silicon_eval.runtimes.base import Runtime


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Outcome of one evaluator run: named metrics plus wall-clock duration."""

    name: str
    metrics: dict[str, float | int]
    duration_s: float


@dataclass(frozen=True, slots=True)
class MultipleChoiceItem:
    """One multiple-choice question: pick the most likely continuation."""

    context: str
    choices: tuple[str, ...]
    answer_index: int


@runtime_checkable
class Evaluator(Protocol):
    """Quality eval contract: given a loaded runtime, produce metrics."""

    name: str

    def run(self, runtime: Runtime) -> EvalResult:
        """Evaluate the currently loaded model."""
        ...
