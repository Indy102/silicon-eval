"""Tests for the MMLU evaluator (runtime mocked)."""

from __future__ import annotations

import pytest

from silicon_eval.evals.base import Evaluator
from silicon_eval.evals.mmlu import MMLUConfig, MMLUEvaluator, record_to_item
from silicon_eval.exceptions import DatasetLoadError
from tests.conftest import FakeRuntime

RECORDS: list[dict[str, object]] = [
    {
        "question": "What is 2 + 2?",
        "subject": "abstract_algebra",
        "choices": ["3", "4", "5", "22"],
        "answer": 1,
    },
    {
        "question": "Which planet is largest?",
        "subject": "astronomy",
        "choices": ["Mars", "Venus", "Jupiter", "Pluto"],
        "answer": 2,
    },
]


def make_evaluator(
    config: MMLUConfig | None = None,
    records: list[dict[str, object]] | None = None,
) -> MMLUEvaluator:
    data = records if records is not None else RECORDS
    return MMLUEvaluator(config, records_loader=lambda max_items: data[:max_items])


class TestRecordToItem:
    def test_formats_hendrycks_style_prompt(self) -> None:
        item = record_to_item(RECORDS[0])
        assert item.context == (
            "The following are multiple choice questions (with answers) "
            "about abstract algebra.\n\n"
            "What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 22\nAnswer:"
        )
        assert item.choices == ("A", "B", "C", "D")
        assert item.answer_index == 1

    def test_malformed_choices_raise(self) -> None:
        record = dict(RECORDS[0])
        record["choices"] = ["only", "two"]
        with pytest.raises(DatasetLoadError, match="malformed"):
            record_to_item(record)


class TestMMLUEvaluator:
    def test_satisfies_evaluator_protocol(self) -> None:
        assert isinstance(make_evaluator(), Evaluator)

    def test_accuracy_from_letter_likelihoods(self, fake_runtime: FakeRuntime) -> None:
        # Item 1 answer is B; item 2 answer is C. Make " B" the most likely
        # continuation everywhere: item 1 scores correct, item 2 wrong.
        fake_runtime.completion_nlls = {" A": 4.0, " B": 1.0, " C": 2.0, " D": 4.0}
        result = make_evaluator().run(fake_runtime)

        assert result.name == "mmlu"
        assert result.metrics["accuracy"] == pytest.approx(0.5)
        assert result.metrics["items"] == 2
        assert result.duration_s >= 0

    def test_letters_scored_with_leading_space(self, fake_runtime: FakeRuntime) -> None:
        make_evaluator().run(fake_runtime)
        continuations = {call[1] for call in fake_runtime.completion_calls}
        assert continuations == {" A", " B", " C", " D"}

    def test_config_plumbed(self, fake_runtime: FakeRuntime) -> None:
        make_evaluator(MMLUConfig(max_items=1, max_context_tokens=256)).run(fake_runtime)
        assert len({call[0] for call in fake_runtime.completion_calls}) == 1
        assert all(call[2] == 256 for call in fake_runtime.completion_calls)

    def test_items_loaded_once_across_sweep(self, fake_runtime: FakeRuntime) -> None:
        loads = 0

        def counting_loader(max_items: int | None) -> list[dict[str, object]]:
            nonlocal loads
            loads += 1
            return RECORDS

        evaluator = MMLUEvaluator(records_loader=counting_loader)
        evaluator.run(fake_runtime)
        evaluator.run(fake_runtime)
        assert loads == 1

    def test_empty_records_raise(self, fake_runtime: FakeRuntime) -> None:
        with pytest.raises(DatasetLoadError, match="no records"):
            make_evaluator(records=[]).run(fake_runtime)
