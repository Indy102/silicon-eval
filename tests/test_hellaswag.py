"""Tests for the HellaSwag evaluator (runtime mocked)."""

from __future__ import annotations

import pytest

from silicon_eval.evals.base import Evaluator
from silicon_eval.evals.hellaswag import (
    HellaSwagConfig,
    HellaSwagEvaluator,
    preprocess_text,
    record_to_item,
)
from silicon_eval.exceptions import DatasetLoadError
from tests.conftest import FakeRuntime

RECORDS: list[dict[str, object]] = [
    {
        "activity_label": "Baking",
        "ctx_a": "A person mixes flour and water.",
        "ctx_b": "they",
        "endings": ["good", "badbadbad"],
        "label": "0",
    },
    {
        "activity_label": "Sports",
        "ctx_a": "A player lines up the shot.",
        "ctx_b": "then",
        "endings": ["yes", "no"],
        "label": "1",
    },
]


def make_evaluator(
    config: HellaSwagConfig | None = None,
    records: list[dict[str, object]] | None = None,
) -> HellaSwagEvaluator:
    data = records if records is not None else RECORDS
    return HellaSwagEvaluator(config, records_loader=lambda max_items: data[:max_items])


class TestPreprocessText:
    def test_strips_bracketed_artifacts(self) -> None:
        assert preprocess_text("Do it. [step] Then [substeps] rest") == "Do it. Then rest"

    def test_title_marker_becomes_sentence_break(self) -> None:
        assert preprocess_text("Intro [title] Next part") == "Intro. Next part"

    def test_strips_and_collapses_spaces(self) -> None:
        assert preprocess_text("  a  b  ") == "a b"


class TestRecordToItem:
    def test_builds_lm_eval_style_context(self) -> None:
        item = record_to_item(RECORDS[0])
        assert item.context == "Baking: A person mixes flour and water. They"
        assert item.choices == ("good", "badbadbad")
        assert item.answer_index == 0

    def test_integer_label_accepted(self) -> None:
        record = dict(RECORDS[1])
        record["label"] = 1
        assert record_to_item(record).answer_index == 1

    def test_malformed_endings_raise(self) -> None:
        record = dict(RECORDS[0])
        record["endings"] = "not-a-list"
        with pytest.raises(DatasetLoadError, match="malformed"):
            record_to_item(record)


class TestHellaSwagEvaluator:
    def test_satisfies_evaluator_protocol(self) -> None:
        assert isinstance(make_evaluator(), Evaluator)

    def test_accuracy_and_normalized_accuracy_diverge(self, fake_runtime: FakeRuntime) -> None:
        # Item 1: raw LL picks "good" (correct); per-char normalization flips the
        # choice to the longer "badbadbad" (-2/9 > -1/4), making acc_norm wrong.
        # Item 2: both scorings pick "no" (correct).
        fake_runtime.completion_nlls = {
            " good": 1.0,
            " badbadbad": 2.0,
            " yes": 5.0,
            " no": 1.0,
        }
        result = make_evaluator().run(fake_runtime)

        assert result.name == "hellaswag"
        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.metrics["accuracy_norm"] == pytest.approx(0.5)
        assert result.metrics["items"] == 2
        assert result.duration_s >= 0

    def test_continuations_scored_with_leading_space(self, fake_runtime: FakeRuntime) -> None:
        make_evaluator().run(fake_runtime)
        continuations = {call[1] for call in fake_runtime.completion_calls}
        assert continuations == {" good", " badbadbad", " yes", " no"}

    def test_max_items_and_context_config_plumbed(self, fake_runtime: FakeRuntime) -> None:
        config = HellaSwagConfig(max_items=1, max_context_tokens=512)
        make_evaluator(config).run(fake_runtime)
        assert len({call[0] for call in fake_runtime.completion_calls}) == 1  # one item
        assert all(call[2] == 512 for call in fake_runtime.completion_calls)

    def test_items_loaded_once_across_sweep(self, fake_runtime: FakeRuntime) -> None:
        loads = 0

        def counting_loader(max_items: int | None) -> list[dict[str, object]]:
            nonlocal loads
            loads += 1
            return RECORDS

        evaluator = HellaSwagEvaluator(records_loader=counting_loader)
        evaluator.run(fake_runtime)
        evaluator.run(fake_runtime)
        assert loads == 1

    def test_empty_records_raise(self, fake_runtime: FakeRuntime) -> None:
        with pytest.raises(DatasetLoadError, match="no records"):
            make_evaluator(records=[]).run(fake_runtime)
