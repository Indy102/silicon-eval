"""Tests for the runtime protocol module."""

from __future__ import annotations

import pytest

from silicon_eval.runtimes.base import Quantization, Runtime, parse_quant_list
from tests.conftest import FakeRuntime


class TestParseQuantList:
    def test_single_level(self) -> None:
        assert parse_quant_list("4bit") == [Quantization.Q4]

    def test_multiple_levels_preserve_order(self) -> None:
        assert parse_quant_list("8bit,4bit,fp16") == [
            Quantization.Q8,
            Quantization.Q4,
            Quantization.FP16,
        ]

    def test_whitespace_and_duplicates(self) -> None:
        assert parse_quant_list(" 4bit , 4bit ,8bit") == [Quantization.Q4, Quantization.Q8]

    def test_unknown_level_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown quantization '2bit'"):
            parse_quant_list("4bit,2bit")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="no quantization levels"):
            parse_quant_list(" , ")


def test_fake_runtime_satisfies_protocol(fake_runtime: FakeRuntime) -> None:
    assert isinstance(fake_runtime, Runtime)


def test_quantization_values_are_cli_names() -> None:
    assert [q.value for q in Quantization] == ["4bit", "8bit", "fp16", "bf16"]
