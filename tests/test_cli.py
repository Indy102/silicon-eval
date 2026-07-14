"""CLI tests using a fake runtime — no MLX required."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from silicon_eval import __version__, cli
from silicon_eval.runtimes.base import Quantization
from tests.conftest import FakeRuntime

runner = CliRunner()


@pytest.fixture
def injected_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    fake = FakeRuntime()
    monkeypatch.setattr(cli, "get_runtime", lambda name: fake)
    return fake


def test_version_flag() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_run_sweeps_all_quant_levels(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(
        cli.app, ["run", "--model", "mlx-community/TestModel", "--quant", "4bit,8bit"]
    )
    assert result.exit_code == 0
    assert [s.quantization for s in injected_runtime.loaded_specs] == [
        Quantization.Q4,
        Quantization.Q8,
    ]
    assert injected_runtime.unload_count == 2
    assert "4bit" in result.output
    assert "8bit" in result.output
    assert "42.5" in result.output  # generation tok/s from CANNED_METRICS


def test_run_show_text_prints_generation(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--quant", "4bit", "--show-text"])
    assert result.exit_code == 0
    assert "fake output" in result.output


def test_run_rejects_unknown_quant(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--quant", "3bit"])
    assert result.exit_code == 1
    assert "unknown quantization" in result.output
    assert injected_runtime.loaded_specs == []


def test_run_rejects_unknown_runtime() -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--runtime", "nope"])
    assert result.exit_code == 1
    assert "unknown runtime" in result.output


def test_run_requires_model() -> None:
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code != 0
