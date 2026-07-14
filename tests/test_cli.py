"""CLI tests using a fake runtime — no MLX and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from silicon_eval import __version__, cli
from silicon_eval.evals import perplexity
from silicon_eval.runtimes.base import ModelSpec, Quantization
from tests.conftest import FakeRuntime

runner = CliRunner()


@pytest.fixture
def injected_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    fake = FakeRuntime()
    monkeypatch.setattr(cli, "get_runtime", lambda name: fake)
    # Keep the default perplexity eval off the network.
    monkeypatch.setattr(perplexity, "load_wikitext2_text", lambda: "offline corpus")
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
    assert "7.39" in result.output  # perplexity from CANNED_SCORE


def test_run_perplexity_uses_flags(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(
        cli.app,
        ["run", "--model", "m", "--ppl-windows", "0", "--ppl-context", "256"],
    )
    assert result.exit_code == 0
    assert injected_runtime.scored_texts == ["offline corpus"]
    assert injected_runtime.score_args == [(256, None)]  # 0 windows → full corpus


def test_run_no_perplexity_skips_scoring(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--no-perplexity"])
    assert result.exit_code == 0
    assert injected_runtime.scored_texts == []
    assert "n/a" in result.output  # ppl column


def test_run_writes_json_report(injected_runtime: FakeRuntime, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(cli.app, ["run", "--model", "m", "--runs", "2", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert data["schema_version"] == 1
    assert data["variants"][0]["generation"]["runs"] == 2
    ppl = data["variants"][0]["evals"][0]["metrics"]["perplexity"]
    assert ppl == pytest.approx(7.389, rel=1e-3)


def test_run_rejects_negative_ppl_windows(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--ppl-windows", "-1"])
    assert result.exit_code == 1
    assert "--ppl-windows must be >= 0" in result.output
    assert injected_runtime.loaded_specs == []  # rejected before any model load


def test_mid_sweep_failure_keeps_finished_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailsOnSecondLoad(FakeRuntime):
        def load(self, spec: ModelSpec) -> None:
            if self.unload_count >= 1:
                raise RuntimeError("out of memory")
            super().load(spec)

    fake = FailsOnSecondLoad()
    monkeypatch.setattr(cli, "get_runtime", lambda name: fake)
    monkeypatch.setattr(perplexity, "load_wikitext2_text", lambda: "offline corpus")
    out = tmp_path / "partial.json"
    result = runner.invoke(
        cli.app, ["run", "--model", "m", "--quant", "4bit,8bit", "--output", str(out)]
    )

    assert result.exit_code == 1
    assert "42.5" in result.output  # 4bit row still printed
    assert "m @ 8bit: out of memory" in result.output
    data = json.loads(out.read_text())  # partial report still written
    assert [v["quantization"] for v in data["variants"]] == ["4bit"]


def test_eval_crash_reports_clean_error(
    injected_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dead_network() -> str:
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(perplexity, "load_wikitext2_text", dead_network)
    result = runner.invoke(cli.app, ["run", "--model", "m"])
    assert result.exit_code == 1
    assert "hub unreachable" in result.output
    assert injected_runtime.unload_count == 1  # model still released


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
