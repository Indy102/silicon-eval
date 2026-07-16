"""CLI tests using a fake runtime — no MLX, no network, no sudo, tmp cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from silicon_eval import __version__, cli
from silicon_eval.evals import hellaswag, perplexity
from silicon_eval.runtimes.base import ModelSpec, Quantization
from tests.conftest import FakeRuntime, UnavailableEnergySampler

runner = CliRunner()

# With the fake runtime's uniform completion NLL (3.0), raw LL ties resolve to
# index 0 and per-char normalization favors the longer ending. Labels are set
# so accuracy_norm lands at exactly 1.0 (table shows "1.000").
HS_RECORDS: list[dict[str, object]] = [
    {
        "activity_label": "Baking",
        "ctx_a": "A person mixes flour.",
        "ctx_b": "they",
        "endings": ["bake it", "eat raw flour"],
        "label": "1",
    },
    {
        "activity_label": "Sports",
        "ctx_a": "A player lines up.",
        "ctx_b": "then",
        "endings": ["shoots", "sleeps"],
        "label": "0",
    },
]


@pytest.fixture
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No network, no sudo, per-test cache dir."""
    monkeypatch.setattr(perplexity, "load_wikitext2_text", lambda: "offline corpus")
    monkeypatch.setattr(
        hellaswag, "load_hellaswag_records", lambda max_items=None: HS_RECORDS[:max_items]
    )
    monkeypatch.setattr("silicon_eval.runner.PowerMetricsSampler", UnavailableEnergySampler)
    monkeypatch.setenv("SILICON_EVAL_CACHE_DIR", str(tmp_path / "result-cache"))


@pytest.fixture
def injected_runtime(offline_env: None, monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
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
    assert "7.39" in result.output  # perplexity from CANNED_SCORE
    assert "1.000" in result.output  # hellaswag acc_norm engineered by HS_RECORDS labels


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


def test_run_no_hellaswag_skips_completions(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--no-hellaswag"])
    assert result.exit_code == 0
    assert injected_runtime.completion_calls == []


def test_mmlu_off_by_default_and_flag_enables(
    injected_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from silicon_eval.evals import mmlu as mmlu_module

    mmlu_records: list[dict[str, object]] = [
        {"question": "Q?", "subject": "astronomy", "choices": ["a", "b", "c", "d"], "answer": 0}
    ]
    monkeypatch.setattr(
        mmlu_module, "load_mmlu_records", lambda max_items=None: mmlu_records[:max_items]
    )

    default = runner.invoke(cli.app, ["run", "--model", "m", "--no-hellaswag"])
    assert default.exit_code == 0
    assert "n/a" in default.output  # mmlu column absent by default
    assert injected_runtime.completion_calls == []

    enabled = runner.invoke(
        cli.app, ["run", "--model", "m", "--no-hellaswag", "--mmlu", "--mmlu-items", "1"]
    )
    assert enabled.exit_code == 0
    letters = {call[1] for call in injected_runtime.completion_calls}
    assert letters == {" A", " B", " C", " D"}
    assert "1.00" in enabled.output  # answer 0 wins the uniform-NLL tie → correct


def test_hs_items_flag_plumbed(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--hs-items", "1"])
    assert result.exit_code == 0
    contexts = {call[0] for call in injected_runtime.completion_calls}
    assert len(contexts) == 1  # only the first record scored


def test_energy_unavailable_note_printed(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m"])
    assert result.exit_code == 0
    assert "note: energy sampling unavailable" in result.output


def test_no_energy_flag_suppresses_note(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--no-energy"])
    assert result.exit_code == 0
    assert "energy sampling unavailable" not in result.output


def test_run_writes_json_report(injected_runtime: FakeRuntime, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(cli.app, ["run", "--model", "m", "--runs", "2", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert data["schema_version"] == 2
    assert data["variants"][0]["generation"]["runs"] == 2
    ppl = data["variants"][0]["evals"][0]["metrics"]["perplexity"]
    assert ppl == pytest.approx(7.389, rel=1e-3)
    assert data["variants"][0]["energy"] is None


def test_run_writes_markdown_report(injected_runtime: FakeRuntime, tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    result = runner.invoke(cli.app, ["run", "--model", "m", "--markdown", str(out)])
    assert result.exit_code == 0
    text = out.read_text()
    assert text.startswith("# silicon-eval report")
    assert "| 4bit" in text


def test_cache_hit_skips_measurement(injected_runtime: FakeRuntime) -> None:
    args = ["run", "--model", "m", "--quant", "4bit"]
    first = runner.invoke(cli.app, args)
    assert first.exit_code == 0
    assert len(injected_runtime.loaded_specs) == 1

    second = runner.invoke(cli.app, args)
    assert second.exit_code == 0
    assert "cached" in second.output
    assert len(injected_runtime.loaded_specs) == 1  # no new load
    assert "7.39" in second.output  # cached results still render


def test_no_cache_flag_recomputes(injected_runtime: FakeRuntime) -> None:
    args = ["run", "--model", "m", "--no-cache"]
    runner.invoke(cli.app, args)
    runner.invoke(cli.app, args)
    assert len(injected_runtime.loaded_specs) == 2


def test_no_cache_still_refreshes_the_cache(injected_runtime: FakeRuntime) -> None:
    # --no-cache means "re-measure", not "leave stale entries in place":
    # the refreshed result must serve the next default run.
    runner.invoke(cli.app, ["run", "--model", "m", "--no-cache"])
    result = runner.invoke(cli.app, ["run", "--model", "m"])
    assert result.exit_code == 0
    assert "cached" in result.output
    assert len(injected_runtime.loaded_specs) == 1


def test_cache_key_covers_config(injected_runtime: FakeRuntime) -> None:
    runner.invoke(cli.app, ["run", "--model", "m"])
    runner.invoke(cli.app, ["run", "--model", "m", "--max-tokens", "32"])
    assert len(injected_runtime.loaded_specs) == 2  # config change → cache miss


def test_poisoned_cache_entry_recomputed(injected_runtime: FakeRuntime, tmp_path: Path) -> None:
    first = runner.invoke(cli.app, ["run", "--model", "m"])
    assert first.exit_code == 0
    cache_dir = tmp_path / "result-cache"
    entries = list(cache_dir.glob("*.json"))
    assert entries
    for entry in entries:
        entry.write_text('{"bogus": 1}')  # valid JSON, wrong shape

    second = runner.invoke(cli.app, ["run", "--model", "m"])
    assert second.exit_code == 0
    assert len(injected_runtime.loaded_specs) == 2  # recomputed, not crashed


def test_cache_write_failure_never_costs_results(
    injected_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "cache-is-a-file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("SILICON_EVAL_CACHE_DIR", str(blocker))
    out = tmp_path / "report.json"
    result = runner.invoke(cli.app, ["run", "--model", "m", "--output", str(out)])

    assert result.exit_code == 0
    assert "could not write result cache" in result.output
    assert "42.5" in result.output  # table still printed
    assert out.exists()  # report still written


def test_stale_cached_energy_reason_is_labeled(injected_runtime: FakeRuntime) -> None:
    runner.invoke(cli.app, ["run", "--model", "m"])
    second = runner.invoke(cli.app, ["run", "--model", "m"])
    assert second.exit_code == 0
    assert "energy sampling unavailable" in second.output
    assert "from a cached result" in second.output
    assert "--no-cache" in second.output


def test_run_rejects_nonpositive_max_tokens(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--max-tokens", "-1"])
    assert result.exit_code == 1
    assert "--max-tokens must be >= 1" in result.output
    assert injected_runtime.loaded_specs == []


def test_run_rejects_negative_ppl_windows(injected_runtime: FakeRuntime) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--ppl-windows", "-1"])
    assert result.exit_code == 1
    assert "--ppl-windows must be >= 0" in result.output
    assert injected_runtime.loaded_specs == []  # rejected before any model load


def test_mid_sweep_failure_keeps_finished_variants(
    offline_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailsOnSecondLoad(FakeRuntime):
        def load(self, spec: ModelSpec) -> None:
            if self.unload_count >= 1:
                raise RuntimeError("out of memory")
            super().load(spec)

    fake = FailsOnSecondLoad()
    monkeypatch.setattr(cli, "get_runtime", lambda name: fake)
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


def test_run_rejects_unknown_runtime(offline_env: None) -> None:
    result = runner.invoke(cli.app, ["run", "--model", "m", "--runtime", "nope"])
    assert result.exit_code == 1
    assert "unknown runtime" in result.output


def test_run_requires_model() -> None:
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code != 0
