"""Tests for report serialization and machine info collection."""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest

from silicon_eval.report import machine
from silicon_eval.report.json_io import report_to_json, write_report_json
from silicon_eval.runner import build_report, run_variant
from silicon_eval.runtimes.base import ModelSpec, Quantization
from tests.conftest import FakeRuntime


def make_report(fake_runtime: FakeRuntime) -> tuple[dict[str, Any], str]:
    spec = ModelSpec(model_id="some/model", quantization=Quantization.Q4)
    variant = run_variant(fake_runtime, spec, prompt="p", runs=1, warmup=0)
    text = report_to_json(build_report([variant]))
    return json.loads(text), text


def test_json_round_trip(fake_runtime: FakeRuntime) -> None:
    data, text = make_report(fake_runtime)

    assert text.endswith("\n")
    assert data["schema_version"] == 1
    variant = data["variants"][0]
    assert variant["quantization"] == "4bit"  # StrEnum serializes as its value
    assert variant["runtime"] == "fake"
    assert variant["generation"]["ttft_s"]["mean"] == pytest.approx(0.35)
    assert variant["generation"]["peak_metal_bytes"] == 512 * 1024 * 1024
    assert variant["peak_rss_bytes"] > 0


def test_nan_metric_rejected(fake_runtime: FakeRuntime) -> None:
    spec = ModelSpec(model_id="m", quantization=Quantization.Q4)
    variant = run_variant(fake_runtime, spec, prompt="p", runs=1, warmup=0)
    broken = dataclasses.replace(
        variant.generation, ttft_s=dataclasses.replace(variant.generation.ttft_s, mean=float("nan"))
    )
    report = build_report([dataclasses.replace(variant, generation=broken)])
    with pytest.raises(ValueError):
        report_to_json(report)


def test_write_report_json(fake_runtime: FakeRuntime, tmp_path: Path) -> None:
    spec = ModelSpec(model_id="m", quantization=Quantization.FP16)
    report = build_report([run_variant(fake_runtime, spec, prompt="p", runs=1, warmup=0)])
    out = tmp_path / "report.json"
    write_report_json(report, out)
    assert json.loads(out.read_text())["variants"][0]["quantization"] == "fp16"


class TestMachineInfo:
    def test_collects_basics(self) -> None:
        info = machine.collect_machine_info()
        assert info.memory_bytes > 0
        assert info.python_version == platform.python_version()
        assert info.os_version

    def test_chip_is_none_off_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("silicon_eval.report.machine.platform.system", lambda: "Linux")
        assert machine._chip_name() is None

    def test_chip_is_none_when_sysctl_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("silicon_eval.report.machine.platform.system", lambda: "Darwin")

        def broken_run(*args: object, **kwargs: object) -> None:
            raise subprocess.CalledProcessError(1, "sysctl")

        monkeypatch.setattr("silicon_eval.report.machine.subprocess.run", broken_run)
        assert machine._chip_name() is None
