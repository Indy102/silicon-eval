"""Tests for the content-addressed result cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from silicon_eval.cache import ResultCache, cache_key, default_cache_dir
from silicon_eval.report.json_io import variant_from_dict, variant_to_dict
from silicon_eval.report.schema import EnergyProfile
from silicon_eval.runner import run_variant
from silicon_eval.runtimes.base import ModelSpec, Quantization
from tests.conftest import FakeRuntime


class TestCacheKey:
    def test_order_insensitive(self) -> None:
        assert cache_key({"a": 1, "b": [1, 2]}) == cache_key({"b": [1, 2], "a": 1})

    def test_value_sensitive(self) -> None:
        assert cache_key({"quant": "4bit"}) != cache_key({"quant": "8bit"})

    def test_nested_payloads_hash_stably(self) -> None:
        payload = {"machine": {"chip": "Apple M1", "memory_bytes": 8}, "runs": 3}
        assert cache_key(payload) == cache_key(json.loads(json.dumps(payload)))


class TestDefaultCacheDir:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("SILICON_EVAL_CACHE_DIR", str(tmp_path / "custom"))
        assert default_cache_dir() == tmp_path / "custom"

    def test_default_under_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SILICON_EVAL_CACHE_DIR", raising=False)
        assert default_cache_dir() == Path.home() / ".cache" / "silicon-eval"


class TestResultCache:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        assert ResultCache(tmp_path).get("deadbeef") is None

    def test_put_get_round_trip(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "nested" / "dir")
        cache.put("k1", {"value": 42})
        assert cache.get("k1") == {"value": 42}

    def test_corrupt_entry_is_a_miss(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path)
        cache.put("k1", {"value": 42})
        (tmp_path / "k1.json").write_text("{not json", encoding="utf-8")
        assert cache.get("k1") is None

    def test_non_dict_entry_is_a_miss(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path)
        (tmp_path / "k1.json").write_text("[1, 2]", encoding="utf-8")
        assert cache.get("k1") is None


class TestVariantRoundTrip:
    def test_variant_survives_dict_json_dict(self, fake_runtime: FakeRuntime) -> None:
        spec = ModelSpec(model_id="some/model", quantization=Quantization.Q8)
        variant = run_variant(fake_runtime, spec, prompt="p", runs=1, warmup=0)
        rebuilt = variant_from_dict(json.loads(json.dumps(variant_to_dict(variant))))
        assert rebuilt == variant

    def test_energy_profile_survives_round_trip(self, fake_runtime: FakeRuntime) -> None:
        import dataclasses

        spec = ModelSpec(model_id="m", quantization=Quantization.Q4)
        variant = run_variant(fake_runtime, spec, prompt="p", runs=1, warmup=0)
        with_energy = dataclasses.replace(
            variant,
            energy=EnergyProfile(
                mean_power_mw=2500.0,
                energy_per_generated_token_mj=39.1,
                generated_tokens=128,
                samples=10,
                duration_s=2.0,
            ),
        )
        rebuilt = variant_from_dict(json.loads(json.dumps(variant_to_dict(with_energy))))
        assert rebuilt == with_energy
