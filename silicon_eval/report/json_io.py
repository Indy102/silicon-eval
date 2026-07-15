"""JSON (de)serialization of reports and variant results."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from silicon_eval.evals.base import EvalResult
from silicon_eval.profiling.generation import GenerationProfile
from silicon_eval.profiling.stats import Stat
from silicon_eval.report.schema import EnergyProfile, Report, VariantResult
from silicon_eval.runtimes.base import Quantization


def report_to_dict(report: Report) -> dict[str, Any]:
    """Report as JSON-ready nested dicts (dataclasses.asdict)."""
    return dataclasses.asdict(report)


def report_to_json(report: Report) -> str:
    """Report as a pretty-printed strict-JSON string."""
    # allow_nan=False: a NaN/Infinity metric is a measurement bug — fail loudly
    # rather than emit JSON that standard parsers reject.
    return json.dumps(report_to_dict(report), indent=2, allow_nan=False) + "\n"


def write_report_json(report: Report, path: Path) -> None:
    """Serialize ``report`` to ``path`` as JSON."""
    path.write_text(report_to_json(report), encoding="utf-8")


def variant_to_dict(variant: VariantResult) -> dict[str, Any]:
    """Variant as JSON-ready nested dicts (inverse of variant_from_dict)."""
    return dataclasses.asdict(variant)


def variant_from_dict(data: dict[str, Any]) -> VariantResult:
    """Rebuild a VariantResult from its ``variant_to_dict`` form (cache reads)."""
    energy = data.get("energy")
    return VariantResult(
        model_id=data["model_id"],
        quantization=Quantization(data["quantization"]),
        runtime=data["runtime"],
        generation=_generation_from_dict(data["generation"]),
        evals=[EvalResult(**entry) for entry in data["evals"]],
        peak_rss_bytes=data["peak_rss_bytes"],
        energy=EnergyProfile(**energy) if energy is not None else None,
        energy_unavailable_reason=data.get("energy_unavailable_reason"),
        backend_versions=data.get("backend_versions", {}),
    )


def _generation_from_dict(data: dict[str, Any]) -> GenerationProfile:
    return GenerationProfile(
        runs=data["runs"],
        warmup_runs=data["warmup_runs"],
        max_tokens=data["max_tokens"],
        ttft_s=Stat(**data["ttft_s"]),
        prompt_tps=Stat(**data["prompt_tps"]),
        generation_tps=Stat(**data["generation_tps"]),
        peak_metal_bytes=data["peak_metal_bytes"],
    )
