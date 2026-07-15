"""Report data model. Serialized to JSON via :mod:`silicon_eval.report.json_io`."""

from __future__ import annotations

from dataclasses import dataclass, field

from silicon_eval.evals.base import EvalResult
from silicon_eval.profiling.generation import GenerationProfile
from silicon_eval.runtimes.base import Quantization

SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class EnergyProfile:
    """System-wide power/energy sampled over dedicated generation runs.

    powermetrics reports whole-machine CPU+GPU+ANE power, so readings include
    baseline load. ``energy_per_generated_token_mj`` divides the session's
    energy by the tokens generated during it.
    """

    mean_power_mw: float
    energy_per_generated_token_mj: float
    generated_tokens: int
    samples: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class MachineInfo:
    """The hardware/software environment measurements were taken on."""

    chip: str | None
    memory_bytes: int
    os_version: str
    python_version: str


@dataclass(frozen=True, slots=True)
class VariantResult:
    """Everything measured for one model variant.

    ``generation.peak_metal_bytes`` is the accelerator-side peak since this
    variant's model load (weights + inference buffers). ``peak_rss_bytes`` is
    the host-side process RSS peak spanning load + profiling + evals; on macOS
    it may undercount GPU-wired memory, so treat the Metal figure as
    authoritative for unified-memory pressure.
    """

    model_id: str
    quantization: Quantization
    runtime: str
    generation: GenerationProfile
    evals: list[EvalResult]
    peak_rss_bytes: int | None
    energy: EnergyProfile | None = None
    energy_unavailable_reason: str | None = None
    backend_versions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Report:
    """Top-level report: one run of silicon-eval over one or more variants."""

    schema_version: int
    silicon_eval_version: str
    created_at: str
    machine: MachineInfo
    variants: list[VariantResult]
