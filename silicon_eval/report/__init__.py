"""Report schema and renderers (JSON and Markdown)."""

from silicon_eval.report.json_io import (
    report_to_dict,
    report_to_json,
    variant_from_dict,
    variant_to_dict,
    write_report_json,
)
from silicon_eval.report.machine import collect_machine_info
from silicon_eval.report.markdown import render_markdown, write_report_markdown
from silicon_eval.report.schema import (
    SCHEMA_VERSION,
    EnergyProfile,
    MachineInfo,
    Report,
    VariantResult,
)

__all__ = [
    "SCHEMA_VERSION",
    "EnergyProfile",
    "MachineInfo",
    "Report",
    "VariantResult",
    "collect_machine_info",
    "render_markdown",
    "report_to_dict",
    "report_to_json",
    "variant_from_dict",
    "variant_to_dict",
    "write_report_json",
    "write_report_markdown",
]
