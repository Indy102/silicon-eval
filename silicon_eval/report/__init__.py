"""Report schema and renderers (JSON now, Markdown in Phase 3)."""

from silicon_eval.report.json_io import report_to_dict, report_to_json, write_report_json
from silicon_eval.report.machine import collect_machine_info
from silicon_eval.report.schema import (
    SCHEMA_VERSION,
    MachineInfo,
    Report,
    VariantResult,
)

__all__ = [
    "SCHEMA_VERSION",
    "MachineInfo",
    "Report",
    "VariantResult",
    "collect_machine_info",
    "report_to_dict",
    "report_to_json",
    "write_report_json",
]
