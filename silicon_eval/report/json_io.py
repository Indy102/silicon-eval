"""JSON serialization of reports."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from silicon_eval.report.schema import Report


def report_to_dict(report: Report) -> dict[str, Any]:
    return dataclasses.asdict(report)


def report_to_json(report: Report) -> str:
    # allow_nan=False: a NaN/Infinity metric is a measurement bug — fail loudly
    # rather than emit JSON that standard parsers reject.
    return json.dumps(report_to_dict(report), indent=2, allow_nan=False) + "\n"


def write_report_json(report: Report, path: Path) -> None:
    path.write_text(report_to_json(report), encoding="utf-8")
