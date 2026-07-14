"""Collect the machine context a report was produced on."""

from __future__ import annotations

import platform
import subprocess

import psutil

from silicon_eval.report.schema import MachineInfo


def collect_machine_info() -> MachineInfo:
    return MachineInfo(
        chip=_chip_name(),
        memory_bytes=int(psutil.virtual_memory().total),
        os_version=_os_version(),
        python_version=platform.python_version(),
    )


def _chip_name() -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _os_version() -> str:
    mac_version = platform.mac_ver()[0]
    if mac_version:
        return f"macOS {mac_version}"
    return f"{platform.system()} {platform.release()}"
