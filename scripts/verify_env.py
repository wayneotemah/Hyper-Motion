#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import load_dotenv, resolve_path


def _bytes_to_gib(value: int) -> float:
    return round(value / float(1024 ** 3), 2)


def _disk_stats(path: Path) -> dict:
    stats = os.statvfs(path)
    total = stats.f_blocks * stats.f_frsize
    free = stats.f_bavail * stats.f_frsize
    return {"total_gib": _bytes_to_gib(total), "free_gib": _bytes_to_gib(free)}


def _total_ram_bytes() -> int:
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return 0


def _mac_free_ram_bytes() -> int:
    try:
        output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:
        return 0

    if not output:
        return 0

    page_size = 4096
    first_line = output[0]
    if "page size of" in first_line:
        try:
            page_size = int(first_line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
        except (IndexError, ValueError):
            page_size = 4096

    counters = {}
    for line in output[1:]:
        if ":" not in line:
            continue
        label, raw_value = line.split(":", 1)
        try:
            counters[label.strip()] = int(raw_value.strip().strip("."))
        except ValueError:
            continue

    available_pages = (
        counters.get("Pages free", 0)
        + counters.get("Pages speculative", 0)
        + counters.get("Pages inactive", 0)
        + counters.get("Pages purgeable", 0)
    )
    return available_pages * page_size


def _linux_free_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def _free_ram_bytes() -> int:
    system = platform.system()
    if system == "Darwin":
        return _mac_free_ram_bytes()
    if system == "Linux":
        return _linux_free_ram_bytes()
    return 0


def _import_probe(module_name: str) -> dict:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        return {"installed": True, "version": str(version)}
    except Exception as exc:
        return {"installed": False, "error": str(exc)}


def _torch_probe() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "error": str(exc), "mps_available": False, "cuda_available": False}

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    return {
        "installed": True,
        "version": torch.__version__,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
    }


def _classify(system: str, machine: str, total_ram_gib: float, torch_info: dict, disk_free_gib: float) -> str:
    if torch_info.get("cuda_available") and total_ram_gib >= 24:
        return "LOCAL_LIGHT_EXPERIMENT"
    if system == "Darwin" and machine == "arm64":
        if total_ram_gib <= 24:
            return "REMOTE_INFERENCE_REQUIRED"
        return "LOCAL_PREP_ONLY"
    if disk_free_gib < 60 or total_ram_gib < 16:
        return "REMOTE_INFERENCE_REQUIRED"
    return "LOCAL_PREP_ONLY"


def main() -> int:
    load_dotenv()
    disk = _disk_stats(PROJECT_ROOT)
    total_ram_gib = _bytes_to_gib(_total_ram_bytes())
    free_ram_gib = _bytes_to_gib(_free_ram_bytes())
    torch_info = _torch_probe()
    cv2_info = _import_probe("cv2")
    model_root = resolve_path(os.environ.get("HYPERMOTION_MODEL_ROOT"), PROJECT_ROOT) or (PROJECT_ROOT / "assets/model_store")

    result = {
        "os": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch": torch_info,
        "opencv": cv2_info,
        "mps_available": torch_info.get("mps_available", False),
        "cuda_available": torch_info.get("cuda_available", False),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "disk_free_gib": disk["free_gib"],
        "disk_total_gib": disk["total_gib"],
        "total_ram_gib": total_ram_gib,
        "free_ram_estimate_gib": free_ram_gib,
        "model_root": str(model_root.resolve()),
    }
    result["classification"] = _classify(
        result["system"],
        result["machine"],
        total_ram_gib,
        torch_info,
        disk["free_gib"],
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
