from __future__ import annotations

import subprocess
import time
from typing import Any, Callable


NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def _safe_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED", "[NOT SUPPORTED]"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _memory_usage_pct(used_mb: int | None, total_mb: int | None) -> float | None:
    if used_mb is None or total_mb is None or total_mb <= 0:
        return None
    return round(used_mb / total_mb * 100, 1)


def parse_nvidia_smi_output(output: str) -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue

        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue

        index = _safe_int(parts[0])
        total_mb = _safe_int(parts[2])
        used_mb = _safe_int(parts[3])
        utilization_pct = _safe_int(parts[4])
        temperature_c = _safe_int(parts[5])

        gpus.append({
            "index": index if index is not None else len(gpus),
            "name": parts[1],
            "memory_total_mb": total_mb,
            "memory_used_mb": used_mb,
            "memory_usage_pct": _memory_usage_pct(used_mb, total_mb),
            "utilization_pct": utilization_pct,
            "temperature_c": temperature_c,
        })

    return {
        "available": len(gpus) > 0,
        "source": "nvidia-smi",
        "message": None if gpus else "No GPU reported by nvidia-smi",
        "gpus": gpus,
        "updated_at": time.time(),
    }


def _torch_snapshot(torch_module: Any) -> dict[str, Any] | None:
    if torch_module is None or not torch_module.cuda.is_available():
        return None

    gpus = []
    for index in range(torch_module.cuda.device_count()):
        props = torch_module.cuda.get_device_properties(index)
        total_mb = round(props.total_memory / 1024 / 1024)
        reserved_mb = round(torch_module.cuda.memory_reserved(index) / 1024 / 1024)
        allocated_mb = round(torch_module.cuda.memory_allocated(index) / 1024 / 1024)
        used_mb = max(reserved_mb, allocated_mb)
        gpus.append({
            "index": index,
            "name": props.name,
            "memory_total_mb": total_mb,
            "memory_used_mb": used_mb,
            "memory_usage_pct": _memory_usage_pct(used_mb, total_mb),
            "utilization_pct": None,
            "temperature_c": None,
        })

    return {
        "available": len(gpus) > 0,
        "source": "torch",
        "message": "GPU utilization requires nvidia-smi; showing PyTorch memory stats only.",
        "gpus": gpus,
        "updated_at": time.time(),
    }


def get_gpu_snapshot(
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    torch_module: Any = None,
) -> dict[str, Any]:
    try:
        result = run_command(
            NVIDIA_SMI_QUERY,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        snapshot = parse_nvidia_smi_output(result.stdout)
        if snapshot["available"]:
            return snapshot
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:
            torch_module = None

    snapshot = _torch_snapshot(torch_module)
    if snapshot is not None:
        return snapshot

    return {
        "available": False,
        "source": "none",
        "message": "GPU monitor not available: nvidia-smi was not found and CUDA is unavailable.",
        "gpus": [],
        "updated_at": time.time(),
    }
