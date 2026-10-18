"""Rough hardware report to help decide whether a local LLM is practical.

Uses only the standard library. Detects total RAM and (NVIDIA only) GPU name/VRAM.
AMD/Intel/Apple GPUs are not detected here even though Ollama may use them.
The model-size guidance is a rule of thumb, not a guarantee.
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SystemInfo:
    os_name: str
    cpu_count: Optional[int]
    ram_gb: Optional[float]
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None


@dataclass
class Recommendation:
    tier: str                      # "mock-only" | "small" | "medium" | "large" | "unknown"
    summary: str
    suggested_models: List[str] = field(default_factory=list)


def _total_ram_bytes() -> Optional[int]:
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):   # type: ignore[attr-defined]
                return int(stat.ullTotalPhys)
            return None
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def parse_nvidia_smi(output: str) -> Optional[Tuple[str, float]]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`` (first GPU)."""
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                return parts[0], round(float(parts[1]) / 1024, 1)   # MiB -> GiB
            except ValueError:
                continue
    return None


def _detect_nvidia() -> Optional[Tuple[str, float]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(out)


def get_system_info() -> SystemInfo:
    ram = _total_ram_bytes()
    gpu = _detect_nvidia()
    return SystemInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu_count=os.cpu_count(),
        ram_gb=round(ram / 1024**3, 1) if ram else None,
        gpu_name=gpu[0] if gpu else None,
        vram_gb=gpu[1] if gpu else None,
    )


def recommend(info: SystemInfo) -> Recommendation:
    """Map hardware to a model-size class. Example tags: confirm current names at ollama.com/library."""
    if info.vram_gb is not None and info.vram_gb >= 6:
        return Recommendation(
            "large",
            f"NVIDIA GPU with ~{info.vram_gb} GB VRAM: 7-8B-parameter models should run at usable speed.",
            ["llama3.1:8b", "qwen2.5:7b", "llama3.2:3b (fast fallback)"],
        )
    if info.ram_gb is None:
        return Recommendation("unknown", "Could not detect RAM. Start with a ~3B model; use mock mode if it fails.",
                              ["llama3.2:3b"])
    if info.ram_gb >= 15:
        return Recommendation(
            "medium",
            f"{info.ram_gb} GB RAM, no usable GPU detected: 3B models are comfortable; 7-8B models run on CPU "
            "but slowly (expect long waits per answer).",
            ["llama3.2:3b", "qwen2.5:3b", "llama3.1:8b (slow on CPU)"],
        )
    if info.ram_gb >= 7:
        return Recommendation(
            "small",
            f"{info.ram_gb} GB RAM: stick to 1-3B models and close other heavy apps. Output quality will be modest.",
            ["llama3.2:3b", "llama3.2:1b"],
        )
    return Recommendation(
        "mock-only",
        f"{info.ram_gb} GB RAM is too little for a pleasant local LLM. Use TEST MODE (mock) for now, "
        "or try a 1B model and expect it to be slow.",
        ["llama3.2:1b (may struggle)"],
    )
