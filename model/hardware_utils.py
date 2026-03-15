from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from model_utils import flatten_dict, safe_float


SIZE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>KiB|MiB|GiB|KB|MB|GB)\s*$")
FREQ_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)\s*$")
TIME_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ns|us|ms|s)\s*$")


def load_hardware_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_hardware_profile(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _size_to_bytes(text: str) -> float | None:
    match = SIZE_RE.match(text)
    if not match:
        return None
    value = float(match.group("value"))
    unit = match.group("unit")
    scale = {
        "KB": 1000.0,
        "MB": 1000.0**2,
        "GB": 1000.0**3,
        "KiB": 1024.0,
        "MiB": 1024.0**2,
        "GiB": 1024.0**3,
    }[unit]
    return value * scale


def _freq_to_ghz(text: str) -> float | None:
    match = FREQ_RE.match(text)
    if not match:
        return None
    value = float(match.group("value"))
    unit = match.group("unit")
    scale = {
        "Hz": 1e-9,
        "kHz": 1e-6,
        "MHz": 1e-3,
        "GHz": 1.0,
    }[unit]
    return value * scale


def _time_to_ns(text: str) -> float | None:
    match = TIME_RE.match(text)
    if not match:
        return None
    value = float(match.group("value"))
    unit = match.group("unit")
    scale = {
        "ns": 1.0,
        "us": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }[unit]
    return value * scale


def normalize_profile_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    size_bytes = _size_to_bytes(text)
    if size_bytes is not None:
        return size_bytes

    freq_ghz = _freq_to_ghz(text)
    if freq_ghz is not None:
        return freq_ghz

    time_ns = _time_to_ns(text)
    if time_ns is not None:
        return time_ns

    numeric = safe_float(text)
    if numeric is not None:
        return numeric

    return text


def flatten_hardware_features(profile: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_dict(profile)
    out: dict[str, Any] = {}
    for key, value in flat.items():
        if key.startswith("paper_cross_check") or key in {
            "profile_name",
            "source_config",
            "source_paper",
            "notes",
        }:
            continue
        normalized = normalize_profile_value(value)
        if isinstance(normalized, (int, float)) and normalized is not None:
            out[f"hw_{key}"] = normalized
    return out
