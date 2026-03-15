from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


COMBO_RE = re.compile(r"(bs(?P<batch>\d+)_nip(?P<nip>\d+))")
TRACE_OP_RE = re.compile(r"^(?P<op_idx>\d{5})_")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_int(value: Any) -> int | None:
    out = safe_float(value)
    if out is None:
        return None
    return int(out)


def infer_combo_from_text(text: str) -> str | None:
    match = COMBO_RE.search(text)
    if not match:
        return None
    return match.group(1)


def infer_combo_from_path(path: Path) -> str | None:
    for part in [*path.parts, path.stem]:
        combo = infer_combo_from_text(str(part))
        if combo:
            return combo
    return None


def split_combo(combo: str | None) -> tuple[int | None, int | None]:
    if not combo:
        return None, None
    match = COMBO_RE.search(combo)
    if not match:
        return None, None
    return int(match.group("batch")), int(match.group("nip"))


def parse_trace_op_idx(value: str | None) -> int | None:
    if not value:
        return None
    match = TRACE_OP_RE.match(str(value).strip())
    if not match:
        return None
    return int(match.group("op_idx"))


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, prefix=flat_key))
        else:
            out[flat_key] = value
    return out


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
