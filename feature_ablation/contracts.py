from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "latest" / "feature_ablation"

DEFAULT_VARIANT_MODE = "leave_one_out_plus_all"
SUPPORTED_VARIANT_MODES = (
    DEFAULT_VARIANT_MODE,
    "custom_only",
)
