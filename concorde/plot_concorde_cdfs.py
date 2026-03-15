#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import sys
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CONCORDE_ROOT = ROOT / "ORT" / "concorde"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ort-concorde")
if str(CONCORDE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONCORDE_ROOT))

from src.analysis import ecdf_from_series, plot_cdf_bundle  # type: ignore  # noqa: E402


def load_series(artifact_dir: Path) -> dict[str, list[float]]:
    path = artifact_dir / "throughput_series.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_groups(series: dict[str, list[float]], include_branch_types: bool) -> OrderedDict[str, dict[str, list[float]]]:
    filtered = {
        name: values
        for name, values in series.items()
        if include_branch_types or not name.startswith("BR.TYPE.")
    }

    groups: OrderedDict[str, dict[str, list[float]]] = OrderedDict()
    groups["all"] = filtered
    groups["rob"] = {name: values for name, values in filtered.items() if name.startswith("ROB.")}
    groups["static"] = {name: values for name, values in filtered.items() if name.startswith("STATIC.")}
    groups["dynamic"] = {name: values for name, values in filtered.items() if name.startswith("DYN.")}
    return groups


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


def plot_single_series(series_name: str, values: list[float], title: str, png_path: Path, pdf_path: Path) -> None:
    x, y, y_weighted, meta = ecdf_from_series(values, drop_inf=True, drop_nan=True)
    if x.size == 0:
        return

    inf_ratio = meta["n_inf"] / meta["n_total"] if meta["n_total"] > 0 else 0.0

    plt.figure()
    plt.plot(x, y, label=f"CDF (inf={inf_ratio:.2%})", linewidth=2.0)
    plt.plot(x, y_weighted, linestyle="--", label="Weighted CDF", linewidth=2.0)
    plt.xlabel("Throughput (instr/cycle)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylim((0.0, 1.0))
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Concorde throughput CDFs from one artifact directory.")
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory containing throughput_series.json",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for generated PNG/PDF figures; defaults to <artifact-dir>/plots",
    )
    parser.add_argument(
        "--include-branch-types",
        action="store_true",
        help="Include BR.TYPE.* distributions in the combined plots",
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.9,
        help="Tail quantile threshold used for the zoomed figure",
    )
    parser.add_argument(
        "--separate-figs",
        action="store_true",
        help="Render one figure per series inside each group",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Render the original grouped plots instead of the default one-file-per-series layout",
    )
    parser.add_argument(
        "--show-tail",
        action="store_true",
        help="Also emit tail-zoom plots; default is disabled",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else artifact_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    series = load_series(artifact_dir)
    generated = []
    if args.grouped:
        groups = build_groups(series, include_branch_types=args.include_branch_types)
        for group_name, group_series in groups.items():
            if not group_series:
                continue
            png_path = output_dir / f"cdf_{group_name}.png"
            pdf_path = output_dir / f"cdf_{group_name}.pdf"
            plot_cdf_bundle(
                series_dict=group_series,
                title=f"Concorde CDFs: {artifact_dir.name} [{group_name}]",
                out_path_png=str(png_path),
                out_path_pdf=str(pdf_path),
                drop_inf=True,
                xlim=None,
                show_tail_zoom=args.show_tail,
                tail_quantile=args.tail_quantile,
                separate_figs=args.separate_figs,
            )
            generated.append(str(png_path))
            generated.append(str(pdf_path))
    else:
        filtered = {
            name: values
            for name, values in series.items()
            if args.include_branch_types or not name.startswith("BR.TYPE.")
        }
        for series_name, values in filtered.items():
            stem = sanitize_name(series_name)
            png_path = output_dir / f"cdf_{stem}.png"
            pdf_path = output_dir / f"cdf_{stem}.pdf"
            plot_single_series(
                series_name=series_name,
                values=values,
                title=f"{artifact_dir.name} - {series_name}",
                png_path=png_path,
                pdf_path=pdf_path,
            )
            generated.append(str(png_path))
            generated.append(str(pdf_path))

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
