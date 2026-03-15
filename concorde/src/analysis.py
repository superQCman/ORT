#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis and plotting utilities for throughput data.
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import BRANCH_TYPES


def summarize_thr_series(name: str, thr_list, max_show: int = 10):
    """
    Print summary statistics for throughput series.
    
    Args:
        name: Series name
        thr_list: List of throughput values
        max_show: Maximum number of initial values to display
    """
    finite = [x for x in thr_list if x != float("inf") and x > 0]
    if not finite:
        print(f"[THR] {name}: all values are inf or non-positive")
        return
    finite_sorted = sorted(finite)
    p50 = finite_sorted[len(finite_sorted)//2]
    p10 = finite_sorted[int(0.10 * (len(finite_sorted)-1))]
    p90 = finite_sorted[int(0.90 * (len(finite_sorted)-1))]
    print(f"[THR] {name}: samples={len(thr_list)} finite={len(finite)} p10={p10:.6f} p50={p50:.6f} p90={p90:.6f} first={thr_list[:max_show]}")


def branch_type_distribution(instrs, k):
    """
    Analyze branch instruction types distribution at every k instructions.

    Args:
        instrs: List of instruction objects
        k: Window size for analysis

    Returns:
        List contains dicts of branch type counts per k-window
        {type 1:[], type 2:[], ...}
    """
    branch_type_distribution = {}
    for i in range(0, len(instrs), k):
        window = instrs[i:i+k]
        type_count = {}
        for ins in window:
            if ins.instr_type in BRANCH_TYPES:
                btype = ins.instr_type
                type_count[btype] = type_count.get(btype, 0) + 1
        for btype in BRANCH_TYPES:
            if btype not in type_count:
                type_count[btype] = 0
        for btype, count in type_count.items():
            if btype not in branch_type_distribution:
                branch_type_distribution[btype] = []
            branch_type_distribution[btype].append(count)
    return branch_type_distribution


def ecdf_from_series(series, drop_inf=True, drop_nan=True):
    """
    Build empirical CDF from a throughput series.
    
    Args:
        series: List of values
        drop_inf: Drop infinite values
        drop_nan: Drop NaN values
        
    Returns:
        tuple: (x_sorted, y_cdf, meta_dict)
    """
    arr = np.array(series, dtype=float)

    meta = {
        "n_total": len(arr),
        "n_inf": int(np.isinf(arr).sum()),
        "n_nan": int(np.isnan(arr).sum()),
        "n_finite": int(np.isfinite(arr).sum()),
    }

    if drop_nan:
        arr = arr[~np.isnan(arr)]
    if drop_inf:
        arr = arr[~np.isinf(arr)]

    if arr.size == 0:
        return np.array([]), np.array([]), np.array([]), meta

    x = np.sort(arr)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    x_convert = 1/x
    x_sum = np.sum(x_convert)
    w = x_convert / x_sum if x_sum > 0 else np.ones_like(x_convert) / x_convert.size
    y_w = np.cumsum(w)
    return x, y, y_w, meta


def cdf_to_vectors(x, y, y_weighted, quantiles=np.arange(0, 1, 0.02)): 
    """
    Convert CDF (x,y) to quantile vectors.
    
    Args:
        x: X values (sorted)
        y: Y values (CDF)
        quantiles: Array-like of quantiles in [0,1)
        
    Returns:
        array: X values at specified quantiles
        array: Weighted X values at specified quantiles
    """
    quantiles = np.asarray(quantiles)
    x_at_q = np.interp(quantiles, y, x, left=x[0], right=x[-1])
    x_at_q_weighted = np.interp(quantiles, y_weighted, x, left=x[0], right=x[-1])
    # The final scalar feature should stay in throughput units.
    # Using mean(y) collapses every series to ~0.5 regardless of the underlying data.
    x_mean = np.mean(x)
    return x_at_q, x_at_q_weighted, x_mean


def generate_cdf_vectors(series_dict, drop_inf=True, drop_nan=True,
                         quantiles=np.arange(0, 1, 0.02)):
    """
    Generate CDF quantile vectors from a throughput series.
    
    Args:
        series_dict: Dict mapping names to series
        drop_inf: Drop infinite values
        drop_nan: Drop NaN values
        quantiles: Array of quantiles
        
    Returns:
        dict: Quantile vectors for each series
    """
    cdf_vectors = {}
    for name, series in series_dict.items():
        x, y, y_weighted, meta = ecdf_from_series(series, drop_inf=drop_inf, drop_nan=drop_nan)
        if x.size > 0:
            x_at_q, x_at_q_weighted, x_mean = cdf_to_vectors(x, y, y_weighted, quantiles)
            cdf_vectors[name] = np.concatenate([x_at_q, x_at_q_weighted, [x_mean]])
            print(f"vector size for {name}: {cdf_vectors[name].size}")
    return cdf_vectors


def plot_cdf_bundle(series_dict, title, out_path_png=None, out_path_pdf=None,
                    drop_inf=True, xlim=None, show_tail_zoom=False,
                    tail_quantile=0.9, separate_figs=False):   
    """
    Plot multiple CDF curves in one figure.
    series_dict: {name: [thr samples]}
    drop_inf: remove inf samples from CDF, annotate inf ratio in legend.
    xlim: (xmin, xmax) for main plot.
    show_tail_zoom: if True, create a second figure zoomed into tail (>= tail_quantile).
    """
    # Main figure
    plt.figure()
    for name, series in series_dict.items():
        if separate_figs:
            plt.figure()
        x, y, y_weighted, meta = ecdf_from_series(series, drop_inf=drop_inf, drop_nan=True)
        if x.size == 0:
            continue
        inf_ratio = meta["n_inf"] / meta["n_total"] if meta["n_total"] > 0 else 0.0
        label = f"{name} (inf={inf_ratio:.2%})" if drop_inf else name
        if separate_figs:
            label_weighted = f"{name} (inf={inf_ratio:.2%}, weighted)" if drop_inf else name + " (weighted)"
        plt.plot(x, y, label=label)
        if separate_figs:
            plt.plot(x, y_weighted, linestyle='--', label=label_weighted)
        if separate_figs:
            plt.xlabel("Throughput (instr/cycle)")
            plt.ylabel("CDF")
            plt.title(f"{title} - {name}")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.ylim((0.0, 1.0))
            if xlim is not None:
                plt.xlim(xlim)
            plt.legend(fontsize=8)
            if out_path_png:
                out_path_png_name = out_path_png.replace(".png", f"_{name.replace(' ', '_')}.png")
                plt.tight_layout()
                plt.savefig(out_path_png_name, dpi=200)
            if out_path_pdf:
                out_path_pdf_name = out_path_pdf.replace(".pdf", f"_{name.replace(' ', '_')}.pdf")
                plt.tight_layout()
                plt.savefig(out_path_pdf_name)
    if not separate_figs:
        plt.xlabel("Throughput (instr/cycle)")
        plt.ylabel("CDF")
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        if xlim is not None:
            plt.xlim(xlim)
        plt.ylim((0.0, 1.0))
        plt.legend(fontsize=8)

        if out_path_png:
            plt.tight_layout()
            plt.savefig(out_path_png, dpi=200)
        if out_path_pdf:
            plt.tight_layout()
            plt.savefig(out_path_pdf)

    # Optional tail zoom (Concorde often emphasizes tail behavior)
    if show_tail_zoom:
        plt.figure()
        for name, series in series_dict.items():
            x, y, y_weighted, meta = ecdf_from_series(series, drop_inf=drop_inf, drop_nan=True)
            if x.size == 0:
                continue
            # zoom into tail region
            mask = y >= tail_quantile
            if not np.any(mask):
                continue
            inf_ratio = meta["n_inf"] / meta["n_total"] if meta["n_total"] > 0 else 0.0
            label = f"{name} (inf={inf_ratio:.2%})" if drop_inf else name
            label_weighted = f"{name} (inf={inf_ratio:.2%}, weighted)" if drop_inf else name + " (weighted)"
            plt.plot(x[mask], y[mask], label=label)
            plt.plot(x[mask], y_weighted[mask], linestyle='--', label=label_weighted)
        plt.xlabel("Throughput (instr/cycle)")
        plt.ylabel("CDF")
        plt.title(f"{title} (tail zoom, CDF≥{tail_quantile})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.ylim((tail_quantile, 1.0))
        plt.legend(fontsize=8)

        if out_path_png:
            tail_png = out_path_png.replace(".png", f"_tail_q{int(tail_quantile*100)}.png")
            plt.tight_layout()
            plt.savefig(tail_png, dpi=200)
        if out_path_pdf:
            tail_pdf = out_path_pdf.replace(".pdf", f"_tail_q{int(tail_quantile*100)}.pdf")
            plt.tight_layout()
            plt.savefig(tail_pdf)
