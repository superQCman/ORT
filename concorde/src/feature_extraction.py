#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction module for Concorde.

Assembles performance features (z) from analytical models and
architecture parameters (p) into a unified feature vector for ML model g(z, p) → CPI.

According to the Concorde paper (Figure 2):
  - Analytical models A_i(x, p_i) produce performance features z_i
  - Architecture-specific parameters p are appended
  - ML model g(z, p) predicts CPI
"""

import json
import numpy as np
from pathlib import Path

from .analysis import generate_cdf_vectors, branch_type_distribution
from .config import get_config


# ── Feature names for each analytical model output ──────────────────────

# CDF quantiles: default 50 quantiles (0.00..0.98 step 0.02)
# -> 50 uniform + 50 weighted + 1 mean-throughput scalar = 101
DEFAULT_QUANTILES = np.arange(0, 1, 0.02)
CDF_VECTOR_DIM = len(DEFAULT_QUANTILES) * 2 + 1   # 101

# All series names that produce CDF vectors
SERIES_NAMES = [
    "ROB.thr_chunks",
    # Static bandwidth
    "STATIC.fetch_width",
    "STATIC.decode_width",
    "STATIC.rename_width",
    "STATIC.commit_width",
    "STATIC.alu_issue_width",
    "STATIC.fp_issue_width",
    "STATIC.ls_issue_width",
    # Dynamic constraints
    "DYN.pipes_thr_lower",
    "DYN.pipes_thr_upper",
    "DYN.icache_fills_thr",
    "DYN.fb_decode_thr",
]

# Architecture parameter names
ARCH_PARAM_NAMES = [
    "rob.entries",
    "rob.window_size",
    "pipeline.fetch_width",
    "pipeline.decode_width",
    "pipeline.rename_width",
    "pipeline.commit_width",
    "pipeline.issue_widths.alu",
    "pipeline.issue_widths.fp",
    "pipeline.issue_widths.ls",
    "load_store_pipes.load_store_pipes",
    "load_store_pipes.load_only_pipes",
    "cache_hierarchy.l1.size_bytes",
    "cache_hierarchy.l1.associativity",
    "cache_hierarchy.l1.hit_latency",
    "cache_hierarchy.l2.size_bytes",
    "cache_hierarchy.l2.associativity",
    "cache_hierarchy.l2.hit_latency",
    "cache_hierarchy.l3.size_bytes",
    "cache_hierarchy.l3.associativity",
    "cache_hierarchy.l3.hit_latency",
    "cache_hierarchy.memory.latency",
    "icache.max_fills",
    "icache.fill_latency",
    "icache.size_bytes",
    "fetch_buffer.entries",
]


def build_all_series(rob_res, static_thr, pipes_thr, ic_thr, fb_res, br_type_dist):
    """
    Assemble all throughput series from analytical model outputs.
    
    Args:
        rob_res: Result from rob_throughput_model()
        static_thr: Result from static_bandwidth_throughputs()
        pipes_thr: Result from pipes_throughput_bounds()
        ic_thr: Result from icache_fills_throughput()
        fb_res: Result from fetch_buffers_throughput()
        br_type_dist: Result from branch_type_distribution()
        
    Returns:
        dict: Maps series names to value lists (for CDF generation)
    """
    all_series = {}
    
    # Branch type distribution per k-window (used for CDF but not in final series)
    for btype, counts in br_type_dist.items():
        all_series[f"BR.TYPE.{btype}"] = counts
    
    # ROB throughput
    if rob_res.get("thr_chunks"):
        all_series["ROB.thr_chunks"] = rob_res["thr_chunks"]
    
    # Static bandwidth
    for name, series in static_thr.items():
        all_series[f"STATIC.{name}"] = series
    
    # Dynamic constraints
    for name, series in pipes_thr.items():
        all_series[f"DYN.{name}"] = series
    for name, series in ic_thr.items():
        all_series[f"DYN.{name}"] = series
    for name, series in fb_res.items():
        if name == "t_dec":
            continue
        all_series[f"DYN.{name}"] = series
    
    return all_series


def extract_performance_features(all_series, br_tage_result, 
                                  quantiles=None):
    """
    Extract performance feature vector z from analytical model outputs.
    
    This produces the z = [z_1, z_2, ..., z_d] vector from the paper,
    where each z_i is a CDF quantile vector from analytical model A_i.
    
    Args:
        all_series: Dict of {name: [values]} from build_all_series()
        br_tage_result: Branch prediction result from compute_branch_mispred_rate()
        quantiles: Quantile array (default: 0.00 to 0.98 step 0.02)
        
    Returns:
        dict: Feature dict with:
            - 'cdf_vectors': dict of {name: np.array} CDF quantile vectors
            - 'branch_misp_rates': dict of {type: rate} branch misprediction
            - 'rob_avg_ipc': float, analytical model IPC prediction
            - 'feature_vector': np.array, concatenated feature vector z
            - 'feature_names': list of feature name strings
    """
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES
    
    # Remove branch type series from CDF generation (they're count distributions, not throughputs)
    cdf_series = {k: v for k, v in all_series.items() if not k.startswith("BR.TYPE.")}
    
    # Generate CDF vectors
    cdf_vectors = generate_cdf_vectors(cdf_series, drop_inf=True, drop_nan=True, quantiles=quantiles)
    
    # Branch misprediction rates (scalar features)
    br_misp_rates = {}
    if br_tage_result and br_tage_result.get('by_type'):
        for btype, data in br_tage_result['by_type'].items():
            br_misp_rates[btype] = data['misp_rate']
    # Overall misprediction rate
    if br_tage_result:
        br_misp_rates['overall'] = br_tage_result.get('misp_rate', 0.0)
    
    # Build the concatenated feature vector z
    feature_parts = []
    feature_names = []
    cdf_dim = len(quantiles) * 2 + 1  # uniform + weighted + mean
    
    # CDF vectors (ordered consistently)
    for series_name in SERIES_NAMES:
        if series_name in cdf_vectors:
            vec = cdf_vectors[series_name]
            feature_parts.append(vec)
            for qi in range(len(vec)):
                feature_names.append(f"{series_name}.q{qi}")
        else:
            # Pad with zeros if series not available
            feature_parts.append(np.zeros(cdf_dim))
            for qi in range(cdf_dim):
                feature_names.append(f"{series_name}.q{qi}")
    
    # Branch misprediction rates (scalar features, sorted for consistency)
    for btype in sorted(br_misp_rates.keys()):
        feature_parts.append(np.array([br_misp_rates[btype]]))
        feature_names.append(f"BR.TAGE.misp_rate.{btype}")
    
    feature_vector = np.concatenate(feature_parts).astype(np.float32)
    
    return {
        'cdf_vectors': cdf_vectors,
        'branch_misp_rates': br_misp_rates,
        'feature_vector': feature_vector,
        'feature_names': feature_names,
        'z_dim': len(feature_vector),
    }


def extract_arch_params(config=None):
    """
    Extract architecture-specific parameter vector p from config.
    
    Args:
        config: ArchConfig instance (uses global if None)
        
    Returns:
        dict: Contains 'param_vector' (np.array) and 'param_names' (list)
    """
    if config is None:
        config = get_config()
    
    values = []
    names = []
    
    for param_name in ARCH_PARAM_NAMES:
        val = config.get(param_name, 0)
        if val is None:
            val = 0
        values.append(float(val))
        names.append(f"ARCH.{param_name}")
    
    return {
        'param_vector': np.array(values, dtype=np.float32),
        'param_names': names,
        'p_dim': len(values),
    }


def build_ml_input(perf_features, arch_params):
    """
    Build the full ML model input vector by concatenating [z, p].
    
    According to Concorde paper: g(z, p) → CPI
    where z = performance features, p = architecture parameters.
    
    Args:
        perf_features: Result from extract_performance_features()
        arch_params: Result from extract_arch_params()
        
    Returns:
        dict: Contains:
            - 'input_vector': np.array of shape (z_dim + p_dim,)
            - 'input_names': list of feature names
            - 'z_dim': dimension of z
            - 'p_dim': dimension of p
            - 'total_dim': total input dimension
    """
    z = perf_features['feature_vector']
    p = arch_params['param_vector']
    
    input_vector = np.concatenate([z, p]).astype(np.float32)
    input_names = perf_features['feature_names'] + arch_params['param_names']
    
    return {
        'input_vector': input_vector,
        'input_names': input_names,
        'z_dim': len(z),
        'p_dim': len(p),
        'total_dim': len(input_vector),
    }


def save_features(ml_input, output_path, trace_name=None, ground_truth_cpi=None):
    """
    Save extracted features to a JSON file for later training/inference.
    
    Args:
        ml_input: Result from build_ml_input()
        output_path: Path to save features
        trace_name: Name identifier for this trace
        ground_truth_cpi: Ground truth CPI if available (for training)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'trace_name': trace_name or 'unknown',
        'input_vector': ml_input['input_vector'].tolist(),
        'input_names': ml_input['input_names'],
        'z_dim': ml_input['z_dim'],
        'p_dim': ml_input['p_dim'],
        'total_dim': ml_input['total_dim'],
    }
    if ground_truth_cpi is not None:
        data['ground_truth_cpi'] = float(ground_truth_cpi)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[FEATURE] Saved features to {output_path} (dim={ml_input['total_dim']})")


def load_features(feature_path):
    """
    Load features from a JSON file.
    
    Args:
        feature_path: Path to the feature JSON file
        
    Returns:
        dict: Feature data with numpy arrays
    """
    with open(feature_path, 'r') as f:
        data = json.load(f)
    
    data['input_vector'] = np.array(data['input_vector'], dtype=np.float32)
    return data


def load_training_dataset(feature_dir):
    """
    Load all feature files from a directory to build a training dataset.
    
    Args:
        feature_dir: Directory containing feature JSON files
        
    Returns:
        dict: Contains:
            - 'X': np.array of shape (N, total_dim)
            - 'y': np.array of shape (N,) CPI labels
            - 'names': list of trace names
            - 'input_names': feature names (from first file)
    """
    feature_dir = Path(feature_dir)
    feature_files = sorted(feature_dir.glob("*.json"))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {feature_dir}")
    
    X_list = []
    y_list = []
    names = []
    input_names = None
    
    for fp in feature_files:
        data = load_features(fp)
        
        if 'ground_truth_cpi' not in data:
            print(f"[WARNING] Skipping {fp.name}: no ground_truth_cpi")
            continue
        
        X_list.append(data['input_vector'])
        y_list.append(data['ground_truth_cpi'])
        names.append(data.get('trace_name', fp.stem))
        
        if input_names is None:
            input_names = data.get('input_names', [])
    
    if not X_list:
        raise ValueError("No valid training samples found")
    
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"[FEATURE] Loaded {len(X_list)} training samples, feature dim={X.shape[1]}")
    
    return {
        'X': X,
        'y': y,
        'names': names,
        'input_names': input_names or [],
    }
