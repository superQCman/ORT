#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concorde trace analysis package.

Modular architecture for analyzing processor traces with:
- Cache hierarchy simulation
- ROB throughput modeling
- Bandwidth resource analysis
- Branch prediction
- Dynamic constraint modeling
"""

__version__ = "1.0.0"

# Core modules
from .config import ArchConfig, init_config, get_config
from .instruction import (
    Instruction, branchInstruction, Instr_Load, Instr_Store, Instr_NonMem
)
from .cache import CacheLevel, MemoryLevel, build_cache_hierarchy, SharedLLCWithQueuing
from .memory_state import MemoryStateMachine, build_exec_times_by_cache_line
from .rob_model import rob_throughput_model
from .bandwidth import (
    classify_issue_group,
    static_bandwidth_throughputs,
    pipes_throughput_bounds,
    icache_fills_throughput,
    icache_fills_resp_times,
    fetch_buffers_throughput
)
from .branch_prediction import (
    SimplePredictor, TAGEPredictor, classify_branch_type,
    compute_branch_mispred_rate
)
from .trace_parser import parse_trace, parse_trace_with_shared_llc
from .analysis import (
    summarize_thr_series, ecdf_from_series, cdf_to_vectors,
    generate_cdf_vectors, plot_cdf_bundle
)
from .utils import (
    extract_uses_defs_from_ifetch, cache_lines_covered, print_top,
    RE_MARKER, RE_RECORD, RE_IFETCH_MNEM, RE_RW_DETAIL, RE_HEAD,
    RE_IFETCH_PC, RE_REG, RE_BRANCH_STATUS, BRANCH_MNEMS_EXACT
)

__all__ = [
    # Config
    'ArchConfig', 'init_config', 'get_config',
    
    # Instructions
    'Instruction', 'branchInstruction', 'Instr_Load', 'Instr_Store', 'Instr_NonMem',
    
    # Cache
    'CacheLevel', 'MemoryLevel', 'build_cache_hierarchy', 'SharedLLCWithQueuing',
    
    # Memory state
    'MemoryStateMachine', 'build_exec_times_by_cache_line',
    
    # ROB model
    'rob_throughput_model',
    
    # Bandwidth
    'classify_issue_group', 'static_bandwidth_throughputs',
    'pipes_throughput_bounds', 'icache_fills_throughput',
    'icache_fills_resp_times', 'fetch_buffers_throughput',
    
    # Branch prediction
    'SimplePredictor', 'TAGEPredictor', 'classify_branch_type',
    'compute_branch_mispred_rate',
    
    # Trace parsing
    'parse_trace', 'parse_trace_with_shared_llc',
    
    # Analysis
    'summarize_thr_series', 'ecdf_from_series', 'cdf_to_vectors',
    'generate_cdf_vectors', 'plot_cdf_bundle',
    
    # Utils
    'extract_uses_defs_from_ifetch', 'cache_lines_covered', 'print_top',
]
