#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandwidth resources and dynamic constraints models.
Includes static bandwidth, pipes constraints, I-cache fills, and fetch buffers.
"""

import heapq
from collections import OrderedDict
from .config import get_config


# ============================================================
# Instruction classification for issue-width resources
# ============================================================

FP_MNEM_PREFIX = ("f", "fc", "fm", "fs", "fd")  # conservative heuristic for AArch64 FP/NEON mnemonics


def classify_issue_group(ins) -> str:
    """
    Classify an instruction into an issue group for static issue-width constraints.
      - LS: loads/stores
      - FP: floating-point / SIMD (heuristic)
      - ALU: everything else (default)
    
    Args:
        ins: Instruction object
        
    Returns:
        str: Issue group ('LS', 'FP', or 'ALU')
    """
    if ins.instr_type in ("load", "store"):
        return "LS"
    issue_group = getattr(ins, "issue_group", None)
    if issue_group in ("ALU", "FP", "LS"):
        return issue_group
    m = (ins.mnemonic or "").lower()
    if m.startswith(FP_MNEM_PREFIX):
        return "FP"
    return "ALU"


# ============================================================
# Static bandwidth throughput per k-window
# ============================================================

def static_bandwidth_throughputs(instrs_sorted, k: int):
    """
    Static bandwidth resources:
      - For global widths (fetch/decode/rename/commit): throughput bound is simply width.
      - For issue widths (ALU/FP/LS): only a subset of instructions consume that bandwidth.
        If n_g is number of group instructions in window, processing time is ceil(n_g / width_g),
        thus throughput bound = k / time.
    
    Args:
        instrs_sorted: List of sorted instructions
        k: Window size
        
    Returns:
        dict: Maps resource name to list of throughput values per window
    """
    config = get_config()
    widths = {
        "fetch_width": config.get('pipeline.fetch_width', 4),
        "decode_width": config.get('pipeline.decode_width', 4),
        "rename_width": config.get('pipeline.rename_width', 4),
        "commit_width": config.get('pipeline.commit_width', 8),
        "alu_issue_width": config.get('pipeline.issue_widths.alu', 3),
        "fp_issue_width": config.get('pipeline.issue_widths.fp', 2),
        "ls_issue_width": config.get('pipeline.issue_widths.ls', 2),
    }
    
    n = len(instrs_sorted)
    num_win = n // k
    out = {}

    # Global widths: all instructions consume bandwidth equally
    for name in ("fetch_width", "decode_width", "rename_width", "commit_width"):
        w = max(1, widths[name])
        out[name] = [w for _ in range(num_win)]

    # Issue widths: only subset of instructions consume bandwidth
    issue_cfg = [
        ("alu_issue_width", "ALU"),
        ("fp_issue_width",  "FP"),
        ("ls_issue_width",  "LS"),
    ]
    for width_name, grp in issue_cfg:
        w = max(1, widths[width_name])
        series = []
        for j in range(num_win):
            start = j * k
            end = (j + 1) * k
            window = instrs_sorted[start:end]
            n_grp = sum(1 for ins in window if classify_issue_group(ins) == grp)
            if n_grp == 0:
                thr = float('inf')
            else:
                time = (n_grp + w - 1) // w  # ceil(n_grp / w)
                thr = k / time if time > 0 else float('inf')
            series.append(thr)
        out[width_name] = series

    return out


# ============================================================
# Dynamic constraint: Load/Load-Store pipes bounds
# ============================================================

def pipes_throughput_bounds(instrs_sorted, k: int):
    """
    Dynamic constraints: finite Load-Store pipes (LSP) and Load pipes (LP).

    Lower bound (worst-case allocation):
      T_max = nLoad/(LSP+LP) + nStore/LSP
      thr_lower = k / T_max

    Upper bound (best-case schedule):
      - During store-issuing phase, stores occupy LSP for t_store = ceil(nStore/LSP)
      - During those cycles, loads issue on LP: issued_loads = t_store * LP
      - Remaining loads then issue with (LSP+LP) pipes.
      T_min = t_store + ceil(max(0, nLoad - t_store*LP) / (LSP+LP))
      thr_upper = k / T_min
    
    Args:
        instrs_sorted: List of sorted instructions
        k: Window size
        
    Returns:
        dict: Contains 'pipes_thr_lower' and 'pipes_thr_upper' lists
    """
    config = get_config()
    LSP = config.get('load_store_pipes.load_store_pipes', 2)
    LP = config.get('load_store_pipes.load_only_pipes', 10)
    
    n = len(instrs_sorted)
    num_win = n // k
    if num_win == 0:
        return {"pipes_thr_lower": [], "pipes_thr_upper": []}

    LSP = max(1, int(LSP))
    LP = max(0, int(LP))

    lower_series = []
    upper_series = []

    for j in range(num_win):
        start = j * k
        end = (j + 1) * k
        window = instrs_sorted[start:end]
        nLoad = sum(1 for ins in window if ins.instr_type == "load")
        nStore = sum(1 for ins in window if ins.instr_type == "store")
        
        # Lower bound (worst case)
        T_max = nLoad / (LSP + LP) + nStore / LSP if (LSP + LP) > 0 and LSP > 0 else float('inf')
        thr_lower = k / T_max if T_max > 0 else float('inf')
        lower_series.append(thr_lower)
        
        # Upper bound (best case)
        t_store = (nStore + LSP - 1) // LSP if LSP > 0 else 0
        issued_loads = t_store * LP
        remaining_loads = max(0, nLoad - issued_loads)
        t_remaining = (remaining_loads + LSP + LP - 1) // (LSP + LP) if (LSP + LP) > 0 else 0
        T_min = t_store + t_remaining
        thr_upper = k / T_min if T_min > 0 else float('inf')
        upper_series.append(thr_upper)

    return {"pipes_thr_lower": lower_series, "pipes_thr_upper": upper_series}


# ============================================================
# Dynamic constraint: I-cache fills simulation
# ============================================================

def icache_fills_throughput(instrs_sorted, k: int):
    """
    Dynamic constraints: Maximum I-cache fills + finite I-cache capacity with LRU replacement.

    Model:
      - I-cache is modeled as a set of cache lines with capacity = icache_size_bytes / line_size.
      - LRU replacement using OrderedDict: most-recently-used at the end.
      - A fill request is issued only if the line is not in-flight and not currently resident in I-cache.
      - In-flight fills limited by max_fills; each completes after fill_latency cycles.
      - On fill completion, the line is inserted into I-cache (may evict LRU line).
      - resp[i] is the earliest cycle instruction i's line is available (monotonic non-decreasing enforced).
      - Throughput per window: thr_j = k / (resp[end] - resp[start]).
    
    Args:
        instrs_sorted: List of sorted instructions
        k: Window size
        
    Returns:
        dict: Contains 'icache_fills_thr' list
    """
    n = len(instrs_sorted)
    num_win = n // k
    if n == 0 or num_win == 0:
        return {"icache_fills_thr": []}

    resp = icache_fills_resp_times(instrs_sorted)

    # Throughput over windows
    thr = []
    for j in range(1, num_win + 1):
        start_idx = (j - 1) * k
        end_idx = j * k
        delta = resp[end_idx] - resp[start_idx]
        if delta > 0:
            thr.append(k / delta)
        else:
            thr.append(float('inf'))

    return {"icache_fills_thr": thr}


def icache_fills_resp_times(instrs_sorted):
    """
    Return resp[i] (1..n): the cycle when instruction i's I-cache line becomes ready
    under a parallelized max in-flight fill model with fetch width.
    
    Dynamic constraints: Maximum I-cache fills + finite I-cache capacity with LRU replacement.
    
    Parallel fetch model:
      - Each cycle can process up to fetch_width instructions.
      - Instructions are fetched in-order; if an instruction blocks (needs fill but no slots),
        subsequent instructions in that cycle are not fetched.
      - Time advances cycle-by-cycle until all instructions are processed.
    
    Args:
        instrs_sorted: List of sorted instructions
        
    Returns:
        list: Response times for each instruction (1-indexed)
    """
    config = get_config()
    max_fills = config.get('icache.max_fills', 8)
    fill_latency = config.get('icache.fill_latency', 40)
    icache_size_bytes = config.get('icache.size_bytes', 4*1024)
    line_size = config.get('icache.line_size', 64)
    FETCH_WIDTH = config.get('icache.fetch_width', 8)
    
    n = len(instrs_sorted)
    max_fills = max(1, int(max_fills))
    fill_latency = max(1, int(fill_latency))
    line_size = max(1, int(line_size))
    icache_size_bytes = max(line_size, int(icache_size_bytes))
    capacity_lines = icache_size_bytes // line_size
    capacity_lines = max(1, capacity_lines)
    FETCH_WIDTH = max(1, int(FETCH_WIDTH))

    icache_lru = OrderedDict()
    inflight = {}
    heap = []
    cur_time = 0.0
    resp = [0.0] * (n + 1)
    instr_idx = 0  # Current instruction index (0-based)

    def icache_touch(line: int):
        icache_lru.move_to_end(line)

    def icache_insert(line: int):
        if line in icache_lru:
            icache_touch(line)
        else:
            if len(icache_lru) >= capacity_lines:
                icache_lru.popitem(last=False)
            icache_lru[line] = True

    def retire_completed_fills(upto_time: float):
        """Retire all fills that complete by upto_time."""
        while heap and heap[0][0] <= upto_time:
            comp_t, line = heapq.heappop(heap)
            inflight.pop(line, None)
            icache_insert(line)
    
    # Event-driven simulation: process instructions cycle by cycle
    while instr_idx < n:
        # Retire completed fills at the start of this cycle
        retire_completed_fills(cur_time)
        
        # Try to fetch up to FETCH_WIDTH instructions this cycle
        fetched_this_cycle = 0
        blocked = False
        
        while fetched_this_cycle < FETCH_WIDTH and instr_idx < n and not blocked:
            i = instr_idx + 1  # 1-based index for resp array
            ins = instrs_sorted[instr_idx]
            
            # Case 1: No cache line address - can fetch immediately
            if not hasattr(ins, 'I_cache_line_addr') or ins.I_cache_line_addr is None:
                resp[i] = cur_time
                instr_idx += 1
                fetched_this_cycle += 1
                continue
            
            line = ins.I_cache_line_addr
            
            # Case 2: Already in cache - can fetch immediately
            if line in icache_lru:
                icache_touch(line)
                resp[i] = cur_time
                instr_idx += 1
                fetched_this_cycle += 1
                continue
            
            # Case 3: Fill already in-flight - instruction waits for completion
            if line in inflight:
                resp[i] = inflight[line]
                instr_idx += 1
                fetched_this_cycle += 1
                continue
            
            # Case 4: Need to issue new fill
            if len(heap) < max_fills:
                # Can issue fill this cycle
                completion_time = cur_time + fill_latency
                heapq.heappush(heap, (completion_time, line))
                inflight[line] = completion_time
                resp[i] = completion_time
                instr_idx += 1
                fetched_this_cycle += 1
            else:
                # No available fill slots - block further fetch this cycle
                blocked = True
        
        # Advance time to next meaningful event
        if instr_idx < n:
            if blocked or fetched_this_cycle == 0:
                # Need to wait for a fill to complete
                if heap:
                    cur_time = heap[0][0]
                else:
                    # Should not happen if max_fills > 0
                    cur_time += 1
            else:
                # Made progress, advance by 1 cycle for next fetch opportunity
                cur_time += 1
    
    return resp


# ============================================================
# Dynamic constraint: Fetch buffers simulation
# ============================================================

def fetch_buffers_throughput(instrs_sorted, k: int,
                             fb_entries: int,
                             decode_width: int,
                             ready_time=None):
    """
    Dynamic constraint: limited fetch buffer capacity.

    Model:
      - Instruction i becomes eligible to enter fetch buffer at ready_time[i] (cycle).
        If ready_time is None, assumes ready_time[i] = 0 for all i.
      - Fetch buffer capacity = fb_entries (in instructions).
      - Decode consumes up to decode_width instructions per cycle (work-conserving).
      - If buffer is full when an instruction becomes ready, it waits until space exists.

    Output:
      - fb_decode_thr: throughput per k-window using decode completion time:
            thr_j = k / (t_dec[end] - t_dec[start])
    
    Args:
        instrs_sorted: List of sorted instructions
        k: Window size
        fb_entries: Fetch buffer size in entries
        decode_width: Decode width per cycle
        ready_time: List of ready times for each instruction (1-indexed)
        
    Returns:
        dict: Contains 'fb_decode_thr' list and 't_dec' array
    """
    n = len(instrs_sorted)
    if n == 0:
        return {"fb_decode_thr": [], "t_dec": [0]}

    fb_entries = max(1, int(fb_entries))
    decode_width = max(1, int(decode_width))
    if k <= 0:
        return {"fb_decode_thr": [], "t_dec": [0] * (n+1)}

    # ready_time[i] must be length n+1 (1..n)
    if ready_time is None:
        ready_time = [0.0] * (n + 1)
    if len(ready_time) != n + 1:
        ready_time = [0.0] * (n + 1)

    # t_dec[i] = cycle when instruction i is decoded (consumed from fetch buffer)
    t_dec = [0.0] * (n + 1)

    occ = 0                # fetch buffer occupancy (instructions)
    t = 0.0                # current cycle
    decoded_cnt = 0         # number of decoded instructions so far (1..decoded_cnt are completed)

    # helper: perform 1 decode cycle (consume up to decode_width)
    def do_one_cycle_decode(current_t: int):
        nonlocal occ, decoded_cnt
        to_decode = min(decode_width, occ)
        for _ in range(to_decode):
            decoded_cnt += 1
            t_dec[decoded_cnt] = current_t
        occ -= to_decode

    for i in range(1, n + 1):
        r = ready_time[i]
        
        # Advance time to ready or current time
        if r > t:
            # Decode cycles between t and r
            cycles = int(r - t)
            for _ in range(cycles):
                if occ > 0:
                    do_one_cycle_decode(t)
                    t += 1
                else:
                    t = r
                    break
            if t < r:
                t = r
        
        # Wait if buffer is full
        while occ >= fb_entries:
            do_one_cycle_decode(t)
            t += 1
        
        # Insert into fetch buffer
        occ += 1

    # drain remaining buffer after the last arrival
    while occ > 0:
        do_one_cycle_decode(t)
        t += 1

    # compute throughput per k-window using decode completion timestamps
    num_win = n // k
    thr = []
    for j in range(1, num_win + 1):
        start_idx = (j - 1) * k + 1
        end_idx = j * k
        delta = t_dec[end_idx] - t_dec[start_idx - 1] if start_idx > 1 else t_dec[end_idx]
        thr.append(k / delta if delta > 0 else float('inf'))

    return {"fb_decode_thr": thr, "t_dec": t_dec}
