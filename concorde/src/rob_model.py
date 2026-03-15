#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROB (Reorder Buffer) throughput model.
Implements Equations (1)-(5) from the paper.
"""

from .memory_state import build_exec_times_by_cache_line, MemoryStateMachine


def rob_throughput_model(instructions, ROB: int, k: int):
    """
    ROB throughput model implementing Equations (1)-(5).
    
    Equations:
    (1) a[i] = max(a[i-1], c[i-ROB])  # Arrival cycle
    (2) s[i] = max(a[i], max{f[j] : j ∈ dep[i]})  # Start cycle
    (3) f[i] = RespCycle(s[i], instr[i])  # Finish cycle
    (4) c[i] = max(c[i-1], f[i])  # Commit cycle
    (5) thr_j = k / (c[j*k] - c[(j-1)*k])  # Throughput for j-th window
    
    Args:
        instructions: List of instruction objects
        ROB: ROB size (number of entries)
        k: Window size for throughput calculation
        
    Returns:
        dict: Contains arrays a, s, f, c and throughput per window
    """
    instructions_sorted = sorted(instructions, key=lambda x: x.instr_id)
    n = len(instructions_sorted)
    
    if n == 0:
        return {"a": [0], "s": [0], "f": [0], "c": [0], "thr_chunks": [], "avg_ipc": 0.0, "instructions_sorted": []}

    # Build instruction ID to index mapping (1-indexed)
    id2idx = {ins.instr_id: i+1 for i, ins in enumerate(instructions_sorted)}

    # Build dependency index lists
    dep_idx = [[] for _ in range(n+1)]
    for i, ins in enumerate(instructions_sorted, start=1):
        if hasattr(ins, 'reg_deps'):
            dep_idx[i].extend(id2idx.get(dep_id, 0) for dep_id in ins.reg_deps if dep_id in id2idx)
        if hasattr(ins, 'mem_deps'):
            dep_idx[i].extend(id2idx.get(dep_id, 0) for dep_id in ins.mem_deps if dep_id in id2idx)

    # Build memory state machine
    exec_times_by_line = build_exec_times_by_cache_line(instructions_sorted)
    msm = MemoryStateMachine(exec_times_by_line)

    # Arrays for cycle tracking (1-indexed, 0-th element unused)
    a = [0] * (n+1)  # Arrival cycle
    s = [0] * (n+1)  # Start cycle
    f = [0] * (n+1)  # Finish cycle
    c = [0] * (n+1)  # Commit cycle

    for i in range(1, n+1):
        # Eq.(1): Arrival cycle
        a[i] = max(a[i-1], c[i-ROB] if i > ROB else 0)
        
        # Eq.(2): Start cycle (max of arrival and dependency finish times)
        dep_max = max([f[j] for j in dep_idx[i]], default=0)
        s[i] = max(a[i], dep_max)
        
        # Eq.(3): Finish cycle (using memory state machine)
        ins = instructions_sorted[i-1]
        f[i] = msm.resp_cycle(s[i], ins)
        
        # Eq.(4): Commit cycle
        c[i] = max(c[i-1], f[i])

    # Eq.(5): Throughput per k-window
    thr = []
    num_chunks = n // k
    for j in range(1, num_chunks + 1):
        start_idx = (j - 1) * k
        end_idx = j * k
        delta_c = c[end_idx] - c[start_idx]
        if delta_c > 0:
            thr.append(k / delta_c)
        else:
            thr.append(float('inf'))

    # Average IPC
    avg_ipc = (n / c[n]) if c[n] > 0 else 0.0
    
    return {
        "a": a, 
        "s": s, 
        "f": f, 
        "c": c, 
        "thr_chunks": thr, 
        "avg_ipc": avg_ipc, 
        "instructions_sorted": instructions_sorted
    }
