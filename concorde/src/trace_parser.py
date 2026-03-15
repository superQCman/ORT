#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace parsing functions for extracting instructions and statistics.
"""

from collections import Counter, defaultdict
from .utils import (
    RE_MARKER, RE_RECORD, RE_IFETCH_MNEM, RE_RW_DETAIL, RE_HEAD,
    RE_IFETCH_PC, RE_BRANCH_STATUS, BRANCH_MNEMS_EXACT,
    extract_uses_defs_from_ifetch, cache_lines_covered
)
from .instruction import Instruction, branchInstruction, Instr_Load, Instr_Store, Instr_NonMem
from .cache import build_cache_hierarchy, SharedLLCWithQueuing


def parse_trace(path: str):
    """
    Parse trace file and extract instructions with statistics.
    Uses per-core private cache hierarchy.
    
    Args:
        path: Path to trace file
        
    Returns:
        tuple: (record_types, markers, mnemonics, Instruction_classify_arr,
                Load_instr_list, cache_line_access, hit_level_counter, instrs_by_tid)
    """
    record_types = Counter()
    markers = Counter()
    mnemonics = Counter()
    Instruction_classify_arr = []
    branch_instr_classify = Counter()
    ls_instr_classify = Counter()
    Icache_instr_classify = Counter()
    Load_instr_list = []

    last_def_reg = {}
    ifetch_info = {}
    last_store_line = {}
    cache_line_access = {}

    instr_table = {}

    l1, l2, l3, mem = build_cache_hierarchy()
    hit_level_counter = Counter()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # Check for markers
            m_marker = RE_MARKER.search(line)
            if m_marker:
                markers[m_marker.group(1)] += 1
                continue

            # Parse record header
            m_head = RE_HEAD.match(line)
            if not m_head:
                continue

            rec_num = int(m_head.group("rec"))
            instr_id = int(m_head.group("instr"))
            tid = int(m_head.group("tid"))
            rtype = m_head.group("rtype")
            record_types[rtype] += 1

            # Process different record types
            if rtype == "ifetch":
                Icache_instr_classify["Instruction Fetch"] += 1
                
                # Extract mnemonic
                m_mnem = RE_IFETCH_MNEM.match(line)
                mnem = m_mnem.group(1) if m_mnem else None
                if mnem:
                    mnemonics[mnem] += 1

                # Extract PC
                m_pc = RE_IFETCH_PC.search(line)
                pc_val = int(m_pc.group(1), 16) if m_pc else 0

                # Extract uses/defs
                uses, defs = extract_uses_defs_from_ifetch(line)
                
                # Determine instruction type
                mnem_lower = (mnem or "").lower()
                is_branch = (
                    mnem_lower in BRANCH_MNEMS_EXACT or
                    mnem_lower.startswith("b.")
                )

                # Create instruction object
                line_size = 64
                I_cache_line = pc_val // line_size

                def classify_branch_type(mn: str) -> str:
                    mn = (mn or "").lower()
                    if mn.startswith("b."):
                        return "Direct Conditional Branch"
                    if mn in ("cbz", "cbnz", "tbz", "tbnz"):
                        # 也是 direct conditional（只是 encoding 不同）
                        return "Direct Conditional Branch"
                    if mn in ("b", "bl"):
                        return "Direct Unconditional Branch"
                    if mn in ("br", "blr", "ret"):
                        return "Indirect Branch"
                    return "Other Branch"
                
                if is_branch:
                    # Branch instruction
                    branch_type = classify_branch_type(mnem)
                    ins_obj = branchInstruction(branch_type, instr_id, mnem, I_cache_line, pc_val)
                    
                    # Check for taken/untaken status
                    m_status = RE_BRANCH_STATUS.search(line)
                    if m_status:
                        status = m_status.group(1)
                        ins_obj.branch_taken = (status == "taken")
                    
                    branch_instr_classify[branch_type] += 1
                else:
                    # Non-memory instruction (initially)
                    ins_obj = Instr_NonMem("non-memory", instr_id, mnem, I_cache_line)
                    ls_instr_classify["non-memory Instructions"] += 1

                # Register dependencies
                for u_reg in uses:
                    if u_reg in last_def_reg:
                        ins_obj.reg_deps.add(last_def_reg[u_reg])
                for d_reg in defs:
                    last_def_reg[d_reg] = instr_id

                # Store ifetch info for later
                ifetch_info[(tid, instr_id)] = {
                    "mnem": mnem,
                    "pc": pc_val,
                    "I_cache_line": I_cache_line,
                    "ins_obj": ins_obj
                }
                instr_table[(tid, instr_id)] = ins_obj

            elif rtype in ("read", "write"):
                # Memory access
                m_rw = RE_RW_DETAIL.search(line)
                if not m_rw:
                    continue

                addr = int(m_rw.group("addr"), 16)
                size = int(m_rw.group("size"))
                pc = int(m_rw.group("pc"), 16)

                # Find corresponding ifetch
                if (tid, instr_id) not in ifetch_info:
                    continue

                info = ifetch_info[(tid, instr_id)]
                ins_obj = info["ins_obj"]

                # Convert to load/store instruction
                line_size = 64
                I_cache_line = info["I_cache_line"]

                if rtype == "read":
                    # Access cache hierarchy
                    lat, hit_level = l1.access(addr, is_write=False)
                    hit_level_counter[hit_level] += 1

                    # Create or update load instruction
                    if not isinstance(ins_obj, Instr_Load):
                        new_ins = Instr_Load("load", instr_id, info["mnem"],
                                            addr, size, I_cache_line, 1,
                                            reg_deps=ins_obj.reg_deps, mem_deps=ins_obj.mem_deps)
                        new_ins.load_cache_hit_levels.append(hit_level)
                        ins_obj = new_ins
                        instr_table[(tid, instr_id)] = ins_obj
                        info["ins_obj"] = ins_obj
                        
                        # Update classification
                        ls_instr_classify["non-memory Instructions"] -= 1
                        ls_instr_classify["Load Instructions"] += 1
                        Load_instr_list.append(new_ins)
                    else:
                        ins_obj.load_access_count += 1
                        ins_obj.load_cache_hit_levels.append(hit_level)

                    # Track cache line dependencies
                    for cl in cache_lines_covered(addr, size, line_size):
                        if cl in last_store_line:
                            ins_obj.mem_deps.add(last_store_line[cl])
                        cache_line_access[cl] = instr_id

                elif rtype == "write":
                    # Access cache hierarchy
                    lat, hit_level = l1.access(addr, is_write=True)
                    hit_level_counter[hit_level] += 1

                    # Create or update store instruction
                    if not isinstance(ins_obj, Instr_Store):
                        new_ins = Instr_Store("store", instr_id, info["mnem"],
                                             addr, size, I_cache_line,
                                             reg_deps=ins_obj.reg_deps)
                        ins_obj = new_ins
                        instr_table[(tid, instr_id)] = ins_obj
                        info["ins_obj"] = ins_obj
                        
                        # Update classification
                        ls_instr_classify["non-memory Instructions"] -= 1
                        ls_instr_classify["Store Instructions"] += 1

                    # Track store to cache line
                    for cl in cache_lines_covered(addr, size, line_size):
                        last_store_line[cl] = instr_id

    # Finalize classification arrays
    Instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    Instruction_classify_arr.append(ls_instr_classify)
    Instruction_classify_arr.append(Icache_instr_classify)

    # Group instructions by thread
    instrs_by_tid = defaultdict(list)
    for (tid, iid), ins in instr_table.items():
        instrs_by_tid[tid].append(ins)
    for tid in instrs_by_tid:
        instrs_by_tid[tid].sort(key=lambda x: x.instr_id)

    return (
        record_types, markers, mnemonics, Instruction_classify_arr,
        Load_instr_list, cache_line_access, hit_level_counter,
        instrs_by_tid
    )


def parse_trace_with_shared_llc(path: str):
    """
    Parse trace file using shared LLC across all threads.
    Models multi-core contention at LLC and memory levels.
    
    Args:
        path: Path to trace file
        
    Returns:
        tuple: (record_types, markers, mnemonics, Instruction_classify_arr,
                Load_instr_list, cache_line_access, hit_level_counter, instrs_by_tid)
    """
    record_types = Counter()
    markers = Counter()
    mnemonics = Counter()
    Instruction_classify_arr = []
    branch_instr_classify = Counter()
    ls_instr_classify = Counter()
    Icache_instr_classify = Counter()
    Load_instr_list = []
    
    last_def_reg = {}
    ifetch_info = {}
    last_store_line = {}
    cache_line_access = {}
    instr_table = {}

    line_size = 64
    
    # Use shared LLC instead of per-core private caches
    shared_llc = SharedLLCWithQueuing(
        size_bytes=8*1024*1024,
        assoc=16,
        line_size=line_size,
        hit_latency=35,
        num_banks=16,
        mshr_entries=64,
        mem_latency=200,
        mem_bandwidth_gbps=100,
        num_mem_channels=4
    )
    
    # Private L1/L2 per thread (simplified: assume no contention here)
    thread_caches = {}  # tid -> (l1, l2)
    def get_thread_cache(tid):
        if tid not in thread_caches:
            from .cache import CacheLevel
            l2 = CacheLevel("L2", 512*1024, 8, line_size, 12, lower=None)
            l1 = CacheLevel("L1", 32*1024, 8, line_size, 4, lower=l2)
            thread_caches[tid] = (l1, l2)
        return thread_caches[tid]
    
    hit_level_counter = Counter()
    current_time = 0  # global cycle counter (simplified timeline)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # Check for markers
            m_marker = RE_MARKER.search(line)
            if m_marker:
                markers[m_marker.group(1)] += 1
                continue

            # Parse record header
            m_head = RE_HEAD.match(line)
            if not m_head:
                continue

            rec_num = int(m_head.group("rec"))
            instr_id = int(m_head.group("instr"))
            tid = int(m_head.group("tid"))
            rtype = m_head.group("rtype")
            record_types[rtype] += 1

            # Process different record types
            if rtype == "ifetch":
                Icache_instr_classify["Instruction Fetch"] += 1
                
                # Extract mnemonic
                m_mnem = RE_IFETCH_MNEM.match(line)
                mnem = m_mnem.group(1) if m_mnem else None
                if mnem:
                    mnemonics[mnem] += 1

                # Extract PC
                m_pc = RE_IFETCH_PC.search(line)
                pc_val = int(m_pc.group(1), 16) if m_pc else 0

                # Extract uses/defs
                uses, defs = extract_uses_defs_from_ifetch(line)
                
                # Determine instruction type
                mnem_lower = (mnem or "").lower()
                is_branch = (
                    mnem_lower in BRANCH_MNEMS_EXACT or
                    mnem_lower.startswith("b.")
                )

                # Create instruction object
                I_cache_line = pc_val // line_size

                if is_branch:
                    ins_obj = branchInstruction("branch", instr_id, mnem, I_cache_line, pc_val)
                    
                    # Check for taken/untaken status
                    m_status = RE_BRANCH_STATUS.search(line)
                    if m_status:
                        status = m_status.group(1)
                        ins_obj.branch_taken = (status == "taken")
                    
                    branch_instr_classify["Branch Instructions"] += 1
                else:
                    ins_obj = Instr_NonMem("non-memory", instr_id, mnem, I_cache_line)
                    ls_instr_classify["non-memory Instructions"] += 1

                # Register dependencies
                for u_reg in uses:
                    if u_reg in last_def_reg:
                        ins_obj.reg_deps.add(last_def_reg[u_reg])
                for d_reg in defs:
                    last_def_reg[d_reg] = instr_id

                # Store ifetch info
                ifetch_info[(tid, instr_id)] = {
                    "mnem": mnem,
                    "pc": pc_val,
                    "I_cache_line": I_cache_line,
                    "ins_obj": ins_obj
                }
                instr_table[(tid, instr_id)] = ins_obj

            elif rtype in ("read", "write"):
                # Memory access
                m_rw = RE_RW_DETAIL.search(line)
                if not m_rw:
                    continue

                addr = int(m_rw.group("addr"), 16)
                size = int(m_rw.group("size"))
                pc = int(m_rw.group("pc"), 16)

                # Find corresponding ifetch
                if (tid, instr_id) not in ifetch_info:
                    continue

                info = ifetch_info[(tid, instr_id)]
                ins_obj = info["ins_obj"]
                I_cache_line = info["I_cache_line"]

                # Get thread's private caches
                l1, l2 = get_thread_cache(tid)

                if rtype == "read":
                    # Check L1
                    l1_lat, l1_hit = l1.access(addr, is_write=False, count_writeback_cost=False)
                    if l1_hit == "L1":
                        hit_level = "L1"
                        hit_level_counter[hit_level] += 1
                    else:
                        # Check L2
                        l2_lat, l2_hit = l2.access(addr, is_write=False, count_writeback_cost=False)
                        if l2_hit == "L2":
                            hit_level = "L2"
                            hit_level_counter[hit_level] += 1
                        else:
                            # Access shared LLC
                            llc_lat, llc_hit, comp_time = shared_llc.access(addr, current_time, tid, is_write=False)
                            hit_level = llc_hit
                            hit_level_counter[hit_level] += 1
                            current_time = comp_time

                    # Create or update load instruction
                    if not isinstance(ins_obj, Instr_Load):
                        new_ins = Instr_Load("load", instr_id, info["mnem"],
                                            addr, size, I_cache_line, 1,
                                            reg_deps=ins_obj.reg_deps, mem_deps=ins_obj.mem_deps, line_size=line_size)
                        new_ins.load_cache_hit_levels.append(hit_level)
                        ins_obj = new_ins
                        instr_table[(tid, instr_id)] = ins_obj
                        info["ins_obj"] = ins_obj
                        
                        ls_instr_classify["non-memory Instructions"] -= 1
                        ls_instr_classify["Load Instructions"] += 1
                        Load_instr_list.append(new_ins)
                    else:
                        ins_obj.load_access_count += 1
                        ins_obj.load_cache_hit_levels.append(hit_level)

                    # Track cache line dependencies
                    for cl in cache_lines_covered(addr, size, line_size):
                        if cl in last_store_line:
                            ins_obj.mem_deps.add(last_store_line[cl])
                        cache_line_access[cl] = instr_id

                elif rtype == "write":
                    # Check L1
                    l1_lat, l1_hit = l1.access(addr, is_write=True, count_writeback_cost=False)
                    if l1_hit == "L1":
                        hit_level = "L1"
                        hit_level_counter[hit_level] += 1
                    else:
                        # Check L2
                        l2_lat, l2_hit = l2.access(addr, is_write=True, count_writeback_cost=False)
                        if l2_hit == "L2":
                            hit_level = "L2"
                            hit_level_counter[hit_level] += 1
                        else:
                            # Access shared LLC
                            llc_lat, llc_hit, comp_time = shared_llc.access(addr, current_time, tid, is_write=True)
                            hit_level = llc_hit
                            hit_level_counter[hit_level] += 1
                            current_time = comp_time

                    # Create or update store instruction
                    if not isinstance(ins_obj, Instr_Store):
                        new_ins = Instr_Store("store", instr_id, info["mnem"],
                                             addr, size, I_cache_line,
                                             reg_deps=ins_obj.reg_deps)
                        ins_obj = new_ins
                        instr_table[(tid, instr_id)] = ins_obj
                        info["ins_obj"] = ins_obj
                        
                        ls_instr_classify["non-memory Instructions"] -= 1
                        ls_instr_classify["Store Instructions"] += 1

                    # Track store to cache line
                    for cl in cache_lines_covered(addr, size, line_size):
                        last_store_line[cl] = instr_id

    # Finalize classification arrays
    Instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    Instruction_classify_arr.append(ls_instr_classify)
    Instruction_classify_arr.append(Icache_instr_classify)

    # Group instructions by thread
    instrs_by_tid = defaultdict(list)
    for (tid, iid), ins in instr_table.items():
        instrs_by_tid[tid].append(ins)
    for tid in instrs_by_tid:
        instrs_by_tid[tid].sort(key=lambda x: x.instr_id)

    # Print LLC contention stats
    shared_llc.print_stats()

    return (
        record_types, markers, mnemonics, Instruction_classify_arr,
        Load_instr_list, cache_line_access, hit_level_counter,
        instrs_by_tid
    )
