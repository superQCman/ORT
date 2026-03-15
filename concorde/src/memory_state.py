#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory state machine for simulating memory request/response cycles.
Implements Algorithm 1 from the paper.
"""

from collections import defaultdict


def build_exec_times_by_cache_line(instructions_sorted):
    """
    Build a mapping of cache line -> list of (exec_time, instr_id).
    
    Args:
        instructions_sorted: List of instructions sorted by instr_id
        
    Returns:
        dict: Mapping from cache_line_addr to list of (exec_time, instr_id)
    """
    exec_times = defaultdict(list)
    for ins in instructions_sorted:
        if ins.is_load:
            exec_times[ins.data_cache_line].append(int(ins.latency))
    return exec_times


class MemoryStateMachine:
    """
    Implements Algorithm 1: RespCycle(req_cycle, instr)
    """
    def __init__(self, exec_times_by_line):
        self.exec_times = exec_times_by_line
        self.access_counters = defaultdict(int)
        self.last_req_cycles = defaultdict(int)
        self.last_rsp_cycles = defaultdict(int)

    def resp_cycle(self, req_cycle: int, instr) -> int:
        if instr.is_load:
            cl = instr.load_address // instr.line_size  # Calculate cache line address  
            if req_cycle < self.last_req_cycles[cl]:
                req_cycle = self.last_req_cycles[cl]
            self.last_req_cycles[cl] = req_cycle

            prev_rsp = self.last_rsp_cycles[cl]
            access_num = self.access_counters[cl] # 获取当前缓存行访问次数

            et_list = self.exec_times.get(cl, [])
            exec_time = int(et_list[access_num]) if access_num < len(et_list) else int(instr.latency)

            rsp_cycle = max(req_cycle + exec_time, prev_rsp)

            self.last_rsp_cycles[cl] = rsp_cycle
            self.access_counters[cl] += 1
            return rsp_cycle

        return req_cycle + int(instr.latency)
