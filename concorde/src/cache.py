#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache simulation models including private caches and shared LLC.
"""

import heapq
from collections import defaultdict, OrderedDict
from .config import get_config


class CacheLevel:
    """
    Simple set-associative cache with LRU.
    Inclusive hierarchy model is handled by calling lower levels on miss.
    """
    
    def __init__(self, name: str, size_bytes: int, assoc: int, line_size: int, hit_latency: int, lower=None):
        self.name = name
        self.size_bytes = size_bytes # Total cache size in bytes
        self.assoc = assoc
        self.line_size = line_size
        self.hit_latency = hit_latency
        self.lower = lower
        self.num_sets = size_bytes // (line_size * assoc)
        self.sets = [[] for _ in range(self.num_sets)]

    def _index_tag(self, addr: int):
        line = addr // self.line_size
        set_idx = line % self.num_sets
        tag = line // self.num_sets
        return set_idx, tag

    # Evict line if needed
    def _evict_if_needed(self, set_idx: int):
        s = self.sets[set_idx]
        if len(s) >= self.assoc:
            s.pop(0)

    def _insert_line(self, set_idx: int, tag: int, dirty: bool):
        self._evict_if_needed(set_idx)
        self.sets[set_idx].append((tag, dirty))

    # Touch line to update LRU 
    def _touch_line(self, set_idx: int, tag: int, dirty: bool):
        s = self.sets[set_idx]
        for i, (t, d) in enumerate(s):
            if t == tag:
                s.pop(i)
                s.append((tag, dirty or d))
                return

    def access(self, addr: int, is_write: bool, count_writeback_cost: bool = True):
        """
        Access cache and return total latency.
        
        Args:
            addr: Memory address
            is_write: True for write, False for read
            count_writeback_cost: Whether to count writeback cost
            
        Returns:
            tuple: (total_latency, hit_level_name)
        """
        set_idx, tag = self._index_tag(addr)
        s = self.sets[set_idx]
        
        for (t, d) in s:
            if t == tag:
                # Hit
                self._touch_line(set_idx, tag, is_write)
                return self.hit_latency, self.name
        
        # Miss
        if self.lower is not None:
            lower_lat, lower_level = self.lower.access(addr, is_write, count_writeback_cost)
        else:
            lower_lat, lower_level = 0, "MEM"
        
        self._insert_line(set_idx, tag, is_write)
        return self.hit_latency + lower_lat, lower_level


class MemoryLevel:
    """Memory level (bottom of hierarchy)."""
    
    def __init__(self, mem_latency: int):
        self.mem_latency = mem_latency

    def access(self, addr: int, is_write: bool, count_writeback_cost: bool = True):
        """Memory access always returns memory latency."""
        return self.mem_latency, "MEM"


def build_cache_hierarchy():
    """
    Build cache hierarchy from config.
    
    Returns:
        tuple: (l1, l2, l3, mem) cache levels
    """
    config = get_config()
    line_size = config.get('cache_hierarchy.line_size', 64)
    
    mem_latency = config.get('cache_hierarchy.memory.latency', 200)
    mem = MemoryLevel(mem_latency=mem_latency)
    
    l3_cfg = config.get('cache_hierarchy.l3', {})
    l3 = CacheLevel(
        name="L3",
        size_bytes=l3_cfg.get('size_bytes', 8*1024*1024),
        assoc=l3_cfg.get('associativity', 16),
        line_size=line_size,
        hit_latency=l3_cfg.get('hit_latency', 35),
        lower=mem
    )
    
    l2_cfg = config.get('cache_hierarchy.l2', {})
    l2 = CacheLevel(
        name="L2",
        size_bytes=l2_cfg.get('size_bytes', 512*1024),
        assoc=l2_cfg.get('associativity', 8),
        line_size=line_size,
        hit_latency=l2_cfg.get('hit_latency', 12),
        lower=l3
    )
    
    l1_cfg = config.get('cache_hierarchy.l1', {})
    l1 = CacheLevel(
        name="L1",
        size_bytes=l1_cfg.get('size_bytes', 32*1024),
        assoc=l1_cfg.get('associativity', 8),
        line_size=line_size,
        hit_latency=l1_cfg.get('hit_latency', 4),
        lower=l2
    )
    
    return l1, l2, l3, mem


class SharedLLCWithQueuing:
    """
    Shared Last-Level Cache with queuing model for multi-core contention.
    
    Features:
      - Finite MSHR entries (miss request buffer)
      - Bank conflicts (addresses hash to banks, serialized within bank)
      - LRU replacement with capacity pressure from all cores
      - Memory bandwidth queuing (M/M/c model for DRAM)
    """
    
    def __init__(self, size_bytes=8*1024*1024, assoc=16, line_size=64, hit_latency=35,
                 num_banks=16, mshr_entries=64, mem_latency=200, 
                 mem_bandwidth_gbps=100, num_mem_channels=4):
        self.size_bytes = size_bytes
        self.assoc = assoc
        self.line_size = line_size
        self.hit_latency = hit_latency
        self.num_banks = num_banks
        self.mshr_entries = mshr_entries
        self.mem_latency = mem_latency
        self.mem_bandwidth_gbps = mem_bandwidth_gbps
        self.num_mem_channels = num_mem_channels
        
        self.num_sets = size_bytes // (line_size * assoc)
        self.sets = [OrderedDict() for _ in range(self.num_sets)]  # LRU per set
        
        # Bank queues: list of (start_time, duration) for serialization
        self.bank_queues = [[] for _ in range(num_banks)]
        
        # MSHR tracking: in-flight misses
        self.mshrs = []  # heap of (completion_time, line_addr)
        self.inflight_lines = set()
        
        # Memory queue: list of (request_time, tid, line_addr)
        self.mem_queue = []
        self.mem_service_times = []  # list of (completion_time, tid)
        
        # Stats
        self.stats_hits = defaultdict(int)
        self.stats_misses = defaultdict(int)
        self.stats_bank_conflicts = 0
        self.stats_mshr_stalls = 0
        self.stats_mem_accesses = 0
    
    def _index_tag_bank(self, addr: int):
        """Compute set index, tag, and bank from address."""
        line = addr // self.line_size
        set_idx = line % self.num_sets
        tag = line // self.num_sets
        bank = line % self.num_banks
        return set_idx, tag, bank
    
    def _touch_line(self, set_idx: int, tag: int):
        """Touch line in LRU (move to end)."""
        self.sets[set_idx].move_to_end(tag)
    
    def _insert_line(self, set_idx: int, tag: int):
        """Insert line into set, evicting LRU if needed."""
        s = self.sets[set_idx]
        if len(s) >= self.assoc:
            s.popitem(last=False)  # Evict LRU
        s[tag] = True
    
    def _retire_completed_mshrs(self, current_time: int):
        """Remove completed MSHRs."""
        while self.mshrs and self.mshrs[0][0] <= current_time:
            _, line_addr = heapq.heappop(self.mshrs)
            self.inflight_lines.discard(line_addr)
    
    def _simulate_mem_access(self, request_time: int, tid: int, line_addr: int) -> int:
        """
        Simulate memory access with queuing delay.
        Uses simplified M/M/c queuing model.
        
        Returns:
            int: completion time
        """
        self.mem_queue.append((request_time, tid, line_addr))
        self.stats_mem_accesses += 1
        
        # Retire completed memory accesses
        self.mem_service_times = [(t, tid) for t, tid in self.mem_service_times if t > request_time]
        
        # Simple model: if queue empty, immediate service; else add queuing delay
        if len(self.mem_service_times) < self.num_mem_channels:
            completion = request_time + self.mem_latency
        else:
            # Queuing delay proportional to queue depth
            queue_depth = len(self.mem_service_times)
            extra_delay = queue_depth * 10  # Simplified contention model
            completion = request_time + self.mem_latency + extra_delay
        
        self.mem_service_times.append((completion, tid))
        return completion
    
    def access(self, addr: int, current_time: int, tid: int, is_write: bool = False):
        """
        Access shared LLC.
        
        Args:
            addr: Memory address
            current_time: Current simulation time
            tid: Thread ID
            is_write: Write access flag
            
        Returns:
            tuple: (latency, hit_level, completion_time)
        """
        set_idx, tag, bank = self._index_tag_bank(addr)
        line_addr = addr // self.line_size
        
        # Retire completed MSHRs
        self._retire_completed_mshrs(current_time)
        
        # Check for hit
        if tag in self.sets[set_idx]:
            self._touch_line(set_idx, tag)
            self.stats_hits[tid] += 1
            
            # Bank conflict check (simplified)
            bank_queue = self.bank_queues[bank]
            if bank_queue:
                last_end = bank_queue[-1][0] + bank_queue[-1][1]
                if last_end > current_time:
                    self.stats_bank_conflicts += 1
                    actual_start = last_end
                else:
                    actual_start = current_time
            else:
                actual_start = current_time
            
            bank_queue.append((actual_start, self.hit_latency))
            completion = actual_start + self.hit_latency
            return self.hit_latency, "LLC", completion
        
        # Miss
        self.stats_misses[tid] += 1
        
        # Check MSHR availability
        if len(self.mshrs) >= self.mshr_entries:
            self.stats_mshr_stalls += 1
            # Stall until oldest MSHR completes
            earliest_completion = self.mshrs[0][0]
            current_time = max(current_time, earliest_completion)
            self._retire_completed_mshrs(current_time)
        
        # Check if already in-flight
        if line_addr in self.inflight_lines:
            # Wait for in-flight fill
            for comp_time, line in self.mshrs:
                if line == line_addr:
                    return comp_time - current_time, "MEM", comp_time
        
        # Issue memory access
        mem_completion = self._simulate_mem_access(current_time, tid, line_addr)
        
        # Allocate MSHR
        heapq.heappush(self.mshrs, (mem_completion, line_addr))
        self.inflight_lines.add(line_addr)
        
        # Insert into cache on completion (simplified: assume instant)
        self._insert_line(set_idx, tag)
        
        return mem_completion - current_time, "MEM", mem_completion
    
    def print_stats(self):
        """Print LLC contention statistics."""
        print("\n=== Shared LLC Contention Stats ===")
        total_hits = sum(self.stats_hits.values())
        total_misses = sum(self.stats_misses.values())
        total_accesses = total_hits + total_misses
        
        print(f"Total Accesses: {total_accesses}")
        print(f"Total Hits: {total_hits} ({100*total_hits/total_accesses:.2f}%)" if total_accesses > 0 else "Total Hits: 0")
        print(f"Total Misses: {total_misses} ({100*total_misses/total_accesses:.2f}%)" if total_accesses > 0 else "Total Misses: 0")
        print(f"Bank Conflicts: {self.stats_bank_conflicts}")
        print(f"MSHR Stalls: {self.stats_mshr_stalls}")
        print(f"Memory Accesses: {self.stats_mem_accesses}")
        
        print("\nPer-thread breakdown:")
        for tid in sorted(set(list(self.stats_hits.keys()) + list(self.stats_misses.keys()))):
            hits = self.stats_hits[tid]
            misses = self.stats_misses[tid]
            total = hits + misses
            print(f"  Thread {tid}: {total} accesses, {hits} hits ({100*hits/total:.2f}%), {misses} misses" if total > 0 else f"  Thread {tid}: 0 accesses")
