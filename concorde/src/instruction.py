#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instruction data structures for trace analysis.
"""


class Instruction:
    """Base instruction class."""

    __slots__ = ("instr_type", "instr_id", "mnemonic", "I_cache_line_addr", "reg_deps", "mem_deps", "issue_group")

    def __init__(self, instr_type: str, instr_id: int, mnemonic: str, I_cache_line_addr: int = None):
        self.instr_type = instr_type
        self.instr_id = instr_id
        self.mnemonic = mnemonic
        self.I_cache_line_addr = I_cache_line_addr
        self.reg_deps = set()
        self.mem_deps = set()
        self.issue_group = None

    @property
    def latency(self):
        """Base execution latency."""
        return 1
    
    @property
    def is_load(self) -> bool:
        return self.instr_type == "load"


class branchInstruction(Instruction):
    """Branch instruction."""

    __slots__ = ("branch_instr_pc", "branch_taken")

    def __init__(self, instr_type: str, instr_id: int, mnemonic: str,
                 I_cache_line_addr: int, pc: int):
        super().__init__(instr_type, instr_id, mnemonic, I_cache_line_addr)
        self.branch_instr_pc = pc
        self.branch_taken = None  # Will be set to True/False during trace parsing


class Instr_Load(Instruction):
    """Load instruction."""

    __slots__ = (
        "load_address",
        "load_size",
        "load_access_count",
        "load_cache_hit_levels",
        "line_size",
        "data_cache_line",
    )

    def __init__(self, instr_type: str, instr_id: int, mnemonic: str,
                 load_address: int, load_size: int, I_cache_line_addr: int, load_access_count: int = 0,
                 reg_deps=None, mem_deps=None, line_size: int = 64):
        super().__init__(instr_type, instr_id, mnemonic, I_cache_line_addr)
        self.load_address = load_address
        self.load_size = load_size
        self.load_access_count = load_access_count
        self.load_cache_hit_levels = []
        if reg_deps:
            self.reg_deps = reg_deps
        if mem_deps:
            self.mem_deps = mem_deps
        self.line_size = line_size
        self.data_cache_line = self.load_address // self.line_size
    @property
    def latency(self):
        """Load latency based on cache hits."""
        if not self.load_cache_hit_levels:
            return 200  # Memory latency
        level = self.load_cache_hit_levels[0]  # First access determines latency
        latencies = {"L1": 4, "L2": 12, "L3": 35, "MEM": 200}
        return latencies.get(level, 200)


class Instr_Store(Instruction):
    """Store instruction."""

    __slots__ = ("store_address", "store_size", "line_size", "data_cache_line")

    def __init__(self, instr_type: str, instr_id: int, mnemonic: str,
                 store_address: int, store_size: int, I_cache_line_addr: int, reg_deps=None, line_size: int = 64):
        super().__init__(instr_type, instr_id, mnemonic, I_cache_line_addr)
        self.store_address = store_address
        self.store_size = store_size
        if reg_deps:
            self.reg_deps = reg_deps
        self.line_size = line_size
        self.data_cache_line = self.store_address // self.line_size


class Instr_NonMem(Instruction):
    """Non-memory instruction (ALU, FP, etc.)."""

    __slots__ = ()

    def __init__(self, instr_type: str, instr_id: int, mnemonic: str, I_cache_line_addr: int, reg_deps=None):
        super().__init__(instr_type, instr_id, mnemonic, I_cache_line_addr)
        if reg_deps:
            self.reg_deps = reg_deps
