#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions and constants for trace analysis.
"""

import re

# ============================================================
# Regex patterns
# ============================================================
RE_MARKER = re.compile(r"<marker:\s*([^>]+)>")
RE_RECORD = re.compile(r"^\s*\d+\s+\d+:\s+\d+\s+([a-zA-Z_]+)\b")  # ifetch/read/write/...
RE_IFETCH_MNEM = re.compile(
    r"""^\s*\d+\s+\d+:\s+\d+\s+ifetch\s+\d+\s+byte\(s\)\s+@\s+0x[0-9a-fA-F]+
         \s+[0-9a-fA-F]+\s+([a-zA-Z.][a-zA-Z0-9_.]*)\b""",
    re.VERBOSE
)
RE_RW_DETAIL = re.compile(
    r"\b(?P<op>read|write)\s+(?P<size>\d+)\s+byte\(s\)\s+@\s+0x(?P<addr>[0-9a-fA-F]+)\s+by\s+PC\s+0x(?P<pc>[0-9a-fA-F]+)"
)
RE_HEAD = re.compile(r"^\s*(?P<rec>\d+)\s+(?P<instr>\d+):\s+(?P<tid>\d+)\s+(?P<rtype>\S+)\b")
RE_IFETCH_PC = re.compile(r"\bifetch\s+\d+\s+byte\(s\)\s+@\s+0x([0-9a-fA-F]+)")

RE_REG = re.compile(r"%(?:x|w)\d+|%sp|%fp|%lr|%xzr|%wzr", re.IGNORECASE)
RE_BRANCH_STATUS = re.compile(r"\((taken|untaken)\)\s*$")
RE_THREAD_ID = re.compile(r'^\s*\d+\s+\d+:\s+(\d+)\b')

# ============================================================
# Branch instruction sets
# ============================================================
BRANCH_MNEMS_EXACT = {
    "b", "bl", "br", "blr", "ret",
    "cbz", "cbnz", "tbz", "tbnz",
}

BRANCH_TYPES = {
    "Direct Unconditional Branch",
    "Direct Conditional Branch",
    "Indirect Branch",
}


# ============================================================
# Helper functions
# ============================================================
def extract_uses_defs_from_ifetch(line: str):
    """
    Extract register uses and definitions from ifetch line.
    
    Args:
        line: ifetch line string
        
    Returns:
        tuple: (uses, defs) - sets of register names
    """
    parts = line.split("->", 1)  # 1表示只分割一次
    left = parts[0]
    right = parts[1] if len(parts) == 2 else ""
    uses = set(RE_REG.findall(left))  # 匹配源寄存器
    defs = set(RE_REG.findall(right))  # 匹配目标寄存器
    return uses, defs


def cache_lines_covered(addr: int, size: int, line_size: int = 64):
    """
    Calculate which cache lines are covered by a memory access.
    
    Args:
        addr: Starting address
        size: Access size in bytes
        line_size: Cache line size in bytes
        
    Returns:
        range: Range of cache line indices covered
    """
    start = addr // line_size
    end = (addr + size - 1) // line_size
    return range(start, end + 1)


def print_top(counter, title: str, topn: int = 50):
    """
    Print top N items from a Counter.
    
    Args:
        counter: collections.Counter object
        title: Title to display
        topn: Number of top items to show
    """
    print(f"\n== {title} (top {topn}) ==")
    for k, v in counter.most_common(topn):
        print(f"{k:40} {v}")
