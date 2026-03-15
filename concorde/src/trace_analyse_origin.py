#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import heapq  #  for dynamic-constraint simulations
import yaml
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict


# ============================================================
# ✅ Configuration Loader
# ============================================================
class ArchConfig:
    """Load and access hardware architecture configuration from YAML."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default config path (relative to script location)
            script_dir = Path(__file__).parent
            config_path = script_dir.parent / "config" / "arch_config.yml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        """Load YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"[CONFIG] Loaded architecture config from: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"[ERROR] Config file not found: {self.config_path}")
            print("[CONFIG] Using default hardcoded parameters.")
            return self._default_config()
        except yaml.YAMLError as e:
            print(f"[ERROR] Failed to parse YAML config: {e}")
            print("[CONFIG] Using default hardcoded parameters.")
            return self._default_config()
    
    def _default_config(self):
        """Fallback default configuration."""
        return {
            'rob': {'entries': 192, 'window_size': 400},
            'pipeline': {
                'fetch_width': 4, 'decode_width': 4, 'rename_width': 4, 'commit_width': 8,
                'issue_widths': {'alu': 3, 'fp': 2, 'ls': 2}
            },
            'load_store_pipes': {'load_store_pipes': 2, 'load_only_pipes': 10},
            'icache': {
                'size_bytes': 4096, 'line_size': 64, 'max_fills': 8,
                'fill_latency': 40, 'fetch_width': 8
            },
            'fetch_buffer': {'entries': 64},
            'cache_hierarchy': {
                'line_size': 64,
                'l1': {'size_bytes': 32768, 'associativity': 8, 'hit_latency': 4},
                'l2': {'size_bytes': 524288, 'associativity': 8, 'hit_latency': 12},
                'l3': {'size_bytes': 8388608, 'associativity': 16, 'hit_latency': 35},
                'memory': {'latency': 200}
            },
            'shared_llc': {
                'enabled': False, 'size_bytes': 8388608, 'associativity': 16,
                'line_size': 64, 'hit_latency': 35, 'num_banks': 16,
                'mshr_entries': 64,
                'memory': {'latency': 200, 'bandwidth_gbps': 100, 'num_channels': 4}
            },
            'branch_prediction': {
                'simple': {'misprediction_rate': 0.05, 'seed': 1},
                'tage': {
                    'num_tables': 8, 'table_size': 2048, 'tag_bits': 10,
                    'ghr_bits': 200, 'base_size': 4096, 'counter_bits': 3,
                    'usefulness_bits': 2, 'seed': 1
                }
            },
            'analysis': {
                'top_n': 50,
                'cdf': {
                    'quantile_step': 0.02, 'tail_quantile': 0.9,
                    'separate_figs': True,
                    'output': {'png_dpi': 200, 'dir': './result'}
                }
            }
        }
    
    def get(self, path: str, default=None):
        """
        Get config value by dot-separated path.
        Example: config.get('cache_hierarchy.l1.size_bytes')
        """
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

# Global config instance
config = None

# ============================================================
# Regex
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
BRANCH_MNEMS_EXACT = {
    "b", "bl", "br", "blr", "ret",
    "cbz", "cbnz", "tbz", "tbnz",
}
RE_THREAD_ID = re.compile(r'^\s*\d+\s+\d+:\s+(\d+)\b')

# ============================================================
# Cache simulation (unchanged from your base)
# ============================================================
class CacheLevel:
    """
    Simple set-associative cache with LRU.
    Inclusive hierarchy model is handled by calling lower levels on miss.
    """
    def __init__(self, name: str, size_bytes: int, assoc: int, line_size: int, hit_latency: int, lower=None):
        assert size_bytes % line_size == 0
        self.name = name
        self.size_bytes = size_bytes
        self.assoc = assoc
        self.line_size = line_size
        self.hit_latency = hit_latency
        self.lower = lower  # next level CacheLevel or MemoryLevel

        self.num_lines = size_bytes // line_size
        self.num_sets = max(1, self.num_lines // assoc)
        self.sets = [OrderedDict() for _ in range(self.num_sets)]

    def _index_tag(self, addr: int):
        line_addr = addr // self.line_size
        set_idx = line_addr % self.num_sets
        tag = line_addr // self.num_sets
        return set_idx, tag, line_addr

    def _evict_if_needed(self, set_idx: int):
        od = self.sets[set_idx]
        if len(od) < self.assoc:
            return None
        ev_tag, ev_dirty = od.popitem(last=False)
        ev_line_addr = (ev_tag * self.num_sets + set_idx)
        return ev_line_addr, ev_dirty

    def _insert_line(self, set_idx: int, tag: int, dirty: bool):
        od = self.sets[set_idx]
        od[tag] = dirty
        od.move_to_end(tag, last=True)

    def _touch_line(self, set_idx: int, tag: int, dirty: bool):
        od = self.sets[set_idx]
        if dirty:
            od[tag] = True
        od.move_to_end(tag, last=True)

    def access(self, addr: int, is_write: bool, count_writeback_cost: bool = True):
        """
        Return: (hit_level_name, total_cycles)
        total_cycles includes hit_latency of each probed level on the miss path.
        """
        set_idx, tag, line_addr = self._index_tag(addr)
        od = self.sets[set_idx]

        if tag in od:
            self._touch_line(set_idx, tag, dirty=is_write)
            return self.name, self.hit_latency

        lower_level, lower_cycles = self.lower.access(addr, is_write=is_write, count_writeback_cost=count_writeback_cost)
        total = self.hit_latency + lower_cycles

        ev = self._evict_if_needed(set_idx)
        if ev is not None and count_writeback_cost:
            ev_line_addr, ev_dirty = ev
            if ev_dirty:
                wb_addr = ev_line_addr * self.line_size
                _, wb_cycles = self.lower.access(wb_addr, is_write=True, count_writeback_cost=count_writeback_cost)
                total += wb_cycles

        self._insert_line(set_idx, tag, dirty=is_write)
        return lower_level, total

class MemoryLevel:
    def __init__(self, mem_latency: int):
        self.name = "MEM"
        self.mem_latency = mem_latency

    def access(self, addr: int, is_write: bool, count_writeback_cost: bool = True):
        return self.name, self.mem_latency

def build_cache_hierarchy():
    """Build cache hierarchy from config."""
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


# ============================================================
# Instruction structures
# ============================================================
class Instruction:
    def __init__(self, instr_type: str, instr_id: int, mnemonic: str, I_cache_line_addr: int = None):
        self.instr_type = instr_type
        self.instr_id = instr_id
        self.mnemonic = mnemonic
        self.I_cache_line_addr = I_cache_line_addr

        # ROB model fields
        self.exec_time = 1
        self.deps = []
        self.data_cache_line = None

    @property
    def is_load(self) -> bool:
        return self.instr_type == "load"
    
    
class branchInstruction(Instruction):
    def __init__(self, instr_type: str, instr_id: int, mnemonic: str,
                 I_cache_line_addr: int, pc: int):
        super().__init__(instr_type, instr_id, mnemonic, I_cache_line_addr)
        self.pc = pc
        self.branch_taken = None
        self.branch_type = None

class Instr_Load(Instruction):
    def __init__(self, instr_type: str, instr_id: int, mnemonic: str,
                 load_address: int, load_size: int, I_cache_line_addr: int, load_access_count: int = 0,
                 reg_deps=None, mem_deps=None):
        super().__init__(instr_type, instr_id, mnemonic, I_cache_line_addr)
        self.load_address = load_address
        self.load_cache_line = load_address // 64
        self.load_size = load_size

        self.reg_deps = sorted(reg_deps) if reg_deps else []
        self.mem_deps = sorted(mem_deps) if mem_deps else []

        self.hit_level = None
        self.mem_cycles = None
        self.levels_probed = None

        self.dependent_instr_ids = sorted(set(self.reg_deps) | set(self.mem_deps))
        self.load_access_count = load_access_count
        self.req_cycles = None

        # ROB fields
        self.data_cache_line = self.load_cache_line
        self.deps = sorted(set(self.reg_deps) | set(self.mem_deps))

class Instr_Store(Instruction):
    def __init__(self, instr_type: str, instr_id: int, mnemonic: str,
                 store_address: int, store_size: int, I_cache_line_addr: int, reg_deps=None):
        super().__init__(instr_type, instr_id, mnemonic, I_cache_line_addr)
        self.store_address = store_address
        self.store_cache_line = store_address // 64
        self.store_size = store_size
        self.store_latency = None
        self.deps = sorted(reg_deps) if reg_deps else []

class Instr_NonMem(Instruction):
    def __init__(self, instr_type: str, instr_id: int, mnemonic: str, I_cache_line_addr: int, reg_deps=None):
        super().__init__(instr_type, instr_id, mnemonic, I_cache_line_addr)
        self.deps = sorted(reg_deps) if reg_deps else []


# ============================================================
# Helpers (unchanged)
# ============================================================
def extract_uses_defs_from_ifetch(line: str):
    parts = line.split("->", 1) # 1表示只分割一次
    left = parts[0]
    right = parts[1] if len(parts) == 2 else ""
    uses = set(RE_REG.findall(left)) # 匹配源寄存器
    defs = set(RE_REG.findall(right)) # 匹配目标寄存器
    return uses, defs

def cache_lines_covered(addr: int, size: int, line_size: int = 64):
    start = addr // line_size
    end = (addr + size - 1) // line_size
    return range(start, end + 1)


# ============================================================
# Algorithm 1 + ROB model (your current version)
# ============================================================
def build_exec_times_by_cache_line(instructions_sorted):
    exec_times = defaultdict(list)
    for ins in instructions_sorted:
        if ins.is_load:
            exec_times[ins.data_cache_line].append(int(ins.exec_time))
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
            cl = instr.data_cache_line
            if req_cycle < self.last_req_cycles[cl]:
                req_cycle = self.last_req_cycles[cl]
            self.last_req_cycles[cl] = req_cycle

            prev_rsp = self.last_rsp_cycles[cl]
            access_num = self.access_counters[cl] # 获取当前缓存行访问次数

            et_list = self.exec_times.get(cl, [])
            exec_time = int(et_list[access_num]) if access_num < len(et_list) else int(instr.exec_time)

            rsp_cycle = max(req_cycle + exec_time, prev_rsp)

            self.last_rsp_cycles[cl] = rsp_cycle
            self.access_counters[cl] += 1
            return rsp_cycle

        return req_cycle + int(instr.exec_time)

def rob_throughput_model(instructions, ROB: int, k: int):
    """
    Eq.(1)-(5) ROB throughput model
    """
    instructions_sorted = sorted(instructions, key=lambda x: x.instr_id)
    n = len(instructions_sorted)
    if n == 0:
        return {"avg_ipc": 0.0, "thr_chunks": [], "a": [0], "s": [0], "f": [0], "c": [0], "instructions_sorted": []}

    id2idx = {ins.instr_id: i+1 for i, ins in enumerate(instructions_sorted)}  # 1..n

    dep_idx = [[] for _ in range(n+1)]
    for i, ins in enumerate(instructions_sorted, start=1):
        producers = []
        for pid in getattr(ins, "deps", []):
            j = id2idx.get(pid)
            if j is not None and j < i:
                producers.append(j)
        dep_idx[i] = producers

    exec_times_by_line = build_exec_times_by_cache_line(instructions_sorted)
    msm = MemoryStateMachine(exec_times_by_line)

    a = [0] * (n+1)
    s = [0] * (n+1)
    f = [0] * (n+1)
    c = [0] * (n+1)

    for i in range(1, n+1):
        a[i] = c[i-ROB] if i - ROB >= 0 else 0
        dep_finish = max((f[j] for j in dep_idx[i]), default=0)
        s[i] = max(a[i], dep_finish)

        # NOTE: Concorde executes Eq(3) in non-decreasing s_i order;
        # your current implementation does in i-order. We keep it as-is for now.
        f[i] = msm.resp_cycle(s[i], instructions_sorted[i-1])
        c[i] = max(f[i], c[i-1])

    thr = []
    num_chunks = n // k
    for j in range(1, num_chunks + 1):
        end_i = j * k
        start_i = (j - 1) * k
        denom = c[end_i] - c[start_i]
        thr.append((k / denom) if denom > 0 else 0.0)

    avg_ipc = (n / c[n]) if c[n] > 0 else 0.0
    return {"a": a, "s": s, "f": f, "c": c, "thr_chunks": thr, "avg_ipc": avg_ipc, "instructions_sorted": instructions_sorted}


# ============================================================
#  Static bandwidth resources + Dynamic constraints models
# ============================================================

# ----------  instruction classification for issue-width resources ----------
FP_MNEM_PREFIX = ("f", "fc", "fm", "fs", "fd")  # conservative heuristic for AArch64 FP/NEON mnemonics

def classify_issue_group(ins: Instruction) -> str:
    """
    Classify an instruction into an issue group for static issue-width constraints.
      - LS: loads/stores
      - FP: floating-point / SIMD (heuristic)
      - ALU: everything else (default)
    """
    if ins.instr_type in ("load", "store"):
        return "LS"
    m = (ins.mnemonic or "").lower()
    if m.startswith(FP_MNEM_PREFIX):
        return "FP"
    return "ALU"


# ----------  static bandwidth throughput per k-window ----------
def static_bandwidth_throughputs(instrs_sorted, k: int):
    """
    Static bandwidth resources:
      - For global widths (fetch/decode/rename/commit): throughput bound is simply width.
      - For issue widths (ALU/FP/LS): only a subset of instructions consume that bandwidth.
        If n_g is number of group instructions in window, processing time is ceil(n_g / width_g),
        thus throughput bound = k / time.
    Returns a dict: name -> list[thr_j]
    """
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

    for name in ("fetch_width", "decode_width", "rename_width", "commit_width"):
        w = int(widths.get(name, 0))
        if w > 0 and num_win > 0:
            out[name] = [float(w)] * num_win

    issue_cfg = [
        ("alu_issue_width", "ALU"),
        ("fp_issue_width",  "FP"),
        ("ls_issue_width",  "LS"),
    ]
    for width_name, grp in issue_cfg:
        w = int(widths.get(width_name, 0))
        if w <= 0 or num_win == 0:
            continue
        thr_series = []
        for j in range(num_win):
            win = instrs_sorted[j*k:(j+1)*k]
            n_grp = sum(1 for ins in win if classify_issue_group(ins) == grp)
            if n_grp == 0:
                thr_series.append(float("inf"))
            else:
                time = (n_grp + w - 1) // w
                thr_series.append(k / time)
        out[width_name] = thr_series

    return out


# ----------  dynamic constraint: Load/Load-Store pipes bounds ----------
def pipes_throughput_bounds(instrs_sorted, k: int):
    """
    Dynamic constraints example: finite Load-Store pipes (LSP) and Load pipes (LP).

    Lower bound (worst-case allocation) from the paper text:
      T_max = nLoad/(LSP+LP) + nStore/LSP
      thr_lower = k / T_max

    Upper bound (best-case schedule, we implement a constructive upper bound):
      - During store-issuing phase, stores occupy LSP for t_store = ceil(nStore/LSP)
      - During those cycles, loads issue on LP: issued_loads = t_store * LP
      - Remaining loads then issue with (LSP+LP) pipes.
      T_min = t_store + ceil( max(0, nLoad - t_store*LP) / (LSP+LP) )
      thr_upper = k / T_min

    Returns: {"pipes_thr_lower": [..], "pipes_thr_upper":[..]}
    """
    LSP = config.get('load_store_pipes.load_store_pipes', 2)
    LP = config.get('load_store_pipes.load_only_pipes', 10)
    
    n = len(instrs_sorted)
    num_win = n // k
    if num_win == 0:
        return {}

    LSP = max(1, int(LSP))
    LP = max(0, int(LP))

    lower_series = []
    upper_series = []

    for j in range(num_win):
        win = instrs_sorted[j*k:(j+1)*k]
        nL = sum(1 for ins in win if ins.instr_type == "load")
        nS = sum(1 for ins in win if ins.instr_type == "store")

        denom = (nL / (LSP + LP)) + (nS / LSP)
        lower_series.append((k / denom) if denom > 0 else float("inf"))

        t_store = (nS + LSP - 1) // LSP
        issued_loads = t_store * LP
        remL = max(0, nL - issued_loads)
        pipes_total = LSP + LP
        t_rem = (remL + pipes_total - 1) // pipes_total if remL > 0 else 0
        t_min = t_store + t_rem
        upper_series.append((k / t_min) if t_min > 0 else float("inf"))

    return {"pipes_thr_lower": lower_series, "pipes_thr_upper": upper_series}


# ----------  dynamic constraint: I-cache fills simulation + I-cache size & LRU ----------
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
    """
    n = len(instrs_sorted)
    num_win = n // k
    if n == 0 or num_win == 0:
        return {}

    resp = icache_fills_resp_times(instrs_sorted)

    # --------------------
    # Throughput over windows
    # --------------------
    thr = []
    for j in range(1, num_win + 1):
        end_i = j * k
        start_i = (j - 1) * k
        denom = resp[end_i] - resp[start_i]
        if denom < 0:
            raise ValueError("Negative denominator in I-cache fills throughput calculation")
        thr.append((k / denom) if denom > 0 else float("inf"))

    return {"icache_fills_thr": thr}

# ----------  I-cache fills: also expose per-instruction ready_time ----------
def icache_fills_resp_times(instrs_sorted):
    """
    Return resp[i] (1..n): the cycle when instruction i's I-cache line becomes ready
    under a simplified max in-flight fill model.
    
    Dynamic constraints: Maximum I-cache fills + finite I-cache capacity with LRU replacement.

    Model:
      - I-cache is modeled as a set of cache lines with capacity = icache_size_bytes / line_size.
      - LRU replacement using OrderedDict: most-recently-used at the end.
      - A fill request is issued only if the line is not in-flight and not currently resident in I-cache.
      - In-flight fills limited by max_fills; each completes after fill_latency cycles.
      - On fill completion, the line is inserted into I-cache (may evict LRU line).
      - resp[i] is the earliest cycle instruction i's line is available (monotonic non-decreasing enforced).
    """
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

    icache_lru = OrderedDict()
    inflight = {}
    heap = []
    cur_time = 0.0
    resp = [0.0] * (n + 1)
    fetch_time = [0.0] * (n + 1)

    def icache_touch(line: int):
        icache_lru.move_to_end(line, last=True)

    def icache_insert(line: int):
        if line in icache_lru:
            icache_touch(line)
            return
        icache_lru[line] = None
        icache_touch(line)
        if len(icache_lru) > capacity_lines:
            icache_lru.popitem(last=False)

    def retire_completed_fills(upto_time: int):
        while heap and heap[0][0] <= upto_time:
            t_done, line_done = heapq.heappop(heap)
            if inflight.get(line_done) == t_done:
                inflight.pop(line_done, None)
                icache_insert(line_done)
    
    for i, ins in enumerate(instrs_sorted, start=1):
        fetch_time[i] = (i - 1) // FETCH_WIDTH
        cl = ins.I_cache_line_addr
        retire_completed_fills(resp[i-1])
        
        if cl is None:
            resp[i] = max(fetch_time[i], resp[i-1])
            continue
        
        if cl in icache_lru:
            icache_touch(cl)
            resp[i] = max(fetch_time[i], resp[i-1])
            continue
        
        if cl in inflight:
            resp[i] = max(inflight[cl], fetch_time[i], resp[i-1])
            continue
        
        while len(inflight) >= max_fills:
            t_done, line_done = heapq.heappop(heap)
            cur_time = max(cur_time, t_done)
            if inflight.get(line_done) == t_done:
                inflight.pop(line_done, None)
                icache_insert(line_done)
        
        issue_time = max(cur_time, fetch_time[i])
        done_time = issue_time + fill_latency
        
        inflight[cl] = done_time
        heapq.heappush(heap, (done_time, cl))
        resp[i] = max(done_time, resp[i-1])
    
    return resp

# ----------  dynamic constraint: Fetch buffers simulation ----------
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
    """
    n = len(instrs_sorted)
    if n == 0:
        return {}

    fb_entries = max(1, int(fb_entries))
    decode_width = max(1, int(decode_width))
    if k <= 0:
        raise ValueError("k must be > 0")

    # ready_time[i] must be length n+1 (1..n)
    if ready_time is None:
        ready_time = [0] * (n + 1)
    if len(ready_time) != n + 1:
        raise ValueError("ready_time length mismatch: expected n+1")

    # t_dec[i] = cycle when instruction i is decoded (consumed from fetch buffer)
    t_dec = [0.0] * (n + 1)

    occ = 0                # fetch buffer occupancy (instructions)
    t = 0.0                # current cycle
    decoded_cnt = 0         # number of decoded instructions so far (1..decoded_cnt are completed)

    # helper: perform 1 decode cycle (consume up to decode_width)
    def do_one_cycle_decode(current_t: int):
        nonlocal occ, decoded_cnt
        d = min(occ, decode_width)
        if d > 0:
            # the next d instructions complete decode at current_t
            for i in range(d):
                decoded_cnt += 1
                t_dec[decoded_cnt] = current_t + (i + 1) / (decode_width + 1)
            occ -= d

    for i in range(1, n + 1):
        rt = float(ready_time[i])

        # advance time to at least rt, decoding along the way if buffer has content
        while t < rt and occ > 0:
            t += 1
            do_one_cycle_decode(t)

        # if buffer is empty, we can jump time directly
        if t < rt and occ == 0:
            t = rt

        # now try to enqueue instruction i; if buffer full, wait (decoding will free space)
        while occ >= fb_entries:
            t += 1
            do_one_cycle_decode(t)

        # enqueue this instruction
        occ += 1

        # Optional: if you want "decode can happen in the same cycle as enqueue",
        # you could uncomment the following block.
        # This is a modeling choice; by default we only decode on cycle advances.
        #
        # do_one_cycle_decode(t)

    # drain remaining buffer after the last arrival
    while occ > 0:
        t += 1
        do_one_cycle_decode(t)

    # compute throughput per k-window using decode completion timestamps
    num_win = n // k
    thr = []
    for j in range(1, num_win + 1):
        end_i = j * k
        start_i = (j - 1) * k
        denom = t_dec[end_i] - t_dec[start_i]
        thr.append((k / denom) if denom > 0 else float("inf"))

    return {"fb_decode_thr": thr, "t_dec": t_dec}


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


class SimplePredictor:
    def __init__(self, p: float = 0.05, seed: int = 1):
        self.p = float(p)
        self.seed = int(seed)
        self.seq = 0

    @staticmethod
    def _mix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        x = x ^ (x >> 31)
        return x & 0xFFFFFFFFFFFFFFFF

    def update_and_count(self, pc: int, actual_taken: bool):
        self.seq += 1
        h = self._mix64((pc or 0) ^ (self.seq << 1) ^ (self.seed * 0xD6E8FEB86659FD93))
        r = (h & 0xFFFFFFFF) / float(0x100000000)
        flip = (r < self.p)
        pred = (not actual_taken) if flip else actual_taken
        return pred, (pred != actual_taken)

from dataclasses import dataclass

@dataclass
class TageEntry:
    tag: int = 0
    ctr: int = 0     # signed-ish counter encoded [0..(2^cbits-1)]
    u: int = 0       # usefulness (small)
    valid: bool = False

class TAGEPredictor:
    """
    Research-grade simplified TAGE-like predictor (direction only).
    - Base: bimodal 2-bit counter table indexed by PC.
    - Tagged tables: multiple history lengths, each with (idx, tag) from (PC, folded history).
    - Provider: longest history table with matching tag; Alternate: next-longest.
    """
    def __init__(self,
                 num_tables: int = 8,
                 table_size: int = 2048,      # per tagged table
                 tag_bits: int = 10,
                 hist_lengths=None,           # list of lengths; if None, geometric
                 ghr_bits: int = 200,
                 base_size: int = 4096,
                 ctr_bits: int = 3,
                 u_bits: int = 2,
                 seed: int = 1):
        self.num_tables = int(num_tables)
        self.table_size = int(table_size)
        self.tag_bits = int(tag_bits)
        self.ghr_bits = int(ghr_bits)
        self.base_size = int(base_size)
        self.ctr_bits = int(ctr_bits)
        self.u_bits = int(u_bits)
        self.seed = int(seed)

        if hist_lengths is None:
            # geometric history lengths (common TAGE idea)
            # example: [4, 8, 16, 32, 64, 96, 128, 160] (clamped by ghr_bits)
            hl = []
            cur = 4
            for _ in range(self.num_tables):
                hl.append(min(cur, self.ghr_bits))
                cur = int(cur * 1.6) + 1
            self.hist_lengths = hl
        else:
            self.hist_lengths = [min(int(x), self.ghr_bits) for x in hist_lengths]
            self.num_tables = len(self.hist_lengths)

        # global history register (bitvector stored as int, LSB=most recent)
        self.ghr = 0

        # base bimodal counters: 2-bit
        self.base = [1] * self.base_size  # init weak-not-taken

        # tagged tables
        self.tables = []
        for _ in range(self.num_tables):
            self.tables.append([TageEntry() for _ in range(self.table_size)])

        # precompute masks
        self.tag_mask = (1 << self.tag_bits) - 1
        self.ghr_mask = (1 << min(self.ghr_bits, 63)) - 1  # for fast low-bit ops when ghr_bits<=63

    @staticmethod
    def _mix64(x: int) -> int:
        # splitmix64
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        x = x ^ (x >> 31)
        return x & 0xFFFFFFFFFFFFFFFF

    def _get_history_bits(self, L: int) -> int:
        # return low L bits of GHR (LSB = most recent)
        if L <= 0:
            return 0
        if L <= 63:
            return self.ghr & ((1 << L) - 1)
        # if L>63, we still only fold; using low 63 bits is ok for simplified model
        return self.ghr & ((1 << 63) - 1)

    def _fold(self, hist: int, L: int, out_bits: int) -> int:
        """
        Fold L bits history into out_bits via XOR chunks.
        """
        if out_bits <= 0:
            return 0
        mask = (1 << out_bits) - 1
        x = hist
        # simple folding: xor shifted windows
        for shift in (out_bits, 2*out_bits, 3*out_bits, 4*out_bits):
            x ^= (hist >> shift)
        return x & mask

    def _idx_tag(self, pc: int, table_i: int):
        L = self.hist_lengths[table_i]
        hist = self._get_history_bits(L)
        # fold history to index bits and tag bits
        idx_bits = int(math.log2(self.table_size))
        idx_bits = max(1, idx_bits)
        idx_mask = (1 << idx_bits) - 1

        h1 = self._mix64((pc or 0) ^ (hist * 0x9E3779B185EBCA87) ^ (self.seed + table_i))
        h2 = self._mix64((pc or 0) ^ (hist * 0xC2B2AE3D27D4EB4F) ^ (self.seed * 131 + table_i))

        idx = (h1 ^ self._fold(hist, L, idx_bits)) & idx_mask
        idx = idx % self.table_size

        tag = (h2 ^ self._fold(hist, L, self.tag_bits)) & self.tag_mask
        return idx, tag

    def _base_pred(self, pc: int) -> bool:
        idx = (pc or 0) % self.base_size
        ctr = self.base[idx]  # 0..3
        return ctr >= 2

    def _ctr_pred(self, ctr: int) -> bool:
        # ctr in [0..(2^ctr_bits-1)], mid threshold
        return ctr >= (1 << (self.ctr_bits - 1))

    def _update_ctr(self, ctr: int, taken: bool) -> int:
        maxv = (1 << self.ctr_bits) - 1
        if taken:
            return min(maxv, ctr + 1)
        else:
            return max(0, ctr - 1)

    def _update_base(self, pc: int, taken: bool):
        idx = (pc or 0) % self.base_size
        ctr = self.base[idx]
        if taken:
            self.base[idx] = min(3, ctr + 1)
        else:
            self.base[idx] = max(0, ctr - 1)

    def predict(self, pc: int):
        """
        Return prediction + metadata used for update:
          pred, provider_i(or None), provider_idx, provider_tag, alt_pred
        """
        hits = []
        for i in range(self.num_tables):
            idx, tag = self._idx_tag(pc, i)
            e = self.tables[i][idx]
            if e.valid and e.tag == tag:
                hits.append((i, idx, tag, e))

        # provider: longest history => max i (since hist_lengths grows)
        provider = None
        alt_pred = self._base_pred(pc)
        pred = alt_pred

        if hits:
            hits.sort(key=lambda x: x[0])  # increasing table index
            provider = hits[-1]            # last = longest history hit
            if len(hits) >= 2:
                alt = hits[-2]
                alt_pred = self._ctr_pred(alt[3].ctr)
            # choose provider prediction (simplified: always use provider)
            pred = self._ctr_pred(provider[3].ctr)

        return pred, provider, alt_pred

    def update(self, pc: int, actual_taken: bool, pred: bool, provider, alt_pred: bool):
        """
        Update tables, base, and GHR.
        """
        # update base always
        self._update_base(pc, actual_taken)

        mispred = (pred != actual_taken)

        # update provider counter if exists
        if provider is not None:
            i, idx, tag, e = provider
            e.ctr = self._update_ctr(e.ctr, actual_taken)
            # usefulness tweak: reward if provider helped over alternate (very simplified)
            if pred == actual_taken and pred != alt_pred:
                e.u = min((1 << self.u_bits) - 1, e.u + 1)
            elif mispred:
                e.u = max(0, e.u - 1)
            self.tables[i][idx] = e

        # allocation on mispredict: try to allocate in longer tables above provider
        if mispred:
            start_i = (provider[0] + 1) if provider is not None else 0
            for i in range(start_i, self.num_tables):
                idx, tag = self._idx_tag(pc, i)
                e = self.tables[i][idx]
                # allocate if invalid or not useful
                if (not e.valid) or (e.u == 0 and e.tag != tag):
                    e.valid = True
                    e.tag = tag
                    # init ctr biased toward actual
                    e.ctr = (1 << (self.ctr_bits - 1)) + (1 if actual_taken else -1)
                    e.ctr = max(0, min((1 << self.ctr_bits) - 1, e.ctr))
                    e.u = 0
                    self.tables[i][idx] = e
                    break

        # update GHR: shift in actual (LSB=most recent)
        self.ghr = ((self.ghr << 1) | (1 if actual_taken else 0))
        # clamp GHR to reasonable bits to avoid huge ints (keep ghr_bits)
        if self.ghr_bits < 2048:
            self.ghr &= (1 << self.ghr_bits) - 1

        return mispred

    def update_and_count(self, pc: int, actual_taken: bool):
        pred, provider, alt_pred = self.predict(pc)
        mispred = self.update(pc, actual_taken, pred, provider, alt_pred)
        return pred, mispred


def compute_branch_mispred_rate(instrs_sorted,
                                predictor,
                                only_conditional: bool = True,
                                require_label: bool = True):
    """
    Run branch prediction on instr stream and compute misprediction rate.

    only_conditional:
      - True: only count conditional branches (direction prediction is meaningful)
      - False: count all branches that have branch_taken label

    require_label:
      - True: skip branches with branch_taken is None
      - False: treat None as not-counted anyway (recommended keep True)

    Returns dict with totals and per-type breakdown.
    """
    total = 0
    misp = 0
    by_type = defaultdict(lambda: {"total": 0, "misp": 0})

    for ins in instrs_sorted:
        # 判断ins是否是branchInstruction类
        if not isinstance(ins, branchInstruction):
            continue
        if require_label and ins.branch_taken is None:
            continue

        btype = ins.branch_type

        if only_conditional and btype != "Direct Conditional Branch":
            continue


        pseudo_pc = ins.pc if ins.pc is not None else 0

        actual = bool(ins.branch_taken)

        pred, is_misp = predictor.update_and_count(pseudo_pc, actual)

        total += 1
        misp += 1 if is_misp else 0
        by_type[btype]["total"] += 1
        by_type[btype]["misp"] += 1 if is_misp else 0

    out = {
        "total": total,
        "misp": misp,
        "misp_rate": (misp / total) if total > 0 else 0.0,
        "by_type": {}
    }
    for t, d in by_type.items():
        out["by_type"][t] = {
            "total": d["total"],
            "misp": d["misp"],
            "misp_rate": (d["misp"] / d["total"]) if d["total"] > 0 else 0.0
        }
    return out


# ============================================================
# parse_trace (your current version + small glue for throughput models)
# ============================================================
def parse_trace(path: str):
    record_types = Counter()
    markers = Counter()
    mnemonics = Counter()
    Instruction_classify_arr = []
    branch_instr_classify = Counter()
    ls_instr_classify = Counter()
    Icache_instr_classify = Counter()
    Load_instr_list = []
    
    # Branch statistics
    branch_taken_stats = Counter()  # 统计taken/untaken数量
    branch_list = []  # 存储所有分支指令

    last_def_reg = {}
    ifetch_info = {}
    last_store_line = {}
    cache_line_access = {}

    instr_table = {}

    l1, l2, l3, mem = build_cache_hierarchy()
    hit_level_counter = Counter()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_MARKER.search(line)
            if m:
                marker_payload = m.group(1).strip()
                marker_key = marker_payload.split()[0] if marker_payload else "marker"
                record_types["marker"] += 1
                markers[marker_key] += 1
                continue

            m = RE_RECORD.match(line)
            if not m:
                continue
            rtype = m.group(1)
            record_types[rtype] += 1

            head = RE_HEAD.match(line)
            if not head:
                continue
            tid = int(head.group("tid"))
            instr_id = int(head.group("instr"))

            if rtype == "ifetch":
                mm = RE_IFETCH_MNEM.match(line)
                if not mm:
                    continue
                mnemonic = mm.group(1)
                mnemonics[mnemonic] += 1

                mpc = RE_IFETCH_PC.search(line)
                pc = int(mpc.group(1), 16) if mpc else None
                icl = (pc // 64) if pc is not None else None
                
                # print("cache line addr:", icl)

                uses, defs = extract_uses_defs_from_ifetch(line)

                reg_deps = set()
                for r in uses:
                    key = (tid, r.lower())
                    if key in last_def_reg:
                        reg_deps.add(last_def_reg[key])

                ifetch_info[(tid, instr_id)] = {
                    "mnemonic": mnemonic,
                    "pc": pc,
                    "uses": uses,
                    "defs": defs,
                    "reg_deps": reg_deps,
                }

                for r in defs:
                    last_def_reg[(tid, r.lower())] = instr_id

                # create/update stream instruction
                key_ins = (tid, instr_id)
                base = instr_table.get(key_ins)
                if base is None:
                    base = Instr_NonMem("nonmem", instr_id, mnemonic, icl, reg_deps=reg_deps)
                    instr_table[key_ins] = base
                else:
                    base.mnemonic = mnemonic
                    base.I_cache_line_addr = icl
                    base.deps = sorted(set(getattr(base, "deps", [])) | reg_deps)

                # ========== Branch classification & taken/untaken extraction ==========
                if mnemonic in BRANCH_MNEMS_EXACT:
                    # base.is_branch = True
                    # 将base从instruction基类转换成分支指令类
                    base = branchInstruction(
                        instr_type="branch",
                        instr_id=instr_id,
                        mnemonic=mnemonic,
                        I_cache_line_addr=icl,
                        pc = pc
                    )
                    
                    instr_table[key_ins] = base  # 更新回instr_table
                    
                    # Extract taken/untaken status
                    m_status = RE_BRANCH_STATUS.search(line)
                    if m_status:
                        status = m_status.group(1).lower()
                        base.branch_taken = (status == "taken")
                        branch_taken_stats[status] += 1 
                    
                    branch_type = classify_branch_type(mnemonic)
                    base.branch_type = branch_type
                    
                    # Store branch instruction for analysis
                    branch_list.append(base)

                Icache_instr_classify["I-cache Instructions"] += 1
                ls_instr_classify["non-memory Instructions"] += 1
                continue

            if rtype == "read":
                ls_instr_classify["Load Instructions"] += 1
                m_load = RE_RW_DETAIL.search(line)
                if not m_load:
                    continue

                load_size = int(m_load.group("size"))
                load_addr = int(m_load.group("addr"), 16)
                pc = int(m_load.group("pc"), 16)
                icl = pc // 64

                info = ifetch_info.get((tid, instr_id), {})
                mnemonic = info.get("mnemonic", "UNKNOWN")
                reg_deps = set(info.get("reg_deps", set()))

                mem_deps = set()
                for cl in cache_lines_covered(load_addr, load_size, 64):
                    producer = last_store_line.get((tid, cl))
                    if producer is not None:
                        mem_deps.add(producer)

                cache_line_access_key = load_addr // 64
                cache_line_access[cache_line_access_key] = cache_line_access.get(cache_line_access_key, 0) + 1

                # [FIXED] Simulate cache access for EACH cache line touched by this load
                total_mem_cycles = 0
                worst_hit_level = "L1"
                num_cache_lines = 0
                
                for cl in cache_lines_covered(load_addr, load_size, 64):
                    cl_addr = cl * 64  # cache line起始地址
                    hit_level, cycles = l1.access(cl_addr, is_write=False, count_writeback_cost=True)
                    hit_level_counter[hit_level] += 1  # 对每个cache line访问计数
                    total_mem_cycles += cycles
                    num_cache_lines += 1
                    
                    # 记录最差的命中级别(用于指令标注)
                    level_priority = {"L1": 0, "L2": 1, "L3": 2, "MEM": 3}
                    if level_priority.get(hit_level, 0) > level_priority.get(worst_hit_level, 0):
                        worst_hit_level = hit_level

                load_instr = Instr_Load(
                    instr_type="load",
                    instr_id=instr_id,
                    mnemonic=mnemonic,
                    load_address=load_addr,
                    load_size=load_size,
                    I_cache_line_addr=icl,
                    reg_deps=reg_deps,
                    mem_deps=mem_deps,
                    load_access_count=cache_line_access[cache_line_access_key],
                )
                Load_instr_list.append(load_instr)

                # 使用最差情况作为指令的hit level标注
                load_instr.hit_level = worst_hit_level
                load_instr.mem_cycles = total_mem_cycles
                level_map = {"L1": 1, "L2": 2, "L3": 3, "MEM": 4}
                load_instr.levels_probed = level_map.get(worst_hit_level, None)

                # ROB exec_time uses total cache-sim cycles
                load_instr.exec_time = int(total_mem_cycles)

                instr_table[(tid, instr_id)] = load_instr
                continue

            if rtype == "write":
                ls_instr_classify["Store Instructions"] += 1
                m_store = RE_RW_DETAIL.search(line)
                if not m_store:
                    continue

                store_size = int(m_store.group("size"))
                store_addr = int(m_store.group("addr"), 16)
                pc = int(m_store.group("pc"), 16)
                icl = pc // 64

                for cl in cache_lines_covered(store_addr, store_size, 64):
                    last_store_line[(tid, cl)] = instr_id

                # [FIXED] Simulate cache access for EACH cache line
                total_store_cycles = 0
                for cl in cache_lines_covered(store_addr, store_size, 64):
                    cl_addr = cl * 64
                    hit_level_s, cycles_s = l1.access(cl_addr, is_write=True, count_writeback_cost=True)
                    hit_level_counter[hit_level_s] += 1  # 每个cache line计数
                    total_store_cycles += cycles_s

                info = ifetch_info.get((tid, instr_id), {})
                mnemonic = info.get("mnemonic", "UNKNOWN")
                reg_deps = set(info.get("reg_deps", set()))

                store_ins = Instr_Store(
                    instr_type="store",
                    instr_id=instr_id,
                    mnemonic=mnemonic,
                    store_address=store_addr,
                    store_size=store_size,
                    I_cache_line_addr=icl,
                    reg_deps=reg_deps
                )
                store_ins.exec_time = max(1, int(total_store_cycles))
                instr_table[(tid, instr_id)] = store_ins
                continue

    Instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    Instruction_classify_arr.append(ls_instr_classify)
    Instruction_classify_arr.append(Icache_instr_classify)

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


# ============================================================
# 新增：多核共享 LLC + Memory 竞争模型
# ============================================================

class SharedLLCWithQueuing:
    """
    Shared Last-Level Cache with queuing model for multi-core contention.
    
    Features:
      - Finite MSHR entries (miss request buffer)
      - Bank conflicts (addresses hash to banks, serialized within bank)
      - LRU replacement with capacity pressure from all cores
      - Memory bandwidth queuing (M/M/c model for DRAM)
    """
    def __init__(self):
        """Initialize from global config."""
        llc_cfg = config.config.get('shared_llc', {})
        
        self.name = "L3"  # Unified L3/LLC
        self.size_bytes = llc_cfg.get('size_bytes', 8*1024*1024)
        self.assoc = llc_cfg.get('associativity', 16)
        self.line_size = llc_cfg.get('line_size', 64)
        self.hit_latency = llc_cfg.get('hit_latency', 35)
        self.num_banks = llc_cfg.get('num_banks', 16)
        self.mshr_entries = llc_cfg.get('mshr_entries', 64)
        
        mem_cfg = llc_cfg.get('memory', {})
        self.mem_latency = mem_cfg.get('latency', 200)
        self.mem_bandwidth_gbps = mem_cfg.get('bandwidth_gbps', 100)
        self.num_mem_channels = mem_cfg.get('num_channels', 4)
        
        # Initialize structures
        self.num_lines = self.size_bytes // self.line_size
        self.num_sets = max(1, self.num_lines // self.assoc)
        self.sets = [OrderedDict() for _ in range(self.num_sets)]
        
        self.mshr_queue = []
        self.bank_last_access = [0] * self.num_banks
        self.bank_access_latency = 2
        
        self.mem_service_rate = (self.mem_bandwidth_gbps * 1e9) / (self.line_size * 8)
        self.mem_service_time_cycles = 1.0 / (self.mem_service_rate / 1e9)
        self.mem_active_channels = [0] * self.num_mem_channels
        
        self.stats = {
            "hits": 0, "misses": 0, "mshr_stalls": 0,
            "bank_conflicts": 0, "mem_queue_depth_sum": 0, "mem_queue_samples": 0
        }
    
    def _index_tag_bank(self, addr: int):
        """Hash address to (set_idx, tag, bank_id)"""
        line_addr = addr // self.line_size
        set_idx = line_addr % self.num_sets
        tag = line_addr // self.num_sets
        bank_id = line_addr % self.num_banks
        return set_idx, tag, bank_id, line_addr
    
    def _touch_line(self, set_idx: int, tag: int):
        """Update LRU (move to MRU position)"""
        self.sets[set_idx].move_to_end(tag, last=True)
    
    def _insert_line(self, set_idx: int, tag: int):
        """Insert line with LRU eviction if needed"""
        od = self.sets[set_idx]
        if len(od) >= self.assoc:
            od.popitem(last=False)  # evict LRU
        od[tag] = None
        od.move_to_end(tag, last=True)
    
    def _retire_completed_mshrs(self, current_time: int):
        """Complete MSHRs that finished by current_time"""
        while self.mshr_queue and self.mshr_queue[0][0] <= current_time:
            _, line_addr, tid = heapq.heappop(self.mshr_queue)
            set_idx = (line_addr // self.num_sets) % self.num_sets
            tag = line_addr // self.num_sets
            self._insert_line(set_idx, tag)
    
    def _simulate_mem_access(self, request_time: int, tid: int, line_addr: int) -> int:
        """
        Simulate memory access with queuing (M/M/c model approximation).
        Returns completion time.
        """
        # Find earliest available channel
        min_finish = min(self.mem_active_channels)
        service_start = max(request_time, min_finish)
        
        # Service time = base latency + queueing delay
        service_cycles = self.mem_latency + int(self.mem_service_time_cycles)
        completion = service_start + service_cycles
        
        # Update channel state
        earliest_idx = self.mem_active_channels.index(min_finish)
        self.mem_active_channels[earliest_idx] = completion
        
        # Stats
        queue_depth = sum(1 for t in self.mem_active_channels if t > request_time)
        self.stats["mem_queue_depth_sum"] += queue_depth
        self.stats["mem_queue_samples"] += 1
        
        return completion
    
    def access(self, addr: int, current_time: int, tid: int, is_write: bool = False):
        """
        Access LLC at current_time by thread tid.
        Returns: (hit_level, completion_time)
        
        Flow:
          1. Retire completed MSHRs
          2. Check LLC hit (account for bank conflict)
          3. On miss: check MSHR availability, queue memory request
        """
        self._retire_completed_mshrs(current_time)
        
        set_idx, tag, bank_id, line_addr = self._index_tag_bank(addr)
        
        # Bank conflict delay
        bank_ready = self.bank_last_access[bank_id] + self.bank_access_latency
        if current_time < bank_ready:
            self.stats["bank_conflicts"] += 1
            access_time = bank_ready
        else:
            access_time = current_time
        self.bank_last_access[bank_id] = access_time
        
        # Check cache hit
        od = self.sets[set_idx]
        if tag in od:
            self._touch_line(set_idx, tag)
            self.stats["hits"] += 1
            return "LLC", access_time + self.hit_latency
        
        # Cache miss
        self.stats["misses"] += 1
        
        # Check MSHR availability (block if full)
        while len(self.mshr_queue) >= self.mshr_entries:
            self.stats["mshr_stalls"] += 1
            # Must wait for oldest MSHR to complete
            next_completion, _, _ = self.mshr_queue[0]
            self._retire_completed_mshrs(next_completion)
            access_time = max(access_time, next_completion)
        
        # Issue memory request
        mem_completion = self._simulate_mem_access(access_time, tid, line_addr)
        
        # Allocate MSHR
        heapq.heappush(self.mshr_queue, (mem_completion, line_addr, tid))
        
        return "MEM", mem_completion
    
    def print_stats(self):
        """Print contention statistics"""
        total = self.stats["hits"] + self.stats["misses"]
        miss_rate = self.stats["misses"] / total if total > 0 else 0
        avg_queue = (self.stats["mem_queue_depth_sum"] / 
                     self.stats["mem_queue_samples"]) if self.stats["mem_queue_samples"] > 0 else 0
        
        print(f"\n[LLC] Shared LLC Statistics:")
        print(f"  Hits: {self.stats['hits']}, Misses: {self.stats['misses']}, Miss Rate: {miss_rate:.2%}")
        print(f"  MSHR Stalls: {self.stats['mshr_stalls']}")
        print(f"  Bank Conflicts: {self.stats['bank_conflicts']}")
        print(f"  Avg Memory Queue Depth: {avg_queue:.2f}")


# ============================================================
# 修改 parse_trace 使用共享 LLC
# ============================================================

def parse_trace_with_shared_llc(path: str):
    """
    Modified parse_trace that uses shared LLC across all threads.
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
    
    # [NEW] Use shared LLC instead of per-core private caches
    shared_llc = SharedLLCWithQueuing(
        size_bytes=8*1024*1024,   # 8MB
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
            mem_stub = MemoryLevel(mem_latency=0)  # stub, won't be used
            l2 = CacheLevel(name="L2", size_bytes=512*1024, assoc=8, 
                           line_size=line_size, hit_latency=12, lower=mem_stub)
            l1 = CacheLevel(name="L1", size_bytes=32*1024, assoc=8,
                           line_size=line_size, hit_latency=4, lower=l2)
            thread_caches[tid] = (l1, l2)
        return thread_caches[tid]
    
    hit_level_counter = Counter()
    current_time = 0  # global cycle counter (simplified timeline)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_MARKER.search(line)
            if m:
                marker_payload = m.group(1).strip()
                marker_key = marker_payload.split()[0] if marker_payload else "marker"
                record_types["marker"] += 1
                markers[marker_key] += 1
                continue

            m = RE_RECORD.match(line)
            if not m:
                continue
            rtype = m.group(1)
            record_types[rtype] += 1

            head = RE_HEAD.match(line)
            if not head:
                continue
            tid = int(head.group("tid"))
            instr_id = int(head.group("instr"))
            
            # [简化] 用 instr_id 作为时间戳（实际应从 trace 中提取）
            current_time = instr_id

            if rtype == "ifetch":
                mm = RE_IFETCH_MNEM.match(line)
                if not mm:
                    continue
                mnemonic = mm.group(1)
                mnemonics[mnemonic] += 1

                mpc = RE_IFETCH_PC.search(line)
                pc = int(mpc.group(1), 16) if mpc else None
                icl = (pc // 64) if pc is not None else None

                uses, defs = extract_uses_defs_from_ifetch(line)
                reg_deps = set()
                for r in uses:
                    key = (tid, r.lower())
                    if key in last_def_reg:
                        reg_deps.add(last_def_reg[key])

                ifetch_info[(tid, instr_id)] = {
                    "mnemonic": mnemonic,
                    "pc": pc,
                    "uses": uses,
                    "defs": defs,
                    "reg_deps": reg_deps,
                }

                for r in defs:
                    last_def_reg[(tid, r.lower())] = instr_id

                key_ins = (tid, instr_id)
                base = instr_table.get(key_ins)
                if base is None:
                    base = Instr_NonMem("nonmem", instr_id, mnemonic, icl, reg_deps=reg_deps)
                    instr_table[key_ins] = base
                else:
                    base.mnemonic = mnemonic
                    base.I_cache_line_addr = icl
                    base.deps = sorted(set(getattr(base, "deps", [])) | reg_deps)

                if mnemonic in BRANCH_MNEMS_EXACT:
                    base = branchInstruction(
                        instr_type="branch",
                        instr_id=instr_id,
                        mnemonic=mnemonic,
                        I_cache_line_addr=icl,
                        pc=pc
                    )
                    instr_table[key_ins] = base
                    
                    m_status = RE_BRANCH_STATUS.search(line)
                    if m_status:
                        status = m_status.group(1).lower()
                        base.branch_taken = (status == "taken")
                    
                    branch_type = classify_branch_type(mnemonic)
                    base.branch_type = branch_type

                Icache_instr_classify["I-cache Instructions"] += 1
                ls_instr_classify["non-memory Instructions"] += 1
                continue

            if rtype == "read":
                ls_instr_classify["Load Instructions"] += 1
                m_load = RE_RW_DETAIL.search(line)
                if not m_load:
                    continue

                load_size = int(m_load.group("size"))
                load_addr = int(m_load.group("addr"), 16)
                pc = int(m_load.group("pc"), 16)
                icl = pc // 64

                info = ifetch_info.get((tid, instr_id), {})
                mnemonic = info.get("mnemonic", "UNKNOWN")
                reg_deps = set(info.get("reg_deps", set()))

                mem_deps = set()
                for cl in cache_lines_covered(load_addr, load_size, 64):
                    producer = last_store_line.get((tid, cl))
                    if producer is not None:
                        mem_deps.add(producer)

                cache_line_access_key = load_addr // 64
                cache_line_access[cache_line_access_key] = cache_line_access.get(cache_line_access_key, 0) + 1

                # [修改] 先访问私有 L1/L2，未命中则访问共享 LLC
                l1, l2 = get_thread_cache(tid)
                total_mem_cycles = 0
                worst_hit_level = "L1"
                
                for cl in cache_lines_covered(load_addr, load_size, 64):
                    cl_addr = cl * 64
                    
                    # Try L1
                    set_idx, tag, _ = l1._index_tag(cl_addr)
                    if tag in l1.sets[set_idx]:
                        l1._touch_line(set_idx, tag, dirty=False)
                        hit_level = "L1"
                        cycles = l1.hit_latency
                    else:
                        # Try L2
                        set_idx2, tag2, _ = l2._index_tag(cl_addr)
                        if tag2 in l2.sets[set_idx2]:
                            l2._touch_line(set_idx2, tag2, dirty=False)
                            l1._insert_line(set_idx, tag, dirty=False)
                            hit_level = "L2"
                            cycles = l1.hit_latency + l2.hit_latency
                        else:
                            # [NEW] Access shared LLC with queuing
                            hit_level, completion = shared_llc.access(
                                cl_addr, current_time, tid, is_write=False
                            )
                            cycles = completion - current_time
                            
                            # Install in L2 and L1
                            l2._insert_line(set_idx2, tag2, dirty=False)
                            l1._insert_line(set_idx, tag, dirty=False)
                    
                    hit_level_counter[hit_level] += 1
                    total_mem_cycles += cycles
                    
                    level_priority = {"L1": 0, "L2": 1, "LLC": 2, "MEM": 3}
                    if level_priority.get(hit_level, 0) > level_priority.get(worst_hit_level, 0):
                        worst_hit_level = hit_level

                load_instr = Instr_Load(
                    instr_type="load",
                    instr_id=instr_id,
                    mnemonic=mnemonic,
                    load_address=load_addr,
                    load_size=load_size,
                    I_cache_line_addr=icl,
                    reg_deps=reg_deps,
                    mem_deps=mem_deps,
                    load_access_count=cache_line_access[cache_line_access_key],
                )
                Load_instr_list.append(load_instr)

                load_instr.hit_level = worst_hit_level
                load_instr.mem_cycles = total_mem_cycles
                level_map = {"L1": 1, "L2": 2, "LLC": 3, "MEM": 4}
                load_instr.levels_probed = level_map.get(worst_hit_level, None)
                load_instr.exec_time = int(total_mem_cycles)

                instr_table[(tid, instr_id)] = load_instr
                continue

            if rtype == "write":
                ls_instr_classify["Store Instructions"] += 1
                m_store = RE_RW_DETAIL.search(line)
                if not m_store:
                    continue

                store_size = int(m_store.group("size"))
                store_addr = int(m_store.group("addr"), 16)
                pc = int(m_store.group("pc"), 16)
                icl = pc // 64

                for cl in cache_lines_covered(store_addr, store_size, 64):
                    last_store_line[(tid, cl)] = instr_id

                # [简化] Store 只访问 L1，假设 write-through
                l1, l2 = get_thread_cache(tid)
                total_store_cycles = 0
                for cl in cache_lines_covered(store_addr, store_size, 64):
                    cl_addr = cl * 64
                    set_idx, tag, _ = l1._index_tag(cl_addr)
                    l1._insert_line(set_idx, tag, dirty=True)
                    total_store_cycles += l1.hit_latency

                info = ifetch_info.get((tid, instr_id), {})
                mnemonic = info.get("mnemonic", "UNKNOWN")
                reg_deps = set(info.get("reg_deps", set()))

                store_ins = Instr_Store(
                    instr_type="store",
                    instr_id=instr_id,
                    mnemonic=mnemonic,
                    store_address=store_addr,
                    store_size=store_size,
                    I_cache_line_addr=icl,
                    reg_deps=reg_deps
                )
                store_ins.exec_time = max(1, int(total_store_cycles))
                instr_table[(tid, instr_id)] = store_ins
                continue

    Instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    Instruction_classify_arr.append(ls_instr_classify)
    Instruction_classify_arr.append(Icache_instr_classify)

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

def print_top(counter: Counter, title: str, topn: int = 50):
    print(f"\n== {title} (top {topn}) ==")
    for k, v in counter.most_common(topn):
        print(f"{k:20s} {v}")


# ============================================================
#  a small pretty-printer for throughput series
# ============================================================
def summarize_thr_series(name: str, thr_list, max_show: int = 10):
    finite = [x for x in thr_list if x != float("inf") and x > 0]
    if not finite:
        print(f"[THR] {name}: no finite samples")
        return
    finite_sorted = sorted(finite)
    p50 = finite_sorted[len(finite_sorted)//2]
    p10 = finite_sorted[int(0.10 * (len(finite_sorted)-1))]
    p90 = finite_sorted[int(0.90 * (len(finite_sorted)-1))]
    print(f"[THR] {name}: samples={len(thr_list)} finite={len(finite)} p10={p10:.6f} p50={p50:.6f} p90={p90:.6f} first={thr_list[:max_show]}")

import math
import numpy as np
import matplotlib.pyplot as plt

def ecdf_from_series(series, drop_inf=True, drop_nan=True):
    """
    Build empirical CDF from a throughput series.
    Returns: (x_sorted, y_cdf, meta_dict)
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

    # keep non-negative (optional; comment out if you want to see negatives)
    # arr = arr[arr >= 0]

    if arr.size == 0:
        return np.array([]), np.array([]), meta

    x = np.sort(arr)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y, meta

def cdf_to_vectors(x, y, quantiles=np.arange(0, 1, 0.02)): 
    """
    Convert CDF (x,y) to quantile vectors.
    quantiles: array-like of quantiles in [0,1)
    Returns: x_at_quantiles
    """
    quantiles = np.asarray(quantiles)
    x_at_q = np.interp(quantiles, y, x, left=x[0], right=x[-1])
    return x_at_q

def generate_cdf_vectors(series_dict, drop_inf=True, drop_nan=True,
                         quantiles=np.arange(0, 1, 0.02)):
    """
    Generate CDF quantile vectors from a throughput series.
    Returns: quantiles vectors of each series.
    """
    cdf_vectors = {}
    for name, series in series_dict.items():
        x, y, meta = ecdf_from_series(series, drop_inf=drop_inf, drop_nan=drop_nan)
        if x.size == 0:
            continue
        x_at_q = cdf_to_vectors(x, y, quantiles=quantiles)
        cdf_vectors[name] = x_at_q
    return cdf_vectors



def plot_cdf_bundle(series_dict, title, out_path_png=None, out_path_pdf=None,
                    drop_inf=True, xlim=None, show_tail_zoom=False,
                    tail_quantile=0.9, seperate_figs=False):   
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
        if seperate_figs:
            plt.figure()
        x, y, meta = ecdf_from_series(series, drop_inf=drop_inf, drop_nan=True)
        if x.size == 0:
            continue
        inf_ratio = meta["n_inf"] / meta["n_total"] if meta["n_total"] > 0 else 0.0
        label = f"{name} (inf={inf_ratio:.2%})" if drop_inf else name
        plt.plot(x, y, label=label)
        if seperate_figs:
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
    if not seperate_figs:
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
            x, y, meta = ecdf_from_series(series, drop_inf=drop_inf, drop_nan=True)
            if x.size == 0:
                continue
            # zoom into tail region
            mask = y >= tail_quantile
            if not np.any(mask):
                continue
            inf_ratio = meta["n_inf"] / meta["n_total"] if meta["n_total"] > 0 else 0.0
            label = f"{name} (inf={inf_ratio:.2%})" if drop_inf else name
            plt.plot(x[mask], y[mask], label=label)

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

    # If you run in an interactive env, show plots
    # plt.show()


def main():
    global config
    
    if len(sys.argv) < 2:
        print("Usage: python trace_analyse.py trace.log [config.yml] [--shared-llc]")
        print("       If config.yml is omitted, uses default: ../config/arch_config.yml")
        sys.exit(1)

    path = sys.argv[1]
    
    # ✅ Parse command line arguments
    config_file = None
    use_shared_llc = "--shared-llc" in sys.argv
    
    for arg in sys.argv[2:]:
        if arg.endswith('.yml') or arg.endswith('.yaml'):
            config_file = arg
    
    # ✅ Load config
    config = ArchConfig(config_file)
    
    # ✅ Get parameters from config
    ROB = config.get('rob.entries', 192)
    k = config.get('rob.window_size', 400)
    topn = config.get('analysis.top_n', 50)

    # ✅ Parse trace
    if use_shared_llc:
        print("[CONFIG] Using Shared L3/LLC model (multi-core contention)")
        (
            record_types, markers, mnemonics, Instruction_classify_arr,
            load_instr_list, cache_line_access, hit_level_counter,
            instrs_by_tid
        ) = parse_trace_with_shared_llc(path)
    else:
        print("[CONFIG] Using Private L1/L2/L3 model (no contention)")
        (
            record_types, markers, mnemonics, Instruction_classify_arr,
            load_instr_list, cache_line_access, hit_level_counter,
            instrs_by_tid
        ) = parse_trace(path)

    if hit_level_counter:
        print_top(hit_level_counter, "Load hit level (cache sim)", topn=10)

    if not instrs_by_tid:
        print("\n[ROB] No instructions collected for ROB model.")
        return
    
    target_tid = max(instrs_by_tid.items(), key=lambda kv: len(kv[1]))[0]
    instrs = instrs_by_tid.get(target_tid, [])
    instrs_sorted = sorted(instrs, key=lambda x: x.instr_id)

    print(f"\n[ROB] Target TID = {target_tid}, instructions = {len(instrs_sorted)}, ROB={ROB}, k={k}")
    rob_res = rob_throughput_model(instrs_sorted, ROB=ROB, k=k)
    print(f"[ROB] Avg IPC (model): {rob_res['avg_ipc']:.6f}")

    # ✅ Static bandwidth (from config)
    static_thr = static_bandwidth_throughputs(instrs_sorted, k=k)
    print("\n[STATIC] Static bandwidth resources throughput (per k-window):")
    for name, series in static_thr.items():
        summarize_thr_series(name, series)

    # ✅ Dynamic constraints (from config)
    pipes_thr = pipes_throughput_bounds(instrs_sorted, k=k)
    print("\n[DYNAMIC] Load/Load-Store pipes throughput bounds (per k-window):")
    for name, series in pipes_thr.items():
        summarize_thr_series(name, series)

    # ✅ I-cache (from config)
    ic_thr = icache_fills_throughput(instrs_sorted, k=k)
    print("\n[DYNAMIC] I-cache fills throughput (per k-window):")
    for name, series in ic_thr.items():
        summarize_thr_series(name, series)

    # ✅ Fetch buffer (from config)
    fb_entries = config.get('fetch_buffer.entries', 64)
    decode_width = config.get('pipeline.decode_width', 4)
    ready_time = icache_fills_resp_times(instrs_sorted)
    fb_res = fetch_buffers_throughput(instrs_sorted, k=k,
                                      fb_entries=fb_entries,
                                      decode_width=decode_width,
                                      ready_time=ready_time)

    print("\n[DYNAMIC] Fetch buffers throughput (per k-window):")
    for name, series in fb_res.items():
        if name == "t_dec":
            continue
        summarize_thr_series(name, series)

    # ✅ Branch prediction (from config)
    simple_cfg = config.get('branch_prediction.simple', {})
    simple = SimplePredictor(
        p=simple_cfg.get('misprediction_rate', 0.05),
        seed=simple_cfg.get('seed', 1)
    )
    br_simple = compute_branch_mispred_rate(instrs_sorted, simple, only_conditional=True, require_label=True)
    print("\n[BR] Simple predictor misprediction rate (conditional only):")
    print(f"     total={br_simple['total']} misp={br_simple['misp']} rate={br_simple['misp_rate']:.4%}")

    tage_cfg = config.get('branch_prediction.tage', {})
    tage = TAGEPredictor(
        num_tables=tage_cfg.get('num_tables', 8),
        table_size=tage_cfg.get('table_size', 2048),
        tag_bits=tage_cfg.get('tag_bits', 10),
        ghr_bits=tage_cfg.get('ghr_bits', 200),
        base_size=tage_cfg.get('base_size', 4096),
        ctr_bits=tage_cfg.get('counter_bits', 3),
        u_bits=tage_cfg.get('usefulness_bits', 2),
        seed=tage_cfg.get('seed', 1)
    )
    br_tage = compute_branch_mispred_rate(instrs_sorted, tage, only_conditional=True, require_label=True)
    print("\n[BR] TAGE-like predictor misprediction rate (conditional only):")
    print(f"     total={br_tage['total']} misp={br_tage['misp']} rate={br_tage['misp_rate']:.4%}")

    # ✅ CDF plotting (from config)
    all_series = {}
    if rob_res.get("thr_chunks"):
        all_series["ROB.thr_chunks"] = rob_res["thr_chunks"]
    for name, series in static_thr.items():
        all_series[f"STATIC.{name}"] = series
    for name, series in pipes_thr.items():
        all_series[f"DYN.{name}"] = series
    for name, series in ic_thr.items():
        all_series[f"DYN.{name}"] = series
    for name, series in fb_res.items():
        if name == "t_dec":
            continue
        all_series[f"DYN.{name}"] = series

    cdf_cfg = config.get('analysis.cdf', {})
    output_dir = cdf_cfg.get('output', {}).get('dir', './result')
    
    plot_cdf_bundle(
        all_series,
        title="Throughput CDF per Resource (per k-window)",
        out_path_png=f"{output_dir}/throughput_cdf_all.png",
        out_path_pdf=f"{output_dir}/throughput_cdf_all.pdf",
        drop_inf=True,
        xlim=None,
        show_tail_zoom=True,
        tail_quantile=cdf_cfg.get('tail_quantile', 0.9),
        seperate_figs=cdf_cfg.get('separate_figs', True)
    )

    print("\n[PLOT] Saved CDF plots.")


if __name__ == "__main__":
    main()
