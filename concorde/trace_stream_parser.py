#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import mmap
from pathlib import Path
import struct
import sys
from typing import Iterable


CONCORDE_ROOT = Path(__file__).resolve().parent
if str(CONCORDE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONCORDE_ROOT))

from src.cache import SharedLLCWithQueuing, build_cache_hierarchy  # type: ignore
from src.instruction import Instr_Load, Instr_NonMem, Instr_Store, branchInstruction  # type: ignore
from src.utils import (  # type: ignore
    BRANCH_MNEMS_EXACT,
    RE_BRANCH_STATUS,
    RE_HEAD,
    RE_IFETCH_MNEM,
    RE_IFETCH_PC,
    RE_MARKER,
    RE_RW_DETAIL,
    cache_lines_covered,
    extract_uses_defs_from_ifetch,
)


COMPACT_TRACE_MAGIC = b"CTRCBIN1"
COMPACT_TRACE_HEADER = struct.Struct("<QIBBBBQQQ")
COMPACT_MEMOP_HEADER = struct.Struct("<BQI")
REG_INDEX_SP = 31
REG_INDEX_XZR = 32


def _branch_kind_to_instr_type(branch_kind: int) -> str:
    if branch_kind == 1:
        return "Direct Conditional Branch"
    if branch_kind == 2:
        return "Direct Unconditional Branch"
    if branch_kind == 3:
        return "Indirect Branch"
    if branch_kind == 4:
        return "Other Branch"
    return "non-memory"


def _issue_group_from_code(issue_group: int) -> str:
    if issue_group == 1:
        return "FP"
    if issue_group == 2:
        return "LS"
    return "ALU"


def _branch_kind_to_mnemonic(branch_kind: int) -> str:
    if branch_kind == 1:
        return "b.cond"
    if branch_kind == 2:
        return "b"
    if branch_kind == 3:
        return "br"
    if branch_kind == 4:
        return "other_branch"
    return ""


def _assign_reg_dependencies(dep_set: set[int], mask: int, last_def_reg: list[int]) -> None:
    while mask:
        lsb = mask & -mask
        reg_idx = lsb.bit_length() - 1
        dep_id = last_def_reg[reg_idx]
        if dep_id:
            dep_set.add(dep_id)
        mask ^= lsb


def _update_def_registers(mask: int, instr_id: int, last_def_reg: list[int]) -> None:
    while mask:
        lsb = mask & -mask
        reg_idx = lsb.bit_length() - 1
        last_def_reg[reg_idx] = instr_id
        mask ^= lsb


def _classify_branch_type(mnemonic: str | None) -> str:
    mnemonic = (mnemonic or "").lower()
    if mnemonic.startswith("b."):
        return "Direct Conditional Branch"
    if mnemonic in ("cbz", "cbnz", "tbz", "tbnz"):
        return "Direct Conditional Branch"
    if mnemonic in ("b", "bl"):
        return "Direct Unconditional Branch"
    if mnemonic in ("br", "blr", "ret"):
        return "Indirect Branch"
    return "Other Branch"


def _parse_compact_memops(memops_field: str):
    if not memops_field:
        return []
    memops = []
    for item in memops_field.split(";"):
        if not item:
            continue
        parts = item.split(",", 2)
        if len(parts) != 3:
            continue
        kind, addr_hex, size_text = parts
        try:
            memops.append((kind, int(addr_hex, 16), int(size_text)))
        except ValueError:
            continue
    return memops


def _parse_compact_reg_field(field: str):
    if not field:
        return set()
    return {token for token in field.split(",") if token}


def _parse_compact_trace_cache(path: Path, use_shared_llc: bool, max_instructions: int = 0):
    record_types = Counter()
    markers = Counter()
    mnemonics = Counter()
    instruction_classify_arr = []
    branch_instr_classify = Counter()
    ls_instr_classify = Counter()
    icache_instr_classify = Counter()
    load_instr_list = []

    last_def_reg = [0] * 64
    last_store_line = {}
    cache_line_access = {}
    instrs_by_tid = defaultdict(list)

    line_size = 64
    if use_shared_llc:
        shared_llc = SharedLLCWithQueuing(
            size_bytes=8 * 1024 * 1024,
            assoc=16,
            line_size=line_size,
            hit_latency=35,
            num_banks=16,
            mshr_entries=64,
            mem_latency=200,
            mem_bandwidth_gbps=100,
            num_mem_channels=4,
        )
        thread_caches = {}

        def get_thread_cache(tid: int):
            if tid not in thread_caches:
                from src.cache import CacheLevel  # type: ignore

                l2 = CacheLevel("L2", 512 * 1024, 8, line_size, 12, lower=None)
                l1 = CacheLevel("L1", 32 * 1024, 8, line_size, 4, lower=l2)
                thread_caches[tid] = (l1, l2)
            return thread_caches[tid]

        current_time = 0
    else:
        l1, l2, l3, mem = build_cache_hierarchy()

    hit_level_counter = Counter()

    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if mm.size() < len(COMPACT_TRACE_MAGIC) or mm[: len(COMPACT_TRACE_MAGIC)] != COMPACT_TRACE_MAGIC:
                raise ValueError(f"Invalid compact trace cache header: {path}")

            offset = len(COMPACT_TRACE_MAGIC)
            parsed_instructions = 0
            mm_size = mm.size()
            while offset + COMPACT_TRACE_HEADER.size <= mm_size:
                (
                    instr_id,
                    tid,
                    issue_group_code,
                    branch_kind,
                    branch_taken_code,
                    memop_count,
                    pc_val,
                    uses_mask,
                    defs_mask,
                ) = COMPACT_TRACE_HEADER.unpack_from(mm, offset)
                offset += COMPACT_TRACE_HEADER.size
                parsed_instructions += 1
                if max_instructions > 0 and parsed_instructions > max_instructions:
                    break

                record_types["ifetch"] += 1
                icache_instr_classify["Instruction Fetch"] += 1
                icache_line = pc_val // line_size

                if branch_kind:
                    instr_type = _branch_kind_to_instr_type(branch_kind)
                    ins_obj = branchInstruction(
                        instr_type,
                        instr_id,
                        _branch_kind_to_mnemonic(branch_kind),
                        icache_line,
                        pc_val,
                    )
                    if branch_taken_code == 1:
                        ins_obj.branch_taken = False
                    elif branch_taken_code == 2:
                        ins_obj.branch_taken = True
                    branch_instr_classify[instr_type] += 1
                else:
                    ins_obj = Instr_NonMem("non-memory", instr_id, "", icache_line)
                    ins_obj.issue_group = _issue_group_from_code(issue_group_code)
                    ls_instr_classify["non-memory Instructions"] += 1

                _assign_reg_dependencies(ins_obj.reg_deps, uses_mask, last_def_reg)
                _update_def_registers(defs_mask, instr_id, last_def_reg)

                if use_shared_llc:
                    l1_local, l2_local = get_thread_cache(tid)
                memop_offset = offset
                for _ in range(memop_count):
                    if memop_offset + COMPACT_MEMOP_HEADER.size > mm_size:
                        raise ValueError(f"Truncated compact trace cache: {path}")
                    kind_code, addr, size = COMPACT_MEMOP_HEADER.unpack_from(mm, memop_offset)
                    memop_offset += COMPACT_MEMOP_HEADER.size

                    if kind_code == 0:
                        record_types["read"] += 1
                        if use_shared_llc:
                            lat, hit_level = l1_local.access(addr, is_write=False)
                            if hit_level == "MEM":
                                lat = shared_llc.access(addr, current_time)
                                hit_level = "L3" if lat <= 35 else "MEM"
                        else:
                            lat, hit_level = l1.access(addr, is_write=False)
                        hit_level_counter[hit_level] += 1
                        if not isinstance(ins_obj, Instr_Load):
                            new_ins = Instr_Load(
                                "load",
                                instr_id,
                                "",
                                addr,
                                size,
                                icache_line,
                                1,
                                reg_deps=ins_obj.reg_deps,
                                mem_deps=ins_obj.mem_deps,
                            )
                            new_ins.issue_group = "LS"
                            new_ins.load_cache_hit_levels.append(hit_level)
                            ins_obj = new_ins
                            ls_instr_classify["non-memory Instructions"] -= 1
                            ls_instr_classify["Load Instructions"] += 1
                            load_instr_list.append(new_ins)
                        else:
                            ins_obj.load_access_count += 1
                            ins_obj.load_cache_hit_levels.append(hit_level)

                        for cl in cache_lines_covered(addr, size, line_size):
                            if cl in last_store_line:
                                ins_obj.mem_deps.add(last_store_line[cl])
                            cache_line_access[cl] = instr_id
                    else:
                        record_types["write"] += 1
                        if use_shared_llc:
                            lat, hit_level = l1_local.access(addr, is_write=True)
                            if hit_level == "MEM":
                                lat = shared_llc.access(addr, current_time)
                                hit_level = "L3" if lat <= 35 else "MEM"
                        else:
                            lat, hit_level = l1.access(addr, is_write=True)
                        hit_level_counter[hit_level] += 1
                        if not isinstance(ins_obj, Instr_Store):
                            new_ins = Instr_Store(
                                "store",
                                instr_id,
                                "",
                                addr,
                                size,
                                icache_line,
                                reg_deps=ins_obj.reg_deps,
                            )
                            new_ins.issue_group = "LS"
                            ins_obj = new_ins
                            ls_instr_classify["non-memory Instructions"] -= 1
                            ls_instr_classify["Store Instructions"] += 1
                        for cl in cache_lines_covered(addr, size, line_size):
                            last_store_line[cl] = instr_id

                    if use_shared_llc:
                        current_time += 1

                offset = memop_offset
                instrs_by_tid[tid].append(ins_obj)
        finally:
            mm.close()

    instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    instruction_classify_arr.append(ls_instr_classify)
    instruction_classify_arr.append(icache_instr_classify)

    return (
        record_types,
        markers,
        mnemonics,
        instruction_classify_arr,
        load_instr_list,
        cache_line_access,
        hit_level_counter,
        instrs_by_tid,
    )


def parse_compact_trace_cache(path: str | Path, max_instructions: int = 0):
    return _parse_compact_trace_cache(Path(path), use_shared_llc=False, max_instructions=max_instructions)


def parse_compact_trace_cache_with_shared_llc(path: str | Path, max_instructions: int = 0):
    return _parse_compact_trace_cache(Path(path), use_shared_llc=True, max_instructions=max_instructions)


def parse_compact_trace_stream(lines: Iterable[str]):
    record_types = Counter()
    markers = Counter()
    mnemonics = Counter()
    instruction_classify_arr = []
    branch_instr_classify = Counter()
    ls_instr_classify = Counter()
    icache_instr_classify = Counter()
    load_instr_list = []

    last_def_reg = {}
    last_store_line = {}
    cache_line_access = {}
    instrs_by_tid = defaultdict(list)

    l1, l2, l3, mem = build_cache_hierarchy()
    hit_level_counter = Counter()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line or not line.startswith("I|"):
            continue

        parts = line.split("|", 8)
        if len(parts) != 9:
            continue

        _, instr_text, tid_text, pc_hex, mnem, uses_field, defs_field, taken_field, memops_field = parts
        try:
            instr_id = int(instr_text)
            tid = int(tid_text)
            pc_val = int(pc_hex, 16)
        except ValueError:
            continue

        record_types["ifetch"] += 1
        icache_instr_classify["Instruction Fetch"] += 1

        if mnem:
            mnemonics[mnem] += 1

        uses = _parse_compact_reg_field(uses_field)
        defs = _parse_compact_reg_field(defs_field)
        mnem_lower = (mnem or "").lower()
        is_branch = mnem_lower in BRANCH_MNEMS_EXACT or mnem_lower.startswith("b.")
        line_size = 64
        icache_line = pc_val // line_size

        if is_branch:
            branch_type = _classify_branch_type(mnem)
            ins_obj = branchInstruction(branch_type, instr_id, mnem, icache_line, pc_val)
            if taken_field == "1":
                ins_obj.branch_taken = True
            elif taken_field == "0":
                ins_obj.branch_taken = False
            branch_instr_classify[branch_type] += 1
        else:
            ins_obj = Instr_NonMem("non-memory", instr_id, mnem, icache_line)
            ls_instr_classify["non-memory Instructions"] += 1

        for u_reg in uses:
            if u_reg in last_def_reg:
                ins_obj.reg_deps.add(last_def_reg[u_reg])
        for d_reg in defs:
            last_def_reg[d_reg] = instr_id

        memops = _parse_compact_memops(memops_field)
        for kind, addr, size in memops:
            if kind == "R":
                record_types["read"] += 1
                lat, hit_level = l1.access(addr, is_write=False)
                hit_level_counter[hit_level] += 1
                if not isinstance(ins_obj, Instr_Load):
                    new_ins = Instr_Load(
                        "load",
                        instr_id,
                        mnem,
                        addr,
                        size,
                        icache_line,
                        1,
                        reg_deps=ins_obj.reg_deps,
                        mem_deps=ins_obj.mem_deps,
                    )
                    new_ins.load_cache_hit_levels.append(hit_level)
                    ins_obj = new_ins
                    ls_instr_classify["non-memory Instructions"] -= 1
                    ls_instr_classify["Load Instructions"] += 1
                    load_instr_list.append(new_ins)
                else:
                    ins_obj.load_access_count += 1
                    ins_obj.load_cache_hit_levels.append(hit_level)

                for cl in cache_lines_covered(addr, size, line_size):
                    if cl in last_store_line:
                        ins_obj.mem_deps.add(last_store_line[cl])
                    cache_line_access[cl] = instr_id
            elif kind == "W":
                record_types["write"] += 1
                lat, hit_level = l1.access(addr, is_write=True)
                hit_level_counter[hit_level] += 1
                if not isinstance(ins_obj, Instr_Store):
                    new_ins = Instr_Store(
                        "store",
                        instr_id,
                        mnem,
                        addr,
                        size,
                        icache_line,
                        reg_deps=ins_obj.reg_deps,
                    )
                    ins_obj = new_ins
                    ls_instr_classify["non-memory Instructions"] -= 1
                    ls_instr_classify["Store Instructions"] += 1
                for cl in cache_lines_covered(addr, size, line_size):
                    last_store_line[cl] = instr_id

        instrs_by_tid[tid].append(ins_obj)

    instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    instruction_classify_arr.append(ls_instr_classify)
    instruction_classify_arr.append(icache_instr_classify)

    return (
        record_types,
        markers,
        mnemonics,
        instruction_classify_arr,
        load_instr_list,
        cache_line_access,
        hit_level_counter,
        instrs_by_tid,
    )


def parse_compact_trace_with_shared_llc_stream(lines: Iterable[str]):
    record_types = Counter()
    markers = Counter()
    mnemonics = Counter()
    instruction_classify_arr = []
    branch_instr_classify = Counter()
    ls_instr_classify = Counter()
    icache_instr_classify = Counter()
    load_instr_list = []

    last_def_reg = {}
    last_store_line = {}
    cache_line_access = {}
    instrs_by_tid = defaultdict(list)

    line_size = 64
    shared_llc = SharedLLCWithQueuing(
        size_bytes=8 * 1024 * 1024,
        assoc=16,
        line_size=line_size,
        hit_latency=35,
        num_banks=16,
        mshr_entries=64,
        mem_latency=200,
        mem_bandwidth_gbps=100,
        num_mem_channels=4,
    )

    thread_caches = {}

    def get_thread_cache(tid: int):
        if tid not in thread_caches:
            from src.cache import CacheLevel  # type: ignore

            l2 = CacheLevel("L2", 512 * 1024, 8, line_size, 12, lower=None)
            l1 = CacheLevel("L1", 32 * 1024, 8, line_size, 4, lower=l2)
            thread_caches[tid] = (l1, l2)
        return thread_caches[tid]

    hit_level_counter = Counter()
    current_time = 0

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line or not line.startswith("I|"):
            continue

        parts = line.split("|", 8)
        if len(parts) != 9:
            continue

        _, instr_text, tid_text, pc_hex, mnem, uses_field, defs_field, taken_field, memops_field = parts
        try:
            instr_id = int(instr_text)
            tid = int(tid_text)
            pc_val = int(pc_hex, 16)
        except ValueError:
            continue

        record_types["ifetch"] += 1
        icache_instr_classify["Instruction Fetch"] += 1
        if mnem:
            mnemonics[mnem] += 1

        uses = _parse_compact_reg_field(uses_field)
        defs = _parse_compact_reg_field(defs_field)
        mnem_lower = (mnem or "").lower()
        is_branch = mnem_lower in BRANCH_MNEMS_EXACT or mnem_lower.startswith("b.")
        icache_line = pc_val // line_size

        if is_branch:
            branch_type = _classify_branch_type(mnem)
            ins_obj = branchInstruction(branch_type, instr_id, mnem, icache_line, pc_val)
            if taken_field == "1":
                ins_obj.branch_taken = True
            elif taken_field == "0":
                ins_obj.branch_taken = False
            branch_instr_classify[branch_type] += 1
        else:
            ins_obj = Instr_NonMem("non-memory", instr_id, mnem, icache_line)
            ls_instr_classify["non-memory Instructions"] += 1

        for u_reg in uses:
            if u_reg in last_def_reg:
                ins_obj.reg_deps.add(last_def_reg[u_reg])
        for d_reg in defs:
            last_def_reg[d_reg] = instr_id

        l1, l2 = get_thread_cache(tid)
        memops = _parse_compact_memops(memops_field)
        for kind, addr, size in memops:
            if kind == "R":
                record_types["read"] += 1
                lat, hit_level = l1.access(addr, is_write=False)
                if hit_level == "MEM":
                    lat = shared_llc.access(addr, current_time)
                    hit_level = "L3" if lat <= 35 else "MEM"
                hit_level_counter[hit_level] += 1

                if not isinstance(ins_obj, Instr_Load):
                    new_ins = Instr_Load(
                        "load",
                        instr_id,
                        mnem,
                        addr,
                        size,
                        icache_line,
                        1,
                        reg_deps=ins_obj.reg_deps,
                        mem_deps=ins_obj.mem_deps,
                    )
                    new_ins.load_cache_hit_levels.append(hit_level)
                    ins_obj = new_ins
                    ls_instr_classify["non-memory Instructions"] -= 1
                    ls_instr_classify["Load Instructions"] += 1
                    load_instr_list.append(new_ins)
                else:
                    ins_obj.load_access_count += 1
                    ins_obj.load_cache_hit_levels.append(hit_level)

                for cl in cache_lines_covered(addr, size, line_size):
                    if cl in last_store_line:
                        ins_obj.mem_deps.add(last_store_line[cl])
                    cache_line_access[cl] = instr_id
            elif kind == "W":
                record_types["write"] += 1
                lat, hit_level = l1.access(addr, is_write=True)
                if hit_level == "MEM":
                    lat = shared_llc.access(addr, current_time)
                    hit_level = "L3" if lat <= 35 else "MEM"
                hit_level_counter[hit_level] += 1

                if not isinstance(ins_obj, Instr_Store):
                    new_ins = Instr_Store(
                        "store",
                        instr_id,
                        mnem,
                        addr,
                        size,
                        icache_line,
                        reg_deps=ins_obj.reg_deps,
                    )
                    ins_obj = new_ins
                    ls_instr_classify["non-memory Instructions"] -= 1
                    ls_instr_classify["Store Instructions"] += 1
                for cl in cache_lines_covered(addr, size, line_size):
                    last_store_line[cl] = instr_id

            current_time += 1

        instrs_by_tid[tid].append(ins_obj)

    instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    instruction_classify_arr.append(ls_instr_classify)
    instruction_classify_arr.append(icache_instr_classify)

    return (
        record_types,
        markers,
        mnemonics,
        instruction_classify_arr,
        load_instr_list,
        cache_line_access,
        hit_level_counter,
        instrs_by_tid,
    )


def parse_trace_stream(lines: Iterable[str]):
    record_types = Counter()
    markers = Counter()
    mnemonics = Counter()
    instruction_classify_arr = []
    branch_instr_classify = Counter()
    ls_instr_classify = Counter()
    icache_instr_classify = Counter()
    load_instr_list = []

    last_def_reg = {}
    ifetch_info = {}
    last_store_line = {}
    cache_line_access = {}
    instr_table = {}

    l1, l2, l3, mem = build_cache_hierarchy()
    hit_level_counter = Counter()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        m_marker = RE_MARKER.search(line)
        if m_marker:
            markers[m_marker.group(1)] += 1
            continue

        m_head = RE_HEAD.match(line)
        if not m_head:
            continue

        instr_id = int(m_head.group("instr"))
        tid = int(m_head.group("tid"))
        rtype = m_head.group("rtype")
        record_types[rtype] += 1

        if rtype == "ifetch":
            icache_instr_classify["Instruction Fetch"] += 1

            m_mnem = RE_IFETCH_MNEM.match(line)
            mnem = m_mnem.group(1) if m_mnem else None
            if mnem:
                mnemonics[mnem] += 1

            m_pc = RE_IFETCH_PC.search(line)
            pc_val = int(m_pc.group(1), 16) if m_pc else 0
            uses, defs = extract_uses_defs_from_ifetch(line)

            mnem_lower = (mnem or "").lower()
            is_branch = mnem_lower in BRANCH_MNEMS_EXACT or mnem_lower.startswith("b.")
            line_size = 64
            icache_line = pc_val // line_size

            if is_branch:
                branch_type = _classify_branch_type(mnem)
                ins_obj = branchInstruction(branch_type, instr_id, mnem, icache_line, pc_val)
                m_status = RE_BRANCH_STATUS.search(line)
                if m_status:
                    ins_obj.branch_taken = m_status.group(1) == "taken"
                branch_instr_classify[branch_type] += 1
            else:
                ins_obj = Instr_NonMem("non-memory", instr_id, mnem, icache_line)
                ls_instr_classify["non-memory Instructions"] += 1

            for u_reg in uses:
                if u_reg in last_def_reg:
                    ins_obj.reg_deps.add(last_def_reg[u_reg])
            for d_reg in defs:
                last_def_reg[d_reg] = instr_id

            ifetch_info[(tid, instr_id)] = {
                "mnem": mnem,
                "pc": pc_val,
                "I_cache_line": icache_line,
                "ins_obj": ins_obj,
            }
            instr_table[(tid, instr_id)] = ins_obj
            continue

        if rtype not in ("read", "write"):
            continue

        m_rw = RE_RW_DETAIL.search(line)
        if not m_rw:
            continue

        addr = int(m_rw.group("addr"), 16)
        size = int(m_rw.group("size"))

        if (tid, instr_id) not in ifetch_info:
            continue

        info = ifetch_info[(tid, instr_id)]
        ins_obj = info["ins_obj"]
        line_size = 64
        icache_line = info["I_cache_line"]

        if rtype == "read":
            lat, hit_level = l1.access(addr, is_write=False)
            hit_level_counter[hit_level] += 1

            if not isinstance(ins_obj, Instr_Load):
                new_ins = Instr_Load(
                    "load",
                    instr_id,
                    info["mnem"],
                    addr,
                    size,
                    icache_line,
                    1,
                    reg_deps=ins_obj.reg_deps,
                    mem_deps=ins_obj.mem_deps,
                )
                new_ins.load_cache_hit_levels.append(hit_level)
                ins_obj = new_ins
                instr_table[(tid, instr_id)] = ins_obj
                info["ins_obj"] = ins_obj
                ls_instr_classify["non-memory Instructions"] -= 1
                ls_instr_classify["Load Instructions"] += 1
                load_instr_list.append(new_ins)
            else:
                ins_obj.load_access_count += 1
                ins_obj.load_cache_hit_levels.append(hit_level)

            for cl in cache_lines_covered(addr, size, line_size):
                if cl in last_store_line:
                    ins_obj.mem_deps.add(last_store_line[cl])
                cache_line_access[cl] = instr_id
            continue

        lat, hit_level = l1.access(addr, is_write=True)
        hit_level_counter[hit_level] += 1

        if not isinstance(ins_obj, Instr_Store):
            new_ins = Instr_Store(
                "store",
                instr_id,
                info["mnem"],
                addr,
                size,
                icache_line,
                reg_deps=ins_obj.reg_deps,
            )
            ins_obj = new_ins
            instr_table[(tid, instr_id)] = ins_obj
            info["ins_obj"] = ins_obj
            ls_instr_classify["non-memory Instructions"] -= 1
            ls_instr_classify["Store Instructions"] += 1

        for cl in cache_lines_covered(addr, size, line_size):
            last_store_line[cl] = instr_id

    instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    instruction_classify_arr.append(ls_instr_classify)
    instruction_classify_arr.append(icache_instr_classify)

    instrs_by_tid = defaultdict(list)
    for (tid, iid), ins in instr_table.items():
        instrs_by_tid[tid].append(ins)
    for tid in instrs_by_tid:
        instrs_by_tid[tid].sort(key=lambda x: x.instr_id)

    return (
        record_types,
        markers,
        mnemonics,
        instruction_classify_arr,
        load_instr_list,
        cache_line_access,
        hit_level_counter,
        instrs_by_tid,
    )


def parse_trace_with_shared_llc_stream(lines: Iterable[str]):
    record_types = Counter()
    markers = Counter()
    mnemonics = Counter()
    instruction_classify_arr = []
    branch_instr_classify = Counter()
    ls_instr_classify = Counter()
    icache_instr_classify = Counter()
    load_instr_list = []

    last_def_reg = {}
    ifetch_info = {}
    last_store_line = {}
    cache_line_access = {}
    instr_table = {}

    line_size = 64
    shared_llc = SharedLLCWithQueuing(
        size_bytes=8 * 1024 * 1024,
        assoc=16,
        line_size=line_size,
        hit_latency=35,
        num_banks=16,
        mshr_entries=64,
        mem_latency=200,
        mem_bandwidth_gbps=100,
        num_mem_channels=4,
    )

    thread_caches = {}

    def get_thread_cache(tid: int):
        if tid not in thread_caches:
            from src.cache import CacheLevel  # type: ignore

            l2 = CacheLevel("L2", 512 * 1024, 8, line_size, 12, lower=None)
            l1 = CacheLevel("L1", 32 * 1024, 8, line_size, 4, lower=l2)
            thread_caches[tid] = (l1, l2)
        return thread_caches[tid]

    hit_level_counter = Counter()
    current_time = 0

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        m_marker = RE_MARKER.search(line)
        if m_marker:
            markers[m_marker.group(1)] += 1
            continue

        m_head = RE_HEAD.match(line)
        if not m_head:
            continue

        instr_id = int(m_head.group("instr"))
        tid = int(m_head.group("tid"))
        rtype = m_head.group("rtype")
        record_types[rtype] += 1

        if rtype == "ifetch":
            icache_instr_classify["Instruction Fetch"] += 1

            m_mnem = RE_IFETCH_MNEM.match(line)
            mnem = m_mnem.group(1) if m_mnem else None
            if mnem:
                mnemonics[mnem] += 1

            m_pc = RE_IFETCH_PC.search(line)
            pc_val = int(m_pc.group(1), 16) if m_pc else 0
            uses, defs = extract_uses_defs_from_ifetch(line)

            mnem_lower = (mnem or "").lower()
            is_branch = mnem_lower in BRANCH_MNEMS_EXACT or mnem_lower.startswith("b.")
            icache_line = pc_val // line_size

            if is_branch:
                ins_obj = branchInstruction("branch", instr_id, mnem, icache_line, pc_val)
                m_status = RE_BRANCH_STATUS.search(line)
                if m_status:
                    ins_obj.branch_taken = m_status.group(1) == "taken"
                branch_instr_classify["Branch Instructions"] += 1
            else:
                ins_obj = Instr_NonMem("non-memory", instr_id, mnem, icache_line)
                ls_instr_classify["non-memory Instructions"] += 1

            for u_reg in uses:
                if u_reg in last_def_reg:
                    ins_obj.reg_deps.add(last_def_reg[u_reg])
            for d_reg in defs:
                last_def_reg[d_reg] = instr_id

            ifetch_info[(tid, instr_id)] = {
                "mnem": mnem,
                "pc": pc_val,
                "I_cache_line": icache_line,
                "ins_obj": ins_obj,
            }
            instr_table[(tid, instr_id)] = ins_obj
            continue

        if rtype not in ("read", "write"):
            continue

        m_rw = RE_RW_DETAIL.search(line)
        if not m_rw:
            continue

        addr = int(m_rw.group("addr"), 16)
        size = int(m_rw.group("size"))

        if (tid, instr_id) not in ifetch_info:
            continue

        info = ifetch_info[(tid, instr_id)]
        ins_obj = info["ins_obj"]
        icache_line = info["I_cache_line"]
        l1, l2 = get_thread_cache(tid)

        if rtype == "read":
            l1_lat, l1_hit = l1.access(addr, is_write=False, count_writeback_cost=False)
            if l1_hit == "L1":
                hit_level = "L1"
                hit_level_counter[hit_level] += 1
            else:
                l2_lat, l2_hit = l2.access(addr, is_write=False, count_writeback_cost=False)
                if l2_hit == "L2":
                    hit_level = "L2"
                    hit_level_counter[hit_level] += 1
                else:
                    llc_lat, llc_hit, comp_time = shared_llc.access(addr, current_time, tid, is_write=False)
                    hit_level = llc_hit
                    hit_level_counter[hit_level] += 1
                    current_time = comp_time

            if not isinstance(ins_obj, Instr_Load):
                new_ins = Instr_Load(
                    "load",
                    instr_id,
                    info["mnem"],
                    addr,
                    size,
                    icache_line,
                    1,
                    reg_deps=ins_obj.reg_deps,
                    mem_deps=ins_obj.mem_deps,
                    line_size=line_size,
                )
                new_ins.load_cache_hit_levels.append(hit_level)
                ins_obj = new_ins
                instr_table[(tid, instr_id)] = ins_obj
                info["ins_obj"] = ins_obj
                ls_instr_classify["non-memory Instructions"] -= 1
                ls_instr_classify["Load Instructions"] += 1
                load_instr_list.append(new_ins)
            else:
                ins_obj.load_access_count += 1
                ins_obj.load_cache_hit_levels.append(hit_level)

            for cl in cache_lines_covered(addr, size, line_size):
                if cl in last_store_line:
                    ins_obj.mem_deps.add(last_store_line[cl])
                cache_line_access[cl] = instr_id
            continue

        l1_lat, l1_hit = l1.access(addr, is_write=True, count_writeback_cost=False)
        if l1_hit == "L1":
            hit_level = "L1"
            hit_level_counter[hit_level] += 1
        else:
            l2_lat, l2_hit = l2.access(addr, is_write=True, count_writeback_cost=False)
            if l2_hit == "L2":
                hit_level = "L2"
                hit_level_counter[hit_level] += 1
            else:
                llc_lat, llc_hit, comp_time = shared_llc.access(addr, current_time, tid, is_write=True)
                hit_level = llc_hit
                hit_level_counter[hit_level] += 1
                current_time = comp_time

        if not isinstance(ins_obj, Instr_Store):
            new_ins = Instr_Store(
                "store",
                instr_id,
                info["mnem"],
                addr,
                size,
                icache_line,
                reg_deps=ins_obj.reg_deps,
            )
            ins_obj = new_ins
            instr_table[(tid, instr_id)] = ins_obj
            info["ins_obj"] = ins_obj
            ls_instr_classify["non-memory Instructions"] -= 1
            ls_instr_classify["Store Instructions"] += 1

        for cl in cache_lines_covered(addr, size, line_size):
            last_store_line[cl] = instr_id

    instruction_classify_arr.append(branch_instr_classify)
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Load Instructions"]
    ls_instr_classify["non-memory Instructions"] -= ls_instr_classify["Store Instructions"]
    instruction_classify_arr.append(ls_instr_classify)
    instruction_classify_arr.append(icache_instr_classify)

    instrs_by_tid = defaultdict(list)
    for (tid, iid), ins in instr_table.items():
        instrs_by_tid[tid].append(ins)
    for tid in instrs_by_tid:
        instrs_by_tid[tid].sort(key=lambda x: x.instr_id)

    shared_llc.print_stats()

    return (
        record_types,
        markers,
        mnemonics,
        instruction_classify_arr,
        load_instr_list,
        cache_line_access,
        hit_level_counter,
        instrs_by_tid,
    )
