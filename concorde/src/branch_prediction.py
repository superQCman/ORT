#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Branch prediction models including simple and TAGE predictors.
"""

import random
from dataclasses import dataclass
from collections import defaultdict


def classify_branch_type(mn: str) -> str:
    """
    Classify branch instruction type.
    
    Args:
        mn: Mnemonic string
        
    Returns:
        str: Branch type classification
    """
    mn = (mn or "").lower()
    if mn.startswith("b."):
        return "Conditional Branch"
    if mn in ("cbz", "cbnz", "tbz", "tbnz"):
        return "Conditional Branch"
    if mn in ("b", "bl"):
        return "Unconditional Branch"
    if mn in ("br", "blr", "ret"):
        return "Indirect Branch"
    return "Other Branch"


class SimplePredictor:
    """
    Simple random branch predictor with fixed misprediction rate.
    """
    
    def __init__(self, p: float = 0.05, seed: int = 1):
        self.p = p  # misprediction rate
        self.rng = random.Random(seed)

    @staticmethod
    def _default_pred(pc: int):
        """Default prediction: taken if PC is odd, not-taken if even."""
        return (pc & 1) == 1

    def update_and_count(self, pc: int, actual_taken: bool):
        """
        Predict and update, returning (mispredicted, prediction).
        
        Args:
            pc: Program counter
            actual_taken: Actual branch outcome
            
        Returns:
            tuple: (is_mispredicted, prediction)
        """
        pred = self._default_pred(pc)
        is_misp = (self.rng.random() < self.p)
        return is_misp, pred


@dataclass
class TageEntry:
    """TAGE table entry."""
    ctr: int = 0        # prediction counter
    tag: int = 0        # tag bits
    u: int = 0          # usefulness counter


class TAGEPredictor:
    """
    Research-grade simplified TAGE-like predictor (direction only).
    - Base: bimodal 2-bit counter table indexed by PC.
    - Tagged tables: multiple history lengths, each with (idx, tag) from (PC, folded history).
    - Provider: longest history table with matching tag; Alternate: next-longest.
    """
    
    def __init__(self,
                 num_tables: int = 8,
                 table_size: int = 2048,
                 tag_bits: int = 10,
                 hist_lengths=None,
                 ghr_bits: int = 200,
                 base_size: int = 4096,
                 ctr_bits: int = 3,
                 u_bits: int = 2,
                 seed: int = 1):
        self.num_tables = num_tables
        self.table_size = table_size
        self.tag_bits = tag_bits
        self.ghr_bits = ghr_bits
        self.base_size = base_size
        self.ctr_bits = ctr_bits
        self.u_bits = u_bits
        self.rng = random.Random(seed)
        
        # History lengths: geometric series
        if hist_lengths is None:
            hist_lengths = []
            L = 2
            for _ in range(num_tables):
                hist_lengths.append(min(L, ghr_bits))
                L = int(L * 1.4)
        self.hist_lengths = hist_lengths
        
        # Base predictor: bimodal 2-bit counters
        self.base_table = [0] * base_size
        
        # Tagged tables
        self.tables = []
        for _ in range(num_tables):
            self.tables.append([TageEntry() for _ in range(table_size)])
        
        # Global history register
        self.ghr = 0

    @staticmethod
    def _saturate(val: int, bits: int):
        """Saturate value to [0, 2^bits - 1]."""
        max_val = (1 << bits) - 1
        if val < 0:
            return 0
        if val > max_val:
            return max_val
        return val

    def _get_history_bits(self, L: int) -> int:
        """Extract L bits from GHR."""
        mask = (1 << L) - 1
        return self.ghr & mask

    def _fold(self, hist: int, L: int, out_bits: int) -> int:
        """
        Fold L-bit history into out_bits.
        Simple XOR folding.
        """
        result = 0
        for i in range(0, L, out_bits):
            chunk = (hist >> i) & ((1 << out_bits) - 1)
            result ^= chunk
        return result & ((1 << out_bits) - 1)

    def _idx_tag(self, pc: int, table_i: int):
        """
        Compute index and tag for table_i.
        Uses folded history XORed with PC bits.
        """
        L = self.hist_lengths[table_i]
        hist_bits = self._get_history_bits(L)
        
        # Index: fold history to log2(table_size) bits and XOR with PC
        idx_bits = self.table_size.bit_length() - 1
        folded_idx = self._fold(hist_bits, L, idx_bits)
        idx = (folded_idx ^ (pc & ((1 << idx_bits) - 1))) % self.table_size
        
        # Tag: fold history to tag_bits and XOR with different PC bits
        folded_tag = self._fold(hist_bits, L, self.tag_bits)
        tag = folded_tag ^ ((pc >> idx_bits) & ((1 << self.tag_bits) - 1))
        
        return idx, tag

    def _base_pred(self, pc: int) -> bool:
        """Base predictor prediction."""
        idx = pc % self.base_size
        return self.base_table[idx] >= (1 << (self.ctr_bits - 1))

    def _ctr_pred(self, ctr: int) -> bool:
        """Counter prediction: taken if >= threshold."""
        threshold = 1 << (self.ctr_bits - 1)
        return ctr >= threshold

    def _update_ctr(self, ctr: int, taken: bool) -> int:
        """Update counter towards taken/not-taken."""
        if taken:
            return self._saturate(ctr + 1, self.ctr_bits)
        else:
            return self._saturate(ctr - 1, self.ctr_bits)

    def _update_base(self, pc: int, taken: bool):
        """Update base predictor."""
        idx = pc % self.base_size
        if taken:
            self.base_table[idx] = self._saturate(self.base_table[idx] + 1, self.ctr_bits)
        else:
            self.base_table[idx] = self._saturate(self.base_table[idx] - 1, self.ctr_bits)

    def predict(self, pc: int):
        """
        Make prediction.
        
        Returns:
            tuple: (prediction, provider_table, alt_prediction)
        """
        base_pred = self._base_pred(pc)
        
        # Search for provider: longest history table with matching tag
        provider = None
        provider_pred = base_pred
        alt_pred = base_pred
        
        for i in range(self.num_tables - 1, -1, -1):
            idx, tag = self._idx_tag(pc, i)
            entry = self.tables[i][idx]
            if entry.tag == tag:
                if provider is None:
                    provider = i
                    provider_pred = self._ctr_pred(entry.ctr)
                else:
                    alt_pred = self._ctr_pred(entry.ctr)
                    break
        
        return provider_pred, provider, alt_pred

    def update(self, pc: int, actual_taken: bool, pred: bool, provider, alt_pred: bool):
        """
        Update predictor state.
        
        Args:
            pc: Program counter
            actual_taken: Actual branch outcome
            pred: Prediction made
            provider: Provider table index (or None for base)
            alt_pred: Alternate prediction
        """
        # Update base predictor
        self._update_base(pc, actual_taken)
        
        # Update provider
        if provider is not None:
            idx, tag = self._idx_tag(pc, provider)
            entry = self.tables[provider][idx]
            entry.ctr = self._update_ctr(entry.ctr, actual_taken)
            
            # Update usefulness
            if pred != alt_pred:
                if pred == actual_taken:
                    entry.u = self._saturate(entry.u + 1, self.u_bits)
                else:
                    entry.u = self._saturate(entry.u - 1, self.u_bits)
        
        # Allocate new entry on misprediction
        if pred != actual_taken:
            # Try to allocate in longer history table
            for i in range(self.num_tables - 1, -1, -1):
                if provider is None or i > provider:
                    idx, tag = self._idx_tag(pc, i)
                    entry = self.tables[i][idx]
                    # Replace if usefulness is 0
                    if entry.u == 0 or self.rng.random() < 0.1:
                        entry.tag = tag
                        entry.ctr = (1 << (self.ctr_bits - 1)) + (1 if actual_taken else -1)
                        entry.u = 0
                        break
        
        # Update GHR
        self.ghr = ((self.ghr << 1) | (1 if actual_taken else 0)) & ((1 << self.ghr_bits) - 1)

    def update_and_count(self, pc: int, actual_taken: bool):
        """
        Predict, update, and return misprediction info.
        
        Args:
            pc: Program counter
            actual_taken: Actual branch outcome
            
        Returns:
            tuple: (is_mispredicted, prediction)
        """
        pred, provider, alt_pred = self.predict(pc)
        self.update(pc, actual_taken, pred, provider, alt_pred)
        is_misp = (pred != actual_taken)
        return is_misp, pred


def compute_branch_mispred_rate(instrs_sorted,
                                predictor,
                                only_conditional: bool = True,
                                require_label: bool = True):
    """
    Run branch prediction on instruction stream and compute misprediction rate.

    Args:
        instrs_sorted: List of instructions
        predictor: Branch predictor instance
        only_conditional: Only count conditional branches
        require_label: Skip branches with branch_taken is None

    Returns:
        dict: Statistics including total, misses, rate, and per-type breakdown
    """
    total = 0
    misp = 0
    by_type = defaultdict(lambda: {"total": 0, "misp": 0})

    for ins in instrs_sorted:
        if ins.instr_type != "branch":
            continue
        
        if require_label and ins.branch_taken is None:
            continue
        
        br_type = classify_branch_type(ins.mnemonic)
        
        if only_conditional:
            if br_type != "Conditional Branch":
                continue
        
        pc = ins.branch_instr_pc
        actual_taken = ins.branch_taken
        
        is_misp, pred = predictor.update_and_count(pc, actual_taken)
        
        total += 1
        by_type[br_type]["total"] += 1
        
        if is_misp:
            misp += 1
            by_type[br_type]["misp"] += 1

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
