"""
Synthetic trace generators for testing the ChampSim integration
without requiring real ChampSim trace files.

Generates traces with known access patterns so we can validate
that the sidecar's recommendations actually improve cache behavior.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class SyntheticTraceConfig:
    """Configuration for synthetic trace generation."""
    num_instructions: int = 100_000
    memory_ratio: float = 0.35       # Fraction of instructions that access memory
    branch_ratio: float = 0.15       # Fraction that are branches
    write_ratio: float = 0.25        # Fraction of memory ops that are stores
    seed: int = 42


def generate_streaming_trace(config: SyntheticTraceConfig, output_path: str):
    """
    Generate a streaming access pattern trace.
    Sequential scan through memory with occasional jumps.
    This should benefit from prefetch recommendations.
    """
    rng = np.random.RandomState(config.seed)
    path = Path(output_path)

    base_addr = 0x7FFF_0000_0000
    stride = 64  # Cache line stride
    current_addr = base_addr

    with open(path, "wb") as f:
        for i in range(config.num_instructions):
            ip = 0x400000 + (i % 1024) * 4
            is_branch = rng.random() < config.branch_ratio
            branch_taken = is_branch and rng.random() < 0.6
            has_memory = rng.random() < config.memory_ratio
            is_write = has_memory and rng.random() < config.write_ratio

            src_mem = 0
            dst_mem = 0

            if has_memory:
                # Streaming: advance by stride with occasional jumps
                if rng.random() < 0.05:
                    current_addr = base_addr + rng.randint(0, 1 << 20) * 64
                else:
                    current_addr += stride

                if is_write:
                    dst_mem = current_addr
                else:
                    src_mem = current_addr

            _write_binary_record(f, ip, is_branch, branch_taken, src_mem, dst_mem)

    print(f"Generated streaming trace: {path} ({config.num_instructions:,} instructions)")


def generate_hotcold_trace(config: SyntheticTraceConfig, output_path: str):
    """
    Generate a hot/cold access pattern trace.
    Small hot working set with occasional cold accesses.
    This should benefit from page preserve/reclaim recommendations.
    """
    rng = np.random.RandomState(config.seed)
    path = Path(output_path)

    hot_base = 0x7FFF_0000_0000
    cold_base = 0x7FFF_1000_0000
    hot_pages = 16    # 64KB hot working set
    cold_pages = 4096  # 16MB cold region

    with open(path, "wb") as f:
        for i in range(config.num_instructions):
            ip = 0x400000 + (i % 2048) * 4
            is_branch = rng.random() < config.branch_ratio
            branch_taken = is_branch and rng.random() < 0.6
            has_memory = rng.random() < config.memory_ratio
            is_write = has_memory and rng.random() < config.write_ratio

            src_mem = 0
            dst_mem = 0

            if has_memory:
                if rng.random() < 0.8:
                    # Hot access (80%)
                    page = rng.randint(0, hot_pages)
                    offset = rng.randint(0, 64) * 64
                    addr = hot_base + page * 4096 + offset
                else:
                    # Cold access (20%)
                    page = rng.randint(0, cold_pages)
                    offset = rng.randint(0, 64) * 64
                    addr = cold_base + page * 4096 + offset

                if is_write:
                    dst_mem = addr
                else:
                    src_mem = addr

            _write_binary_record(f, ip, is_branch, branch_taken, src_mem, dst_mem)

    print(f"Generated hot/cold trace: {path} ({config.num_instructions:,} instructions)")


def generate_phase_trace(config: SyntheticTraceConfig, output_path: str):
    """
    Generate a phase-changing workload trace.
    Alternates between compute-heavy, memory-heavy, and mixed phases.
    Tests the sidecar's phase detection and adaptive behavior.
    """
    rng = np.random.RandomState(config.seed)
    path = Path(output_path)

    phase_length = config.num_instructions // 4
    bases = [0x7FFF_0000_0000, 0x7FFF_2000_0000, 0x7FFF_4000_0000]

    with open(path, "wb") as f:
        for i in range(config.num_instructions):
            phase = (i // phase_length) % 4

            ip = 0x400000 + (i % 4096) * 4
            is_branch = rng.random() < config.branch_ratio

            if phase == 0:
                # Compute-heavy: few memory accesses
                mem_ratio = 0.1
                write_ratio = 0.1
                base = bases[0]
                pages = 8
            elif phase == 1:
                # Memory-heavy streaming
                mem_ratio = 0.6
                write_ratio = 0.15
                base = bases[1]
                pages = 256
            elif phase == 2:
                # Mixed with high write pressure
                mem_ratio = 0.4
                write_ratio = 0.5
                base = bases[2]
                pages = 64
            else:
                # Random / thrashing
                mem_ratio = 0.5
                write_ratio = 0.3
                base = bases[rng.randint(0, 3)]
                pages = 1024

            branch_taken = is_branch and rng.random() < 0.6
            has_memory = rng.random() < mem_ratio
            is_write = has_memory and rng.random() < write_ratio

            src_mem = 0
            dst_mem = 0

            if has_memory:
                page = rng.randint(0, pages)
                offset = rng.randint(0, 64) * 64
                addr = base + page * 4096 + offset

                if is_write:
                    dst_mem = addr
                else:
                    src_mem = addr

            _write_binary_record(f, ip, is_branch, branch_taken, src_mem, dst_mem)

    print(f"Generated phase trace: {path} ({config.num_instructions:,} instructions)")


def _write_binary_record(f, ip: int, is_branch: bool, branch_taken: bool,
                         src_mem: int, dst_mem: int):
    """Write a single binary trace record (64 bytes)."""
    buf = bytearray(64)
    struct.pack_into("<Q", buf, 0, ip)
    buf[8] = 1 if is_branch else 0
    buf[9] = 1 if branch_taken else 0
    # dst_registers[6] at offset 10-15 (zeros)
    # src_registers[6] at offset 16-21 (zeros)
    struct.pack_into("<Q", buf, 24, src_mem)
    struct.pack_into("<Q", buf, 32, dst_mem)
    # padding at 40-63 (zeros)
    f.write(bytes(buf))
