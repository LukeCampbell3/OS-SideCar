"""
Lightweight cache hierarchy model for evaluating sidecar recommendations.

Models a simplified cache hierarchy (L1/L2/LLC) to measure how sidecar
page/prefetch/placement decisions affect hit rates and latency.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum, auto


class EvictionPolicy(Enum):
    LRU = auto()
    SIDECAR_GUIDED = auto()  # Uses sidecar page-state hints


@dataclass
class CacheStats:
    """Per-level cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    sidecar_preserves: int = 0   # Pages sidecar said to keep
    sidecar_reclaims: int = 0    # Pages sidecar said to evict

    @property
    def total_accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.total_accesses)

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class CacheLevel:
    """Single cache level with LRU or sidecar-guided eviction."""

    def __init__(self, name: str, num_sets: int, ways: int, line_size: int = 64):
        self.name = name
        self.num_sets = num_sets
        self.ways = ways
        self.line_size = line_size
        self.stats = CacheStats()

        # sets[set_index] = OrderedDict of {tag: metadata}
        self._sets: Dict[int, OrderedDict] = {
            i: OrderedDict() for i in range(num_sets)
        }
        # Sidecar hints: cache_line -> preserve_score
        self._preserve_hints: Dict[int, float] = {}

    def access(self, cache_line: int, is_write: bool = False) -> bool:
        """Access a cache line. Returns True on hit."""
        set_idx = cache_line % self.num_sets
        tag = cache_line // self.num_sets
        cache_set = self._sets[set_idx]

        if tag in cache_set:
            # Hit — move to MRU
            cache_set.move_to_end(tag)
            self.stats.hits += 1
            return True

        # Miss — insert
        self.stats.misses += 1
        if len(cache_set) >= self.ways:
            self._evict(set_idx, cache_line)

        cache_set[tag] = {"write": is_write, "line": cache_line}
        return False

    def prefetch(self, cache_line: int) -> bool:
        """Insert a prefetched line. Returns True if it was already present."""
        set_idx = cache_line % self.num_sets
        tag = cache_line // self.num_sets
        cache_set = self._sets[set_idx]

        if tag in cache_set:
            self.stats.prefetch_hits += 1
            return True

        self.stats.prefetch_misses += 1
        if len(cache_set) >= self.ways:
            self._evict(set_idx, cache_line)

        cache_set[tag] = {"write": False, "line": cache_line, "prefetched": True}
        return False

    def set_preserve_hint(self, cache_line: int, score: float):
        """Sidecar says this line should be preserved (higher = more important)."""
        self._preserve_hints[cache_line] = score

    def _evict(self, set_idx: int, incoming_line: int):
        """Evict a line from the set, considering sidecar hints."""
        cache_set = self._sets[set_idx]
        self.stats.evictions += 1

        # Check if any lines have low preserve scores (sidecar-guided eviction)
        best_victim_tag = None
        lowest_score = float("inf")

        for tag, meta in cache_set.items():
            line = meta.get("line", 0)
            score = self._preserve_hints.get(line, 0.0)
            if score < lowest_score:
                lowest_score = score
                best_victim_tag = tag

        if best_victim_tag is not None and lowest_score < 0.5:
            # Sidecar-guided: evict the line with lowest preserve score
            del cache_set[best_victim_tag]
            self.stats.sidecar_reclaims += 1
        else:
            # LRU fallback: evict least recently used
            cache_set.popitem(last=False)

    def clear_hints(self):
        self._preserve_hints.clear()


@dataclass
class CacheHierarchyConfig:
    """Configuration for the simulated cache hierarchy."""
    # L1D: 32KB, 8-way, 64B lines → 64 sets
    l1d_sets: int = 64
    l1d_ways: int = 8
    # L2: 256KB, 8-way → 512 sets
    l2_sets: int = 512
    l2_ways: int = 8
    # LLC: 2MB, 16-way → 2048 sets
    llc_sets: int = 2048
    llc_ways: int = 16
    line_size: int = 64


class CacheHierarchy:
    """Three-level cache hierarchy with sidecar integration points."""

    def __init__(self, config: Optional[CacheHierarchyConfig] = None):
        config = config or CacheHierarchyConfig()
        self.l1d = CacheLevel("L1D", config.l1d_sets, config.l1d_ways, config.line_size)
        self.l2 = CacheLevel("L2", config.l2_sets, config.l2_ways, config.line_size)
        self.llc = CacheLevel("LLC", config.llc_sets, config.llc_ways, config.line_size)
        self._levels = [self.l1d, self.l2, self.llc]

    def access(self, address: int, is_write: bool = False) -> Tuple[str, int]:
        """
        Access the cache hierarchy.
        
        Returns (hit_level, latency_cycles).
        """
        cache_line = address >> 6

        if self.l1d.access(cache_line, is_write):
            return "L1D", 4

        if self.l2.access(cache_line, is_write):
            return "L2", 12

        if self.llc.access(cache_line, is_write):
            return "LLC", 40

        # DRAM access
        return "DRAM", 200

    def prefetch_line(self, address: int) -> str:
        """Prefetch into LLC. Returns level where it was found (or 'DRAM')."""
        cache_line = address >> 6
        if self.llc.prefetch(cache_line):
            return "LLC_HIT"
        return "DRAM_FILL"

    def apply_sidecar_hints(self, page: int, page_state: Dict[str, float]):
        """
        Apply sidecar page state to cache eviction hints.
        
        Uses actual inferred page state:
        - cold, reclaimable → evict (low score)
        - burst_hot, recently_reused → preserve (high score)
        """
        base_line = page << 6  # 64 cache lines per 4KB page
        
        # Compute preserve score from page state
        # High score = keep, low score = evict
        cold = page_state.get("cold", 0.0)
        reclaimable = page_state.get("reclaimable", 0.0)
        burst_hot = page_state.get("burst_hot", 0.0)
        recently_reused = page_state.get("recently_reused", 0.0)
        
        # Preserve if hot/recently used, evict if cold/reclaimable
        # Score = 1.0 - (cold + reclaimable) + (burst_hot + recently_reused) / 2
        net_score = 1.0 - cold - reclaimable * 0.5 + burst_hot * 0.3 + recently_reused * 0.3
        net_score = max(0.0, min(1.0, net_score))
        
        for offset in range(64):
            for level in self._levels:
                level.set_preserve_hint(base_line + offset, net_score)

    def get_stats(self) -> Dict[str, CacheStats]:
        return {level.name: level.stats for level in self._levels}

    def get_summary(self) -> Dict[str, float]:
        stats = self.get_stats()
        return {
            f"{name}_hit_rate": s.hit_rate
            for name, s in stats.items()
        }
