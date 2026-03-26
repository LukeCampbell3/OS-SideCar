"""
ChampSim integration for AC-MoE-GA Systems Sidecar.

Bridges ChampSim CPU microarchitecture simulation with the sidecar's
advisory recommendations for page policy, prefetch, NUMA placement,
and cache management decisions.
"""

__version__ = "0.1.0"

from .trace_parser import ChampSimTraceParser, ChampSimRecord
from .event_bridge import ChampSimEventBridge
from .simulator import ChampSimSidecarSimulator
from .metrics import SimulationMetrics, MetricsCollector
