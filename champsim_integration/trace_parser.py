"""
ChampSim trace parser.

Reads ChampSim trace format records and yields structured records.
Supports both text-based and binary trace formats.

ChampSim trace format (per record):
  - IP (instruction pointer / PC)
  - is_branch (bool)
  - branch_taken (bool)
  - destination_register(s)
  - source_register(s)
  - source_memory (virtual address, 0 if none)
  - destination_memory (virtual address, 0 if none)
"""

import struct
import lzma
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, BinaryIO, Union
import logging

logger = logging.getLogger(__name__)

# ChampSim binary trace record: 8 fields packed
# IP(8) + is_branch(1) + branch_taken(1) + dst_regs(6) + src_regs(6) + src_mem(8) + dst_mem(8)
BINARY_RECORD_SIZE = 64  # bytes per record in binary format


@dataclass(slots=True)
class ChampSimRecord:
    """Single instruction record from a ChampSim trace."""
    ip: int                    # Instruction pointer
    is_branch: bool
    branch_taken: bool
    dst_registers: list        # Up to 2 destination registers
    src_registers: list        # Up to 3 source registers
    source_memory: int         # Virtual address of load (0 = no memory)
    destination_memory: int    # Virtual address of store (0 = no memory)
    instruction_id: int = 0    # Sequential ID assigned during parsing

    @property
    def is_load(self) -> bool:
        return self.source_memory != 0

    @property
    def is_store(self) -> bool:
        return self.destination_memory != 0

    @property
    def has_memory_access(self) -> bool:
        return self.is_load or self.is_store

    @property
    def virtual_page(self) -> Optional[int]:
        """Primary memory access page (load takes priority)."""
        if self.source_memory:
            return self.source_memory >> 12
        if self.destination_memory:
            return self.destination_memory >> 12
        return None

    @property
    def cache_line(self) -> Optional[int]:
        """Cache line address (64-byte aligned)."""
        if self.source_memory:
            return self.source_memory >> 6
        if self.destination_memory:
            return self.destination_memory >> 6
        return None


class ChampSimTraceParser:
    """
    Parser for ChampSim trace files.
    
    Supports:
    - .champsimtrace.xz (lzma compressed binary)
    - .champsimtrace.gz (gzip compressed binary)
    - .champsimtrace (uncompressed binary)
    - .trace.txt (text format, one instruction per line)
    """

    def __init__(self, trace_path: Union[str, Path], max_records: Optional[int] = None):
        self.trace_path = Path(trace_path)
        self.max_records = max_records
        self._record_count = 0

        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_path}")

    def parse(self) -> Iterator[ChampSimRecord]:
        """Yield ChampSimRecord from the trace file."""
        suffix = "".join(self.trace_path.suffixes).lower()

        if suffix.endswith(".txt"):
            yield from self._parse_text()
        else:
            yield from self._parse_binary()

    def _open_file(self) -> BinaryIO:
        """Open trace file with appropriate decompression."""
        name = self.trace_path.name.lower()
        if name.endswith(".xz"):
            return lzma.open(self.trace_path, "rb")
        elif name.endswith(".gz"):
            return gzip.open(self.trace_path, "rb")
        else:
            return open(self.trace_path, "rb")

    def _parse_binary(self) -> Iterator[ChampSimRecord]:
        """Parse binary ChampSim trace format."""
        self._record_count = 0
        with self._open_file() as f:
            while True:
                if self.max_records and self._record_count >= self.max_records:
                    break
                buf = f.read(BINARY_RECORD_SIZE)
                if len(buf) < BINARY_RECORD_SIZE:
                    break

                record = self._decode_binary_record(buf, self._record_count)
                if record is not None:
                    self._record_count += 1
                    yield record

        logger.info(f"Parsed {self._record_count} records from {self.trace_path.name}")

    def _decode_binary_record(self, buf: bytes, seq_id: int) -> Optional[ChampSimRecord]:
        """Decode a single binary trace record.
        
        ChampSim binary format (64 bytes):
          uint64 ip
          uint8  is_branch
          uint8  branch_taken
          uint8[6] dst_registers (padded)
          uint8[6] src_registers (padded)
          uint64 source_memory
          uint64 destination_memory
          uint8[26] padding
        """
        try:
            ip = struct.unpack_from("<Q", buf, 0)[0]
            is_branch = buf[8] != 0
            branch_taken = buf[9] != 0
            dst_regs = [b for b in buf[10:16] if b != 0]
            src_regs = [b for b in buf[16:22] if b != 0]
            src_mem = struct.unpack_from("<Q", buf, 24)[0]
            dst_mem = struct.unpack_from("<Q", buf, 32)[0]

            return ChampSimRecord(
                ip=ip,
                is_branch=is_branch,
                branch_taken=branch_taken,
                dst_registers=dst_regs,
                src_registers=src_regs,
                source_memory=src_mem,
                destination_memory=dst_mem,
                instruction_id=seq_id,
            )
        except Exception as e:
            logger.warning(f"Failed to decode record {seq_id}: {e}")
            return None

    def _parse_text(self) -> Iterator[ChampSimRecord]:
        """Parse text-format trace (space-separated fields)."""
        self._record_count = 0
        with open(self.trace_path, "r") as f:
            for line in f:
                if self.max_records and self._record_count >= self.max_records:
                    break
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                record = self._decode_text_line(line, self._record_count)
                if record is not None:
                    self._record_count += 1
                    yield record

        logger.info(f"Parsed {self._record_count} text records from {self.trace_path.name}")

    def _decode_text_line(self, line: str, seq_id: int) -> Optional[ChampSimRecord]:
        """Decode a text-format trace line."""
        try:
            parts = line.split()
            ip = int(parts[0], 16) if parts[0].startswith("0x") else int(parts[0])
            is_branch = int(parts[1]) != 0
            branch_taken = int(parts[2]) != 0

            # Registers are variable-length; memory addresses are last two fields
            src_mem = int(parts[-2], 16) if parts[-2].startswith("0x") else int(parts[-2])
            dst_mem = int(parts[-1], 16) if parts[-1].startswith("0x") else int(parts[-1])

            return ChampSimRecord(
                ip=ip,
                is_branch=is_branch,
                branch_taken=branch_taken,
                dst_registers=[],
                src_registers=[],
                source_memory=src_mem,
                destination_memory=dst_mem,
                instruction_id=seq_id,
            )
        except Exception as e:
            logger.warning(f"Failed to parse text line {seq_id}: {e}")
            return None

    @property
    def records_parsed(self) -> int:
        return self._record_count
