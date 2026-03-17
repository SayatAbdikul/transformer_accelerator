"""Two-pass assembler: text assembly → ProgramBinary."""
import struct
from dataclasses import dataclass, field
from typing import List, Optional
from .syntax import parse_line
from ..isa.encoding import encode
from ..isa.instructions import Instruction


MAGIC = 0x54414343  # "TACC"
VERSION = 0x0001
HEADER_SIZE = 32


@dataclass
class ProgramBinary:
    """Binary program format with header, instructions, and data."""
    instructions: bytes = b""
    data: bytes = b""
    entry_point: int = 0
    insn_count: int = 0

    def to_bytes(self) -> bytes:
        data_offset = HEADER_SIZE + len(self.instructions)
        header = struct.pack(">IHHIIIIQ",
            MAGIC,
            VERSION,
            0,  # flags
            self.insn_count,
            data_offset,
            len(self.data),
            self.entry_point,
            0,  # reserved
        )
        return header + self.instructions + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> "ProgramBinary":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Data too short for header: {len(data)} bytes")
        magic, version, flags, insn_count, data_offset, data_size, entry_point, _ = \
            struct.unpack(">IHHIIIIQ", data[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic:#010x}, expected {MAGIC:#010x}")
        instructions = data[HEADER_SIZE:data_offset]
        prog_data = data[data_offset:data_offset + data_size]
        return cls(
            instructions=instructions,
            data=prog_data,
            entry_point=entry_point,
            insn_count=insn_count,
        )

    def get_instruction_bytes(self, pc: int) -> bytes:
        """Get the 8 bytes for instruction at given PC."""
        offset = pc * 8
        return self.instructions[offset:offset + 8]


class Assembler:
    """Two-pass assembler."""

    def assemble(self, source: str, data: bytes = b"") -> ProgramBinary:
        lines = source.strip().split('\n')

        # Pass 1: collect labels
        labels = {}
        pc = 0
        for line in lines:
            label, insn = parse_line(line)
            if label is not None:
                labels[label] = pc
            if insn is not None:
                pc += 1

        # Pass 2: emit instructions
        insn_bytes = bytearray()
        for line in lines:
            _, insn = parse_line(line)
            if insn is not None:
                insn_bytes.extend(encode(insn))

        insn_count = len(insn_bytes) // 8
        return ProgramBinary(
            instructions=bytes(insn_bytes),
            data=data,
            entry_point=0,
            insn_count=insn_count,
        )
