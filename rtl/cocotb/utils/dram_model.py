"""Behavioral DRAM model for cocotb testbenches.

Stores bytes in a Python bytearray.  Provides helpers to:
  - Write big-endian TACCEL instructions at a given PC index
  - Write raw bytes at a byte address
  - Drive an AXI4 read slave in a coroutine

TACCEL instruction byte layout in DRAM:
  DRAM[pc*8 + 0] = instruction MSByte (opcode, bits [63:56])
  DRAM[pc*8 + 7] = instruction LSByte (bits  [7:0])

AXI4 byte ordering:
  r_data[7:0]   = DRAM[aligned_addr + 0]
  r_data[15:8]  = DRAM[aligned_addr + 1]
  ...
  r_data[127:120] = DRAM[aligned_addr + 15]

The fetch_unit's bswap64() converts from AXI LE byte order back to the
big-endian instruction word.
"""

import struct
import cocotb
from cocotb.triggers import RisingEdge


class DramModel:
    """16 MB (default) byte-addressable DRAM model."""

    def __init__(self, size: int = 16 * 1024 * 1024):
        self.mem = bytearray(size)
        self.size = size

    # -----------------------------------------------------------------------
    # Write helpers
    # -----------------------------------------------------------------------

    def write_insn(self, pc_idx: int, word: int) -> None:
        """Write a 64-bit big-endian instruction at instruction index pc_idx."""
        base = pc_idx * 8
        assert base + 7 < self.size, f"PC {pc_idx} out of DRAM bounds"
        self.mem[base:base + 8] = struct.pack(">Q", word & 0xFFFFFFFFFFFFFFFF)

    def write_program(self, insns: list) -> None:
        """Write a list of 64-bit instruction words starting at PC=0."""
        for i, w in enumerate(insns):
            self.write_insn(i, w)

    def write_bytes(self, addr: int, data: bytes) -> None:
        """Write raw bytes at a byte address."""
        assert addr + len(data) <= self.size
        self.mem[addr:addr + len(data)] = data

    def read_bytes(self, addr: int, length: int) -> bytes:
        return bytes(self.mem[addr:addr + length])

    # -----------------------------------------------------------------------
    # AXI4 slave coroutine
    # -----------------------------------------------------------------------

    async def axi_slave(self, dut, latency: int = 2) -> None:
        """Coroutine: drive AXI4 read slave signals continuously.

        Connects to DUT's m_axi_* ports.  Run as a background task:
            cocotb.start_soon(dram.axi_slave(dut))
        """
        dut.m_axi_ar_ready.value = 1
        dut.m_axi_r_valid.value  = 0
        dut.m_axi_r_data.value   = 0
        dut.m_axi_r_resp.value   = 0
        dut.m_axi_r_last.value   = 0

        pending = []  # list of (addr, countdown)

        while True:
            await RisingEdge(dut.clk)

            # Accept new AR transaction
            if dut.m_axi_ar_valid.value and dut.m_axi_ar_ready.value:
                aligned = int(dut.m_axi_ar_addr.value) & ~0xF
                pending.append([aligned, latency])

            # Decrement pending counters
            for req in pending:
                req[1] -= 1

            # Serve the oldest ready request
            served = []
            for req in pending:
                if req[1] <= 0:
                    aligned_addr = req[0]
                    # Build 128-bit response: 16 bytes, little-endian in r_data
                    chunk = self.mem[aligned_addr:aligned_addr + 16]
                    chunk = chunk.ljust(16, b'\x00')  # pad if at DRAM boundary
                    val = int.from_bytes(chunk, 'little')
                    dut.m_axi_r_data.value = val
                    dut.m_axi_r_valid.value = 1
                    dut.m_axi_r_resp.value  = 0   # OKAY
                    dut.m_axi_r_last.value  = 1
                    served.append(req)
                    # Wait for master to accept
                    while True:
                        await RisingEdge(dut.clk)
                        if dut.m_axi_r_ready.value:
                            break
                    dut.m_axi_r_valid.value = 0
                    dut.m_axi_r_last.value  = 0
                    break  # serve one per iteration

            for req in served:
                pending.remove(req)
