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
from cocotb.triggers import RisingEdge, Timer


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

    @staticmethod
    def _sig_int(sig, default: int = 0) -> int:
        """Best-effort signal conversion that tolerates unresolved X/Z values."""
        try:
            v = sig.value
            return int(v) if v.is_resolvable else default
        except Exception:
            return default

    # -----------------------------------------------------------------------
    # AXI4 slave coroutine
    # -----------------------------------------------------------------------

    async def axi_slave(self, dut, latency: int = 2) -> None:
        """Coroutine: drive AXI4 read slave (AR/R channels only).

        Handles multi-beat AR/R bursts. Start axi_write_slave() separately
        for AW/W/B support.

        Run as background tasks:
            cocotb.start_soon(dram.axi_slave(dut))
            cocotb.start_soon(dram.axi_write_slave(dut))  # for STORE
        """
        dut.m_axi_ar_ready.value = 1
        dut.m_axi_r_valid.value  = 0
        dut.m_axi_r_data.value   = 0
        dut.m_axi_r_resp.value   = 0
        dut.m_axi_r_last.value   = 0

        # pending: list of {'addr': int, 'ar_len': int, 'beat': int, 'cd': int}
        pending = []

        while True:
            await RisingEdge(dut.clk)

            # Keep read channel quiescent during reset.
            if self._sig_int(dut.rst_n, default=0) == 0:
                pending.clear()
                dut.m_axi_ar_ready.value = 1
                dut.m_axi_r_valid.value  = 0
                dut.m_axi_r_last.value   = 0
                continue

            # Accept new AR transaction
            if self._sig_int(dut.m_axi_ar_valid, 0) and self._sig_int(dut.m_axi_ar_ready, 0):
                aligned  = int(dut.m_axi_ar_addr.value) & ~0xF
                ar_len   = int(dut.m_axi_ar_len.value)
                pending.append({'addr': aligned, 'ar_len': ar_len,
                                'beat': 0, 'cd': latency})

            # Decrement countdowns
            for req in pending:
                if req['cd'] > 0:
                    req['cd'] -= 1

            # Serve one beat for the oldest ready request
            for req in pending:
                if req['cd'] <= 0:
                    beat_addr = req['addr'] + req['beat'] * 16
                    chunk = self.mem[beat_addr:beat_addr + 16]
                    chunk = bytes(chunk).ljust(16, b'\x00')
                    val = int.from_bytes(chunk, 'little')
                    dut.m_axi_r_data.value  = val
                    dut.m_axi_r_valid.value = 1
                    dut.m_axi_r_resp.value  = 0
                    is_last = (req['beat'] >= req['ar_len'])
                    dut.m_axi_r_last.value  = 1 if is_last else 0

                    # Gate ar_ready while waiting for this beat to be accepted,
                    # so a concurrent fetch AR isn't silently dropped.
                    dut.m_axi_ar_ready.value = 0

                    # Wait for master to accept this beat
                    while True:
                        await RisingEdge(dut.clk)
                        if self._sig_int(dut.m_axi_r_ready, 0):
                            break
                    dut.m_axi_r_valid.value  = 0
                    dut.m_axi_r_last.value   = 0
                    dut.m_axi_ar_ready.value = 1  # re-open AR channel

                    if is_last:
                        pending.remove(req)
                    else:
                        req['beat'] += 1
                        req['cd'] = 1   # next beat next cycle
                    break  # one beat per outer loop iteration

    async def axi_write_slave(self, dut) -> None:
        """Coroutine: drive AXI4 write slave (AW/W/B channels).

        Run as a background task alongside axi_slave():
            cocotb.start_soon(dram.axi_write_slave(dut))
        """
        dut.m_axi_aw_ready.value = 1
        dut.m_axi_w_ready.value  = 0
        dut.m_axi_b_valid.value  = 0
        dut.m_axi_b_resp.value   = 0

        while True:
            await RisingEdge(dut.clk)

            # Keep channel outputs in a safe state while reset is asserted.
            if self._sig_int(dut.rst_n, default=0) == 0:
                dut.m_axi_aw_ready.value = 1
                dut.m_axi_w_ready.value  = 0
                dut.m_axi_b_valid.value  = 0
                dut.m_axi_b_resp.value   = 0
                continue

            # Accept AW only on an edge-stable handshake. Do not use delta-only
            # acceptance; that can make the model think AW happened while the DUT
            # (which samples on clock edges) does not.
            aw_v = self._sig_int(dut.m_axi_aw_valid, 0)
            aw_r = self._sig_int(dut.m_axi_aw_ready, 0)
            if aw_v == 0 or aw_r == 0:
                continue

            aw_addr = int(dut.m_axi_aw_addr.value)
            aw_len = int(dut.m_axi_aw_len.value)
            beats_expected = aw_len + 1
            dut._log.info(f"[write_slave] AW: addr=0x{aw_addr:08x} len={aw_len}")

            # Block additional AW while this burst is in flight.
            dut.m_axi_aw_ready.value = 0
            dut.m_axi_w_ready.value = 1

            beat = 0
            while beat < beats_expected:
                await RisingEdge(dut.clk)

                if self._sig_int(dut.rst_n, default=0) == 0:
                    dut.m_axi_w_ready.value = 0
                    dut.m_axi_b_valid.value = 0
                    dut.m_axi_aw_ready.value = 1
                    break

                # Accept W only on an edge-stable handshake.
                w_valid = self._sig_int(dut.m_axi_w_valid, 0)
                w_ready = self._sig_int(dut.m_axi_w_ready, 0)
                if not (w_valid and w_ready):
                    continue

                w_val = int(dut.m_axi_w_data.value)
                w_strb = int(dut.m_axi_w_strb.value)
                w_last = self._sig_int(dut.m_axi_w_last, 0)

                beat_addr = aw_addr + beat * 16
                for b in range(16):
                    if ((w_strb >> b) & 0x1) and (beat_addr + b < self.size):
                        self.mem[beat_addr + b] = (w_val >> (b * 8)) & 0xFF

                expected_last = 1 if beat == beats_expected - 1 else 0
                if w_last != expected_last:
                    dut._log.warning(
                        f"[write_slave] WLAST mismatch beat={beat} w_last={w_last} expected={expected_last}"
                    )

                beat += 1

            # If reset hit during beat collection, restart in idle state.
            if beat < beats_expected:
                continue

            dut.m_axi_w_ready.value = 0

            # Send B response.
            dut.m_axi_b_valid.value = 1
            dut.m_axi_b_resp.value = 0
            while True:
                await RisingEdge(dut.clk)
                if self._sig_int(dut.rst_n, default=0) == 0:
                    dut.m_axi_b_valid.value = 0
                    break
                if self._sig_int(dut.m_axi_b_ready, 0):
                    dut.m_axi_b_valid.value = 0
                    break

            # Re-open AW channel for next transaction.
            dut.m_axi_aw_ready.value = 1
