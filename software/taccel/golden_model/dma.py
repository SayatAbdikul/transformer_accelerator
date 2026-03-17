"""DMA engine and buffer copy model."""
import numpy as np
from . import memory
from .memory import SRAMAccessError, DRAMAccessError

UNIT = 16  # bytes per addressing unit
CYCLE_PER_UNIT = 1


def execute_load(state, insn):
    """DMA LOAD: DRAM → SRAM buffer."""
    # Resolve DRAM address
    base_addr = int(state.addr_regs[insn.addr_reg])
    dram_byte_addr = base_addr + insn.dram_off * UNIT
    xfer_bytes = insn.xfer_len * UNIT
    sram_byte_offset = insn.sram_off * UNIT

    # Bounds check DRAM
    if dram_byte_addr + xfer_bytes > len(state.dram):
        raise DRAMAccessError(dram_byte_addr + xfer_bytes)

    # Read from DRAM
    dram_data = bytes(state.dram[dram_byte_addr:dram_byte_addr + xfer_bytes])

    # Write to SRAM
    memory.write_bytes(state, insn.buf_id, insn.sram_off, dram_data)

    state.cycle_count += insn.xfer_len * CYCLE_PER_UNIT


def execute_store(state, insn):
    """DMA STORE: SRAM buffer → DRAM."""
    base_addr = int(state.addr_regs[insn.addr_reg])
    dram_byte_addr = base_addr + insn.dram_off * UNIT
    xfer_bytes = insn.xfer_len * UNIT
    sram_byte_offset = insn.sram_off * UNIT

    # Read from SRAM
    sram_data = memory.read_bytes(state, insn.buf_id, insn.sram_off, xfer_bytes)

    # Bounds check DRAM
    if dram_byte_addr + xfer_bytes > len(state.dram):
        # Extend DRAM if needed
        state.dram.extend(b'\x00' * (dram_byte_addr + xfer_bytes - len(state.dram)))

    # Write to DRAM
    state.dram[dram_byte_addr:dram_byte_addr + xfer_bytes] = sram_data

    state.cycle_count += insn.xfer_len * CYCLE_PER_UNIT


def execute_buf_copy(state, insn):
    """BUF_COPY: inter-buffer copy with optional transpose.

    When transpose=0: flat copy of length*16 bytes.
    When transpose=1: reads src as [src_rows*16, cols] and writes [cols, src_rows*16] to dst.
    Shape is self-contained in the instruction.
    """
    total_bytes = insn.length * UNIT

    if not insn.transpose:
        # Simple flat copy
        data = memory.read_bytes(state, insn.src_buf, insn.src_off, total_bytes)
        memory.write_bytes(state, insn.dst_buf, insn.dst_off, data)
    else:
        # Transpose copy
        src_row_count = insn.src_rows * 16
        if src_row_count == 0:
            return
        cols = total_bytes // src_row_count
        if cols == 0:
            return

        # Read source as [src_row_count, cols] INT8
        src_data = memory.read_bytes(state, insn.src_buf, insn.src_off, total_bytes)
        src_array = np.frombuffer(src_data, dtype=np.int8).reshape(src_row_count, cols)

        # Transpose to [cols, src_row_count]
        dst_array = src_array.T.copy()

        # Write transposed data
        memory.write_bytes(state, insn.dst_buf, insn.dst_off, dst_array.tobytes())

    state.cycle_count += insn.length * CYCLE_PER_UNIT
