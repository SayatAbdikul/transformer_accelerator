"""Machine state for the golden model simulator."""
import numpy as np
from ..isa.opcodes import ABUF_SIZE, WBUF_SIZE, ACCUM_SIZE


class MachineState:
    """Complete state of the accelerator."""

    def __init__(self, dram_data: bytes = b""):
        # SRAM buffers
        self.abuf = bytearray(ABUF_SIZE)
        self.wbuf = bytearray(WBUF_SIZE)
        # ACCUM is INT32: 64KB = 16384 int32 values
        self.accum = np.zeros(ACCUM_SIZE // 4, dtype=np.int32)

        # DRAM - 16MB minimum, initialized from program data section
        dram_size = max(16 * 1024 * 1024, len(dram_data))
        self.dram = bytearray(dram_size)
        if dram_data:
            self.dram[:len(dram_data)] = dram_data

        # Scale registers: 16 FP16 values
        self.scale_regs = np.zeros(16, dtype=np.float16)

        # DRAM address registers: 4 × 56-bit
        self.addr_regs = np.zeros(4, dtype=np.uint64)

        # Tile configuration: (M, N, K) in 16-element units (0-based encoded)
        self.tile_config = None  # Set by CONFIG_TILE

        # Program counter and execution state
        self.pc = 0
        self.halted = False
        self.cycle_count = 0

    def get_buffer(self, buf_id: int) -> bytearray:
        """Get SRAM buffer by ID."""
        if buf_id == 0:
            return self.abuf
        elif buf_id == 1:
            return self.wbuf
        elif buf_id == 2:
            # Return accum as bytes view - caller must handle int32
            raise ValueError("Use get_accum() for ACCUM buffer")
        else:
            from .simulator import IllegalBufferError
            raise IllegalBufferError(buf_id)

    def get_buffer_or_accum(self, buf_id: int):
        """Get buffer - returns bytearray for ABUF/WBUF, numpy array for ACCUM."""
        if buf_id == 0:
            return self.abuf
        elif buf_id == 1:
            return self.wbuf
        elif buf_id == 2:
            return self.accum
        else:
            from .simulator import IllegalBufferError
            raise IllegalBufferError(buf_id)

    def reset(self):
        """Reset all state."""
        self.abuf[:] = b'\x00' * ABUF_SIZE
        self.wbuf[:] = b'\x00' * WBUF_SIZE
        self.accum[:] = 0
        self.scale_regs[:] = 0
        self.addr_regs[:] = 0
        self.tile_config = None
        self.pc = 0
        self.halted = False
        self.cycle_count = 0
