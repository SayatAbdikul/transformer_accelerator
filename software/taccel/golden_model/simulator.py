"""Fetch-decode-execute loop for the golden model simulator."""
import numpy as np
from ..isa.encoding import decode
from ..isa.opcodes import Opcode, BUF_ABUF, BUF_WBUF, BUF_ACCUM
from ..isa.instructions import (
    NopInsn, HaltInsn, SyncInsn, ConfigTileInsn, SetScaleInsn,
    SetAddrLoInsn, SetAddrHiInsn, LoadInsn, StoreInsn, BufCopyInsn,
    MatmulInsn, RequantInsn, ScaleMulInsn, VaddInsn,
    SoftmaxInsn, LayernormInsn, GeluInsn,
)
from .state import MachineState
from . import memory as mem
from .systolic import execute_matmul
from .sfu import execute_layernorm, execute_softmax, execute_gelu
from .dma import execute_load, execute_store, execute_buf_copy
from ..utils.int8_ops import clip_int8, clip_int32


class SimulatorError(Exception):
    pass

class IllegalOpcodeError(SimulatorError):
    def __init__(self, pc, raw_bytes):
        super().__init__(f"Illegal opcode at PC={pc}: {raw_bytes.hex()}")

class IllegalBufferError(SimulatorError):
    def __init__(self, buf_id):
        super().__init__(f"Illegal buffer ID: {buf_id}")

class ConfigError(SimulatorError):
    pass


class Simulator:
    """Golden model simulator with fetch-decode-execute loop."""

    def __init__(self, state: MachineState = None):
        self.state = state or MachineState()

    def load_program(self, program):
        """Load a ProgramBinary into the simulator."""
        self.program = program
        self.state.pc = program.entry_point
        self.state.halted = False
        # Load data section into DRAM at offset 0
        if program.data:
            if len(program.data) > len(self.state.dram):
                self.state.dram = bytearray(len(program.data) + 1024 * 1024)
            self.state.dram[:len(program.data)] = program.data

    def run(self, max_instructions: int = 10_000_000):
        """Run until HALT or max instructions."""
        count = 0
        while not self.state.halted and count < max_instructions:
            self.step()
            count += 1
        return count

    def step(self):
        """Execute one instruction."""
        if self.state.halted:
            return

        pc = self.state.pc
        if pc >= self.program.insn_count:
            raise SimulatorError(f"PC={pc} past end of program ({self.program.insn_count} insns)")

        raw = self.program.get_instruction_bytes(pc)
        try:
            insn = decode(raw)
        except ValueError:
            raise IllegalOpcodeError(pc, raw)

        self._execute(insn)
        self.state.pc += 1

    def _execute(self, insn):
        """Dispatch instruction to handler."""
        op = insn.opcode

        if op == Opcode.NOP:
            self.state.cycle_count += 1
        elif op == Opcode.HALT:
            self.state.halted = True
        elif op == Opcode.SYNC:
            # In golden model, SYNC is a no-op (all ops are synchronous)
            self.state.cycle_count += 1
        elif op == Opcode.CONFIG_TILE:
            self.state.tile_config = (insn.M, insn.N, insn.K)
        elif op == Opcode.SET_SCALE:
            self._exec_set_scale(insn)
        elif op == Opcode.SET_ADDR_LO:
            self._exec_set_addr_lo(insn)
        elif op == Opcode.SET_ADDR_HI:
            self._exec_set_addr_hi(insn)
        elif op == Opcode.LOAD:
            execute_load(self.state, insn)
        elif op == Opcode.STORE:
            execute_store(self.state, insn)
        elif op == Opcode.BUF_COPY:
            execute_buf_copy(self.state, insn)
        elif op == Opcode.MATMUL:
            execute_matmul(self.state, insn)
        elif op == Opcode.REQUANT:
            self._exec_requant(insn)
        elif op == Opcode.SCALE_MUL:
            self._exec_scale_mul(insn)
        elif op == Opcode.VADD:
            self._exec_vadd(insn)
        elif op == Opcode.SOFTMAX:
            execute_softmax(self.state, insn)
        elif op == Opcode.LAYERNORM:
            execute_layernorm(self.state, insn)
        elif op == Opcode.GELU:
            execute_gelu(self.state, insn)
        else:
            raise IllegalOpcodeError(self.state.pc, b'\x00' * 8)

    def _exec_set_scale(self, insn):
        """SET_SCALE: load FP16 into scale register."""
        if insn.src_mode == 0:
            # Immediate FP16 - store as little-endian for numpy
            fp16_bytes = insn.imm16.to_bytes(2, 'little')
            val = np.frombuffer(fp16_bytes, dtype=np.float16)[0]
            self.state.scale_regs[insn.sreg] = val
        else:
            # From buffer
            buf_id = {1: BUF_ABUF, 2: BUF_WBUF, 3: BUF_ACCUM}[insn.src_mode]
            data = mem.read_bytes(self.state, buf_id, insn.imm16, 2)
            val = np.frombuffer(data, dtype=np.float16)[0]
            self.state.scale_regs[insn.sreg] = val

    def _exec_set_addr_lo(self, insn):
        """SET_ADDR_LO: set bits [27:0] of DRAM address register."""
        reg = insn.addr_reg
        self.state.addr_regs[reg] = (int(self.state.addr_regs[reg]) & ~0xFFFFFFF) | insn.imm28

    def _exec_set_addr_hi(self, insn):
        """SET_ADDR_HI: set bits [55:28] of DRAM address register."""
        reg = insn.addr_reg
        self.state.addr_regs[reg] = (int(self.state.addr_regs[reg]) & 0xFFFFFFF) | (insn.imm28 << 28)

    def _exec_requant(self, insn):
        """REQUANT: INT32 → INT8 via scale register.

        dst = clip(round(src1 × S[sreg]), -128, 127)

        Scale register holds FP16; widened to FP32 for the multiply.
        Rounding: round-half-to-even (numpy default).
        INT32 inputs are exact in FP32 for the accumulator range used here
        (max |acc| = 197 tokens × 127² ≈ 3.2M << 2^24 FP32 exact limit).
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        # FP16 scale register widened to FP32 — no extra precision introduced
        scale = np.float32(self.state.scale_regs[insn.sreg])

        src = mem.read_int32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)

        # Multiply in FP32, round-half-to-even, clip to INT8
        result = np.clip(np.round(src.astype(np.float32) * scale), -128, 127).astype(np.int8)

        mem.write_int8_tile(self.state, insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_scale_mul(self, insn):
        """SCALE_MUL: multiply tile by scale.

        When src is ACCUM (INT32): dst = clip(round(src × scale), INT32_MIN, INT32_MAX)
        When src is ABUF (INT8):   dst = clip(round(src × scale), -128, 127)

        Scale register holds FP16; widened to FP32 for the multiply.
        Rounding: round-half-to-even (numpy default).
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        # FP16 scale register widened to FP32 — no extra precision introduced
        scale = np.float32(self.state.scale_regs[insn.sreg])

        if insn.src1_buf == BUF_ACCUM:
            # INT32 path: multiply in FP32, clip to INT32
            src = mem.read_int32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            scaled = np.round(src.astype(np.float32) * scale)
            result = np.clip(scaled, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
            mem.write_int32_tile(self.state, insn.dst_buf, insn.dst_off, result)
        else:
            # INT8 path: multiply in FP32, clip to INT8
            src = mem.read_int8_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            result = np.clip(np.round(src.astype(np.float32) * scale), -128, 127).astype(np.int8)
            mem.write_int8_tile(self.state, insn.dst_buf, insn.dst_off, result)

        self.state.cycle_count += M * N

    def _exec_vadd(self, insn):
        """VADD: elementwise add.

        Dispatch based on src1_buf:
        - ABUF (00): saturating INT8 add
        - ACCUM (10): INT32 add with broadcast (src2 row broadcast when M > 1 row in src2)
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        if insn.src1_buf == BUF_ABUF:
            # Saturating INT8 add
            src1 = mem.read_int8_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            src2 = mem.read_int8_tile(self.state, insn.src2_buf, insn.src2_off, M, N)
            result = np.zeros((M, N), dtype=np.int8)
            for i in range(M):
                for j in range(N):
                    result[i, j] = clip_int8(int(src1[i, j]) + int(src2[i, j]))
            mem.write_int8_tile(self.state, insn.dst_buf, insn.dst_off, result)

        elif insn.src1_buf == BUF_ACCUM:
            # INT32 add with broadcast: src2 is a bias vector (1 row),
            # broadcast across all M rows of the accumulator.
            src1 = mem.read_int32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            src2_row = mem.read_int32_tile(self.state, insn.src2_buf, insn.src2_off, 1, N)
            src2 = np.tile(src2_row, (M, 1))
            result = src1 + src2
            mem.write_int32_tile(self.state, insn.dst_buf, insn.dst_off, result)

        else:
            raise IllegalBufferError(insn.src1_buf)

        self.state.cycle_count += M * N
