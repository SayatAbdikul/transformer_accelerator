"""TACCEL instruction word builder for cocotb tests.

Mirrors software/taccel/isa/encoding.py exactly.  Returns 64-bit ints.
"""

# Bit-position constants (from software/taccel/isa/opcodes.py)
OPCODE_SHIFT      = 59
R_SRC1_BUF_SHIFT  = 57
R_SRC1_OFF_SHIFT  = 41
R_SRC2_BUF_SHIFT  = 39
R_SRC2_OFF_SHIFT  = 23
R_DST_BUF_SHIFT   = 21
R_DST_OFF_SHIFT   = 5
R_SREG_SHIFT      = 1
R_FLAGS_SHIFT     = 0
M_BUF_ID_SHIFT    = 57
M_SRAM_OFF_SHIFT  = 41
M_XFER_LEN_SHIFT  = 25
M_ADDR_REG_SHIFT  = 23
M_DRAM_OFF_SHIFT  = 7
B_SRC_BUF_SHIFT   = 57
B_SRC_OFF_SHIFT   = 41
B_DST_BUF_SHIFT   = 39
B_DST_OFF_SHIFT   = 23
B_LENGTH_SHIFT    = 7
B_SRC_ROWS_SHIFT  = 1
B_TRANSPOSE_SHIFT = 0
A_ADDR_REG_SHIFT  = 57
A_IMM28_SHIFT     = 29
C_M_SHIFT         = 49
C_N_SHIFT         = 39
C_K_SHIFT         = 29
SS_SREG_SHIFT     = 55
SS_SRC_MODE_SHIFT = 53
SS_IMM16_SHIFT    = 37
SYNC_MASK_SHIFT   = 56

# Buffer IDs
BUF_ABUF  = 0b00
BUF_WBUF  = 0b01
BUF_ACCUM = 0b10


def NOP() -> int:
    return 0x00 << OPCODE_SHIFT


def HALT() -> int:
    return 0x01 << OPCODE_SHIFT


def SYNC(mask: int) -> int:
    return (0x02 << OPCODE_SHIFT) | ((mask & 0x7) << SYNC_MASK_SHIFT)


def CONFIG_TILE(M: int, N: int, K: int) -> int:
    assert 1 <= M <= 1024 and 1 <= N <= 1024 and 1 <= K <= 1024
    return ((0x03 << OPCODE_SHIFT) |
            ((M - 1) << C_M_SHIFT) |
            ((N - 1) << C_N_SHIFT) |
            ((K - 1) << C_K_SHIFT))


def SET_SCALE(sreg: int, imm16: int, src_mode: int = 0) -> int:
    return ((0x04 << OPCODE_SHIFT)       |
            ((sreg & 0xF)   << SS_SREG_SHIFT)     |
            ((src_mode & 3) << SS_SRC_MODE_SHIFT) |
            ((imm16 & 0xFFFF) << SS_IMM16_SHIFT))


def SET_ADDR_LO(reg: int, imm28: int) -> int:
    return ((0x05 << OPCODE_SHIFT) |
            ((reg & 3) << A_ADDR_REG_SHIFT) |
            ((imm28 & 0xFFFFFFF) << A_IMM28_SHIFT))


def SET_ADDR_HI(reg: int, imm28: int) -> int:
    return ((0x06 << OPCODE_SHIFT) |
            ((reg & 3) << A_ADDR_REG_SHIFT) |
            ((imm28 & 0xFFFFFFF) << A_IMM28_SHIFT))


def LOAD(buf_id: int, sram_off: int, xfer_len: int,
         addr_reg: int, dram_off: int) -> int:
    return ((0x07 << OPCODE_SHIFT)        |
            ((buf_id & 3)   << M_BUF_ID_SHIFT)   |
            ((sram_off)     << M_SRAM_OFF_SHIFT)  |
            ((xfer_len)     << M_XFER_LEN_SHIFT)  |
            ((addr_reg & 3) << M_ADDR_REG_SHIFT)  |
            ((dram_off)     << M_DRAM_OFF_SHIFT))


def STORE(buf_id: int, sram_off: int, xfer_len: int,
          addr_reg: int, dram_off: int) -> int:
    return ((0x08 << OPCODE_SHIFT)        |
            ((buf_id & 3)   << M_BUF_ID_SHIFT)   |
            ((sram_off)     << M_SRAM_OFF_SHIFT)  |
            ((xfer_len)     << M_XFER_LEN_SHIFT)  |
            ((addr_reg & 3) << M_ADDR_REG_SHIFT)  |
            ((dram_off)     << M_DRAM_OFF_SHIFT))


def MATMUL(src1_buf: int, src1_off: int, src2_buf: int, src2_off: int,
           dst_buf: int, dst_off: int, sreg: int, flags: int = 0) -> int:
    return ((0x0A << OPCODE_SHIFT)          |
            ((src1_buf & 3) << R_SRC1_BUF_SHIFT) |
            ((src1_off)     << R_SRC1_OFF_SHIFT)  |
            ((src2_buf & 3) << R_SRC2_BUF_SHIFT)  |
            ((src2_off)     << R_SRC2_OFF_SHIFT)  |
            ((dst_buf & 3)  << R_DST_BUF_SHIFT)   |
            ((dst_off)      << R_DST_OFF_SHIFT)    |
            ((sreg & 0xF)   << R_SREG_SHIFT)       |
            ((flags & 1)    << R_FLAGS_SHIFT))


def ILLEGAL_OP() -> int:
    """Reserved opcode 0x11 for fault testing."""
    return 0x11 << OPCODE_SHIFT
