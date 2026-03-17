"""ISA opcode definitions, instruction formats, and field constants."""
from enum import IntEnum


class Opcode(IntEnum):
    NOP = 0x00
    HALT = 0x01
    SYNC = 0x02
    CONFIG_TILE = 0x03
    SET_SCALE = 0x04
    SET_ADDR_LO = 0x05
    SET_ADDR_HI = 0x06
    LOAD = 0x07
    STORE = 0x08
    BUF_COPY = 0x09
    MATMUL = 0x0A
    REQUANT = 0x0B
    SCALE_MUL = 0x0C
    VADD = 0x0D
    SOFTMAX = 0x0E
    LAYERNORM = 0x0F
    GELU = 0x10


class InsnFormat(IntEnum):
    R_TYPE = 0
    M_TYPE = 1
    B_TYPE = 2
    A_TYPE = 3
    C_TYPE = 4
    S_TYPE = 5


OPCODE_FORMAT = {
    Opcode.NOP: InsnFormat.S_TYPE,
    Opcode.HALT: InsnFormat.S_TYPE,
    Opcode.SYNC: InsnFormat.S_TYPE,
    Opcode.CONFIG_TILE: InsnFormat.C_TYPE,
    Opcode.SET_SCALE: InsnFormat.S_TYPE,
    Opcode.SET_ADDR_LO: InsnFormat.A_TYPE,
    Opcode.SET_ADDR_HI: InsnFormat.A_TYPE,
    Opcode.LOAD: InsnFormat.M_TYPE,
    Opcode.STORE: InsnFormat.M_TYPE,
    Opcode.BUF_COPY: InsnFormat.B_TYPE,
    Opcode.MATMUL: InsnFormat.R_TYPE,
    Opcode.REQUANT: InsnFormat.R_TYPE,
    Opcode.SCALE_MUL: InsnFormat.R_TYPE,
    Opcode.VADD: InsnFormat.R_TYPE,
    Opcode.SOFTMAX: InsnFormat.R_TYPE,
    Opcode.LAYERNORM: InsnFormat.R_TYPE,
    Opcode.GELU: InsnFormat.R_TYPE,
}

# Buffer IDs
BUF_ABUF = 0b00
BUF_WBUF = 0b01
BUF_ACCUM = 0b10
BUF_RESERVED = 0b11

BUFFER_NAMES = {BUF_ABUF: "ABUF", BUF_WBUF: "WBUF", BUF_ACCUM: "ACCUM"}

# Per-buffer max offset (in 16-byte units)
ABUF_MAX_OFF = 8191    # 128KB / 16 = 8192 slots, 0-indexed
WBUF_MAX_OFF = 16383   # 256KB / 16
ACCUM_MAX_OFF = 4095   # 64KB / 16

BUFFER_MAX_OFF = {
    BUF_ABUF: ABUF_MAX_OFF,
    BUF_WBUF: WBUF_MAX_OFF,
    BUF_ACCUM: ACCUM_MAX_OFF,
}

# Buffer sizes in bytes
ABUF_SIZE = 128 * 1024
WBUF_SIZE = 256 * 1024
ACCUM_SIZE = 64 * 1024

# Systolic array dimensions
SYSTOLIC_DIM = 16

# --- Bit field positions (from MSB, bit 63 is MSB) ---
# All formats: opcode at [63:59]
OPCODE_SHIFT = 59
OPCODE_MASK = 0x1F

# R-type fields
R_SRC1_BUF_SHIFT = 57
R_SRC1_OFF_SHIFT = 41
R_SRC2_BUF_SHIFT = 39
R_SRC2_OFF_SHIFT = 23
R_DST_BUF_SHIFT = 21
R_DST_OFF_SHIFT = 5
R_SREG_SHIFT = 1
R_FLAGS_SHIFT = 0

# M-type fields
M_BUF_ID_SHIFT = 57
M_SRAM_OFF_SHIFT = 41
M_XFER_LEN_SHIFT = 25
M_ADDR_REG_SHIFT = 23
M_DRAM_OFF_SHIFT = 7
M_STRIDE_LOG2_SHIFT = 3
M_FLAGS_SHIFT = 0

# B-type fields
B_SRC_BUF_SHIFT = 57
B_SRC_OFF_SHIFT = 41
B_DST_BUF_SHIFT = 39
B_DST_OFF_SHIFT = 23
B_LENGTH_SHIFT = 7
B_SRC_ROWS_SHIFT = 1
B_TRANSPOSE_SHIFT = 0

# A-type fields
A_ADDR_REG_SHIFT = 57
A_IMM28_SHIFT = 29

# C-type fields
C_M_SHIFT = 49
C_N_SHIFT = 39
C_K_SHIFT = 29

# S-type SET_SCALE fields
SS_SREG_SHIFT = 55
SS_SRC_MODE_SHIFT = 53
SS_IMM16_SHIFT = 37

# S-type SYNC fields
SYNC_RESOURCE_MASK_SHIFT = 56

# Field widths / masks
MASK_2BIT = 0x3
MASK_3BIT = 0x7
MASK_4BIT = 0xF
MASK_5BIT = 0x1F
MASK_6BIT = 0x3F
MASK_10BIT = 0x3FF
MASK_16BIT = 0xFFFF
MASK_28BIT = 0xFFFFFFF
