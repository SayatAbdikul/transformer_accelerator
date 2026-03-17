"""Bit-accurate INT8 16x16 systolic array matmul model."""
import numpy as np
from . import memory
from ..isa.opcodes import BUF_ACCUM

TILE = 16
CYCLE_COST = 16  # cycles per 16x16 tile


def execute_matmul(state, insn):
    """Execute MATMUL instruction.

    Performs tiled matmul using CONFIG_TILE dimensions.
    src1 = activations (INT8), src2 = weights (INT8), dst = ACCUM (INT32).
    """
    from .simulator import ConfigError

    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    # Tile config: encoded as 0-based, so M_tiles = M+1, etc.
    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    k_tiles = state.tile_config[2] + 1

    M = m_tiles * TILE
    N = n_tiles * TILE
    K = k_tiles * TILE

    accumulate = bool(insn.flags & 1)

    # Read source tiles
    src1 = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, K)
    src2 = memory.read_int8_tile(state, insn.src2_buf, insn.src2_off, K, N)

    # Read existing accumulator if accumulating
    if accumulate:
        dst = memory.read_int32_tile(state, insn.dst_buf, insn.dst_off, M, N)
    else:
        dst = np.zeros((M, N), dtype=np.int32)

    # Vectorized matmul: same INT8→INT32 arithmetic, runs in NumPy C backend
    dst += np.matmul(src1.astype(np.int32), src2.astype(np.int32))

    # Write result
    memory.write_int32_tile(state, insn.dst_buf, insn.dst_off, dst)
    state.cycle_count += m_tiles * n_tiles * k_tiles * CYCLE_COST
