"""Special Function Units: LayerNorm, Softmax, GELU (FP32 internal, INT8 I/O).

Precision spec (matches RTL target):
  - Scale registers store FP16 values, widened to FP32 for arithmetic.
  - All SFU internal operations use FP32 (dequant, reduction, exp/erf, requant).
  - Rounding convention: round-half-to-even (numpy default), then clip to INT8.
  - RTL implements erf to FP32 accuracy (polynomial or LUT approximation is sufficient;
    INT8 inputs limit meaningful precision to ~3 significant digits regardless).
"""
import numpy as np
from . import memory
from ..isa.opcodes import BUF_ACCUM
from ..utils.int8_ops import clip_int8

CYCLE_PER_ELEMENT = 2


def _get_dual_scales(state, sreg: int):
    """Return (in_scale, out_scale) as FP32 from consecutive scale registers.

    Scale registers hold FP16. Widening to FP32 preserves the exact FP16 value
    without adding precision — this is the RTL behaviour (FP16 reg → FP32 datapath).
    """
    from .simulator import ConfigError
    if sreg >= 15:
        raise ConfigError("SFU sreg+1 out of range")
    in_scale  = np.float32(state.scale_regs[sreg])
    out_scale = np.float32(state.scale_regs[sreg + 1])
    return in_scale, out_scale


def execute_layernorm(state, insn):
    """LayerNorm: dequant INT8 → FP32, normalize, requant → INT8.

    src1 = input activations, src2 = gamma/beta (FP16 packed), dst = output.
    All arithmetic in FP32; gamma/beta widened from FP16.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)

    # Read gamma, beta — stored as FP16, widen to FP32
    gb_bytes = memory.read_bytes(state, insn.src2_buf, insn.src2_off, N * 4)
    gamma = np.frombuffer(gb_bytes[:N * 2], dtype=np.float16).astype(np.float32)
    beta  = np.frombuffer(gb_bytes[N * 2:], dtype=np.float16).astype(np.float32)

    # Dequantize: INT8 × FP32(in_scale) → FP32
    x = inp.astype(np.float32) * in_scale

    # Normalize in FP32
    eps = np.float32(1e-6)
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1,  keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    x_out = x_norm * gamma + beta   # FP32 affine

    # Requantize: round-half-to-even, clip to INT8
    if out_scale == np.float32(0):
        result = np.zeros_like(inp)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax(state, insn):
    """Softmax: dequant INT8 → FP32, softmax along last dim, requant → INT8.

    Numerically stable: subtract row-max before exp.
    All arithmetic in FP32.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)

    # Dequantize: INT8 × FP32(in_scale) → FP32
    x = inp.astype(np.float32) * in_scale

    # Numerically stable softmax in FP32
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted).astype(np.float32)
    x_out = exp_x / exp_x.sum(axis=-1, keepdims=True)

    # Requantize: round-half-to-even, clip to INT8
    if out_scale == np.float32(0):
        result = np.zeros_like(inp)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_gelu(state, insn):
    """GELU: dequant INT8 → FP32, GELU activation, requant → INT8.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    All arithmetic in FP32. RTL implements erf to FP32 accuracy
    (polynomial or piecewise-linear approximation sufficient for INT8 I/O).
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)

    # Dequantize: INT8 × FP32(in_scale) → FP32
    x = inp.astype(np.float32) * in_scale

    # GELU in FP32
    from scipy.special import erf
    sqrt2 = np.float32(np.sqrt(np.float32(2.0)))
    x_out = x * np.float32(0.5) * (np.float32(1.0) + erf(x / sqrt2).astype(np.float32))

    # Requantize: round-half-to-even, clip to INT8
    if out_scale == np.float32(0):
        result = np.zeros_like(inp)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT
