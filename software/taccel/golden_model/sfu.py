"""Special Function Units: LayerNorm, Softmax, GELU (FP32 internal, INT8 I/O)."""
import numpy as np
from . import memory
from ..isa.opcodes import BUF_ACCUM
from ..utils.int8_ops import clip_int8

CYCLE_PER_ELEMENT = 2


def _get_dual_scales(state, sreg: int):
    """Get input and output scales from consecutive scale registers.

    SFU dual-scale convention: sreg = input dequant scale, sreg+1 = output requant scale.
    """
    from .simulator import ConfigError
    if sreg >= 15:
        raise ConfigError("SFU sreg+1 out of range")
    in_scale = float(state.scale_regs[sreg])
    out_scale = float(state.scale_regs[sreg + 1])
    return in_scale, out_scale


def execute_layernorm(state, insn):
    """LayerNorm: dequant INT8 → FP32, normalize, requant → INT8.

    src1 = input activations, src2 = gamma/beta weights (FP16 packed), dst = output.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    # Read input INT8
    inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)

    # Read gamma and beta from src2 (stored as FP16, packed: gamma[N] then beta[N])
    gamma_beta_bytes = memory.read_bytes(state, insn.src2_buf, insn.src2_off, N * 4)
    gamma = np.frombuffer(gamma_beta_bytes[:N * 2], dtype=np.float16).astype(np.float32)
    beta = np.frombuffer(gamma_beta_bytes[N * 2:N * 4], dtype=np.float16).astype(np.float32)

    # Dequantize to FP32
    x = inp.astype(np.float32) * in_scale

    # LayerNorm: normalize along last dimension
    eps = 1e-6
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    x_out = x_norm * gamma + beta

    # Requantize to INT8
    if out_scale == 0:
        result = np.zeros_like(inp)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax(state, insn):
    """Softmax: dequant INT8 → FP32, softmax along last dim, requant → INT8."""
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)
    x = inp.astype(np.float32) * in_scale

    # Numerically stable softmax
    x_max = x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    x_out = exp_x / exp_x.sum(axis=-1, keepdims=True)

    # Requantize
    if out_scale == 0:
        result = np.zeros_like(inp)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_gelu(state, insn):
    """GELU: dequant INT8 → FP32, GELU activation, requant → INT8.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
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
    x = inp.astype(np.float32) * in_scale

    # GELU
    from scipy.special import erf
    x_out = x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    # Requantize
    if out_scale == 0:
        result = np.zeros_like(inp)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT
