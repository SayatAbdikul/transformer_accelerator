"""cocotb tests for the Stage D SFU engine."""

import math
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

from utils.dram_model import DramModel
from utils.insn_builder import (
    HALT, SYNC, CONFIG_TILE, SET_SCALE, SET_ADDR_LO, SET_ADDR_HI,
    LOAD, STORE, SOFTMAX, LAYERNORM, GELU,
    BUF_ABUF, BUF_WBUF, BUF_ACCUM,
)


async def _setup(dut, insns, dram_writes=None, dram=None):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    if dram is None:
        dram = DramModel()
    dram.write_program(insns)
    if dram_writes:
        for addr, data in dram_writes.items():
            dram.write_bytes(addr, bytes(data))
    cocotb.start_soon(dram.axi_slave(dut))
    cocotb.start_soon(dram.axi_write_slave(dut))

    dut.rst_n.value = 0
    dut.start.value = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    return dram


async def _wait_halt(dut, max_cycles=800_000):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) or int(dut.fault.value):
            return
    raise TimeoutError("DUT did not halt")


def _set_addr(reg: int, addr: int):
    return [SET_ADDR_LO(reg, addr & 0x0FFFFFFF), SET_ADDR_HI(reg, (addr >> 28) & 0x0FFFFFFF)]


def _fp16_to_float(bits: int) -> float:
    bits = int(bits)
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    sign_v = -1.0 if sign else 1.0
    if exp == 0 and frac == 0:
        return 0.0
    if exp == 0:
        return sign_v * (frac / 1024.0) * math.ldexp(1.0, -14)
    if exp == 31:
        return sign_v * 65504.0
    return sign_v * (1.0 + frac / 1024.0) * math.ldexp(1.0, exp - 15)


def _round_half_even(x: float) -> int:
    floor_i = math.floor(x)
    frac = x - floor_i
    if frac > 0.5:
        return floor_i + 1
    if frac < 0.5:
        return floor_i
    return floor_i + 1 if (floor_i & 1) else floor_i


def _quantize(x: float, out_scale: float) -> int:
    if out_scale == 0.0:
        return 0
    q = _round_half_even(x / out_scale)
    return max(-128, min(127, q))


def _softmax_ref(inp_i32: np.ndarray, in_scale: float, out_scale: float) -> np.ndarray:
    x = inp_i32.astype(np.float64) * in_scale
    x_shifted = x - x.max(axis=-1, keepdims=True)
    probs = np.exp(x_shifted) / np.exp(x_shifted).sum(axis=-1, keepdims=True)
    out = np.zeros_like(inp_i32, dtype=np.int8)
    for idx, value in np.ndenumerate(probs):
        out[idx] = _quantize(float(value), out_scale)
    return out


def _layernorm_ref(inp_i8: np.ndarray, gamma_bits, beta_bits, in_scale: float, out_scale: float) -> np.ndarray:
    gamma = np.array([_fp16_to_float(v) for v in gamma_bits], dtype=np.float64)
    beta = np.array([_fp16_to_float(v) for v in beta_bits], dtype=np.float64)
    x = inp_i8.astype(np.float64) * in_scale
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = ((x - mean) / np.sqrt(var + 1.0e-6)) * gamma + beta
    out = np.zeros_like(inp_i8, dtype=np.int8)
    for idx, value in np.ndenumerate(y):
        out[idx] = _quantize(float(value), out_scale)
    return out


def _gelu_ref_i32(inp_i32: np.ndarray, in_scale: float, out_scale: float) -> np.ndarray:
    x = inp_i32.astype(np.float64) * in_scale
    erf_vals = np.vectorize(lambda v: math.erf(v / math.sqrt(2.0)))(x)
    y = x * 0.5 * (1.0 + erf_vals)
    out = np.zeros_like(inp_i32, dtype=np.int8)
    for idx, value in np.ndenumerate(y):
        out[idx] = _quantize(float(value), out_scale)
    return out


@cocotb.test()
async def test_softmax_accum_large_row(dut):
    m_tiles, n_tiles = 1, 13
    M, N = 16, 208
    src_addr = 0x30000
    dst_addr = 0x34000
    dst_off = 256
    in_scale = _fp16_to_float(0x3400)
    out_scale = _fp16_to_float(0x3400)

    inp = np.zeros((M, N), dtype=np.int32)
    inp[:, -1] = 32
    expected = _softmax_ref(inp, in_scale, out_scale).astype(np.int8).tobytes()

    prog = [
        *_set_addr(0, src_addr),
        LOAD(BUF_ACCUM, 0, (M * N * 4) // 16, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(m_tiles, n_tiles, 1),
        SET_SCALE(2, 0x3400),
        SET_SCALE(3, 0x3400),
        SOFTMAX(BUF_ACCUM, 0, BUF_ABUF, dst_off, 2),
        SYNC(0b100),
        *_set_addr(1, dst_addr),
        STORE(BUF_ABUF, dst_off, (M * N) // 16, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_addr: inp.astype("<i4").tobytes()})
    await _wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    assert got == expected


@cocotb.test()
async def test_layernorm_identity(dut):
    m_tiles, n_tiles = 1, 12
    M, N = 16, 192
    src_addr = 0x38000
    gb_addr = 0x39000
    dst_addr = 0x3A000
    dst_off = 384
    in_scale = _fp16_to_float(0x3400)
    out_scale = _fp16_to_float(0x3400)

    inp = np.zeros((M, N), dtype=np.int8)
    for r in range(M):
        for c in range(N):
            inp[r, c] = np.int8(((r * 17 + c * 5) % 41) - 20)

    gamma = np.full((N,), 0x3C00, dtype=np.uint16)
    beta = np.zeros((N,), dtype=np.uint16)
    gb = gamma.astype("<u2").tobytes() + beta.astype("<u2").tobytes()
    expected = _layernorm_ref(inp, gamma, beta, in_scale, out_scale).astype(np.int8).tobytes()

    prog = [
        *_set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, (M * N) // 16, 0, 0),
        SYNC(0b001),
        *_set_addr(1, gb_addr),
        LOAD(BUF_WBUF, 0, len(gb) // 16, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(m_tiles, n_tiles, 1),
        SET_SCALE(3, 0x3400),
        SET_SCALE(4, 0x3400),
        LAYERNORM(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ABUF, dst_off, 3),
        SYNC(0b100),
        *_set_addr(2, dst_addr),
        STORE(BUF_ABUF, dst_off, (M * N) // 16, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_addr: inp.tobytes(), gb_addr: gb})
    await _wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    assert got == expected


@cocotb.test()
async def test_gelu_accum_roundtrip(dut):
    M, N = 16, 16
    src_addr = 0x3C000
    dst_addr = 0x3D000
    dst_off = 640
    in_scale = _fp16_to_float(0x3400)
    out_scale = _fp16_to_float(0x3400)

    inp = np.zeros((M, N), dtype=np.int32)
    for r in range(M):
        for c in range(N):
            inp[r, c] = (c - 8) * 2 + r
    expected = _gelu_ref_i32(inp, in_scale, out_scale).astype(np.int8).tobytes()

    prog = [
        *_set_addr(0, src_addr),
        LOAD(BUF_ACCUM, 0, (M * N * 4) // 16, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        SET_SCALE(8, 0x3400),
        SET_SCALE(9, 0x3400),
        GELU(BUF_ACCUM, 0, BUF_ABUF, dst_off, 8),
        SYNC(0b100),
        *_set_addr(1, dst_addr),
        STORE(BUF_ABUF, dst_off, (M * N) // 16, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_addr: inp.astype("<i4").tobytes()})
    await _wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    assert got == expected
