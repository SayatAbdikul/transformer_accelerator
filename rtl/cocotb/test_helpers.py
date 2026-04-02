"""cocotb helper-engine tests for Phase C."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from utils.dram_model import DramModel
from utils.insn_builder import (
    HALT, SYNC, CONFIG_TILE, SET_SCALE, SET_ADDR_LO, SET_ADDR_HI,
    LOAD, STORE, BUF_COPY, MATMUL, REQUANT, VADD,
    BUF_ABUF, BUF_WBUF, BUF_ACCUM,
)


async def _setup(dut, insns, dram_writes=None):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
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


async def _wait_halt(dut, max_cycles=200_000):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) or int(dut.fault.value):
            return
    raise TimeoutError("DUT did not halt")


def _set_addr(reg: int, addr: int):
    return [SET_ADDR_LO(reg, addr & 0x0FFFFFFF), SET_ADDR_HI(reg, (addr >> 28) & 0x0FFFFFFF)]


def _sat_add(a: int, b: int) -> int:
    s = a + b
    if s > 127:
        return 127
    if s < -128:
        return -128
    return s


def _fp16_mul_round_even(src: int, fp16: int) -> int:
    sign = (fp16 >> 15) & 1
    exp = (fp16 >> 10) & 0x1F
    frac = fp16 & 0x3FF
    if exp == 0 and frac == 0:
        return 0
    if exp == 0:
        mant = frac
        shift = -24
    else:
        mant = 1024 + frac
        shift = exp - 25
    prod = src * mant
    if sign:
        prod = -prod
    if shift >= 0:
        return prod << shift
    rshift = -shift
    abs_prod = -prod if prod < 0 else prod
    q = abs_prod >> rshift
    rem = abs_prod & ((1 << rshift) - 1)
    half = 1 << (rshift - 1)
    if rem > half or (rem == half and (q & 1)):
        q += 1
    return -q if prod < 0 else q


def _requant(src: int, scale: int) -> int:
    scaled = _fp16_mul_round_even(src, scale)
    if scaled > 127:
        return 127
    if scaled < -128:
        return -128
    return scaled


def _pack_i32_le(vals):
    out = bytearray()
    for v in vals:
        out.extend(int(v & 0xFFFFFFFF).to_bytes(4, "little", signed=False))
    return bytes(out)


@cocotb.test()
async def test_buf_copy_flat_roundtrip(dut):
    src_addr = 0x10000
    dst_addr = 0x12000
    src = bytes((0x20 + 7 * i) & 0xFF for i in range(48))
    prog = [
        *_set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 3, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 0, BUF_WBUF, 8, 3, 0, 0),
        *_set_addr(1, dst_addr),
        STORE(BUF_WBUF, 8, 3, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_addr: src})
    await _wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(src)]) == src


@cocotb.test()
async def test_buf_copy_overlap_roundtrip(dut):
    src_addr = 0x14000
    dst_addr = 0x15000
    src = bytearray((0x51 + 11 * i) & 0xFF for i in range(96))
    expected = bytearray(src)
    expected[16:64] = expected[32:80]
    prog = [
        *_set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 6, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 2, BUF_ABUF, 1, 3, 0, 0),
        *_set_addr(1, dst_addr),
        STORE(BUF_ABUF, 0, 6, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_addr: bytes(src)})
    await _wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_buf_copy_transpose_unaligned_roundtrip(dut):
    src_addr = 0x18000
    dst_addr = 0x19000
    rows, cols = 16, 18
    src = bytearray(rows * cols)
    expected = bytearray(cols * rows)
    for r in range(rows):
        for c in range(cols):
            src[r * cols + c] = (r * 19 + c * 7 + 3) & 0xFF
            expected[c * rows + r] = src[r * cols + c]
    prog = [
        *_set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 18, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 0, BUF_WBUF, 0, 18, 1, 1),
        *_set_addr(1, dst_addr),
        STORE(BUF_WBUF, 0, 18, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_addr: bytes(src)})
    await _wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_vadd_int8_roundtrip(dut):
    src_a_addr = 0x1A000
    src_b_addr = 0x1B000
    dst_addr = 0x1C000
    src_a = bytearray(256)
    src_b = bytearray(256)
    expected = bytearray(256)
    for i in range(256):
        a = 120 if i % 5 == 0 else -120 if i % 7 == 0 else (i % 17) - 8
        b = 30 if i % 5 == 0 else -30 if i % 7 == 0 else (i % 11) - 5
        src_a[i] = a & 0xFF
        src_b[i] = b & 0xFF
        expected[i] = _sat_add(a, b) & 0xFF
    prog = [
        *_set_addr(0, src_a_addr),
        LOAD(BUF_ABUF, 0, 16, 0, 0),
        SYNC(0b001),
        *_set_addr(1, src_b_addr),
        LOAD(BUF_WBUF, 0, 16, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        VADD(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ABUF, 32, 0),
        *_set_addr(2, dst_addr),
        STORE(BUF_ABUF, 32, 16, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_a_addr: bytes(src_a), src_b_addr: bytes(src_b)})
    await _wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_requant_roundtrip(dut):
    src_addr = 0x1D000
    dst_addr = 0x1E000
    scale = 0x3800
    pattern = [1, 3, 5, -1, -3, -5, 255, 257, -255, -257, 300, -300, 0, 2, -2, 7]
    src = []
    expected = bytearray(256)
    for r in range(16):
        for c in range(16):
            v = pattern[c] + r
            src.append(v)
            expected[r * 16 + c] = _requant(v, scale) & 0xFF
    prog = [
        *_set_addr(0, src_addr),
        LOAD(BUF_ACCUM, 0, 64, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        SET_SCALE(0, scale),
        REQUANT(BUF_ACCUM, 0, BUF_ABUF, 64, 0),
        *_set_addr(1, dst_addr),
        STORE(BUF_ABUF, 64, 16, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_addr: _pack_i32_le(src)})
    await _wait_halt(dut, max_cycles=500_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_matmul_then_requant_roundtrip(dut):
    src_a_addr = 0x20000
    src_b_addr = 0x21000
    dst_addr = 0x22000
    a = [[(i * 5 + j) - 20 for j in range(16)] for i in range(16)]
    eye = [[1 if i == j else 0 for j in range(16)] for i in range(16)]

    # Current systolic contract consumes ABUF tiles transposed per 16x16 tile.
    a_layout = bytearray()
    for r in range(16):
        for c in range(16):
            a_layout.append(a[c][r] & 0xFF)
    b_layout = bytearray()
    for r in range(16):
        for c in range(16):
            b_layout.append(eye[r][c] & 0xFF)

    expected = bytearray()
    for r in range(16):
        for c in range(16):
            expected.append(a[r][c] & 0xFF)

    prog = [
        *_set_addr(0, src_a_addr),
        LOAD(BUF_ABUF, 128, 16, 0, 0),
        SYNC(0b001),
        *_set_addr(1, src_b_addr),
        LOAD(BUF_WBUF, 0, 16, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        MATMUL(BUF_ABUF, 128, BUF_WBUF, 0, BUF_ACCUM, 0, 0),
        SYNC(0b010),
        SET_SCALE(0, 0x3C00),
        REQUANT(BUF_ACCUM, 0, BUF_ABUF, 256, 0),
        *_set_addr(2, dst_addr),
        STORE(BUF_ABUF, 256, 16, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {src_a_addr: bytes(a_layout), src_b_addr: bytes(b_layout)})
    await _wait_halt(dut, max_cycles=600_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)
