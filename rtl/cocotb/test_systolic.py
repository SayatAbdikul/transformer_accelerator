"""cocotb Phase 3 systolic tests.

These tests validate MATMUL dispatch, SYNC behavior, and ACCUM writeback.
Current controller consumes ABUF in transposed layout; tests prepare ABUF rows
accordingly to validate mathematical A @ B behavior.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from utils.insn_builder import (
    HALT, SYNC, CONFIG_TILE, MATMUL,
    SET_ADDR_LO, SET_ADDR_HI, LOAD,
    BUF_ABUF, BUF_WBUF, BUF_ACCUM,
)
from utils.dram_model import DramModel


def _pack_i8_row(vals):
    out = 0
    for i, v in enumerate(vals):
        out |= (int(v) & 0xFF) << (8 * i)
    return out


def _matmul_ref(a, b):
    c = [[0 for _ in range(16)] for _ in range(16)]
    for i in range(16):
        for j in range(16):
            acc = 0
            for k in range(16):
                acc += int(a[i][k]) * int(b[k][j])
            c[i][j] = acc
    return c


def _matmul_ref_32(a, b):
    c = [[0 for _ in range(32)] for _ in range(32)]
    for i in range(32):
        for j in range(32):
            acc = 0
            for k in range(32):
                acc += int(a[i][k]) * int(b[k][j])
            c[i][j] = acc
    return c


def _matmul_ref_16x64x16(a, b):
    c = [[0 for _ in range(16)] for _ in range(16)]
    for i in range(16):
        for j in range(16):
            acc = 0
            for k in range(64):
                acc += int(a[i][k]) * int(b[k][j])
            c[i][j] = acc
    return c


def _transpose_16x16(m):
    return [[m[r][c] for r in range(16)] for c in range(16)]


def _to_i8(v):
    v &= 0xFF
    return v - 256 if v >= 128 else v


async def _setup(dut, insns):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dram = DramModel()
    dram.write_program(insns)
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


async def _wait_halt(dut, max_cycles=300_000):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1 or int(dut.fault.value) == 1:
            return
    raise TimeoutError("DUT did not halt")


def _abuf_row_handle(dut, row):
    return dut.u_sram.u_abuf.mem[row]


def _wbuf_row_handle(dut, row):
    return dut.u_sram.u_wbuf.mem[row]


def _accum_row_u32x4(dut, row):
    v = int(dut.u_sram.u_accum.mem[row].value)
    return [
        (v >> 0) & 0xFFFFFFFF,
        (v >> 32) & 0xFFFFFFFF,
        (v >> 64) & 0xFFFFFFFF,
        (v >> 96) & 0xFFFFFFFF,
    ]


def _read_accum_16x16(dut, dst_off=0):
    out = [[0 for _ in range(16)] for _ in range(16)]
    for i in range(16):
        for grp in range(4):
            row = dst_off + i * 4 + grp
            lanes = _accum_row_u32x4(dut, row)
            for lane in range(4):
                j = grp * 4 + lane
                u = lanes[lane]
                out[i][j] = u if u < (1 << 31) else (u - (1 << 32))
    return out


def _write_abuf_matrix_transposed(dut, mat, off=0):
    # Controller expects ABUF rows to represent A^T rows for A @ B semantics.
    t = _transpose_16x16(mat)
    for r in range(16):
        _abuf_row_handle(dut, off + r).value = _pack_i8_row(t[r])


def _write_wbuf_matrix(dut, mat, off=0):
    for r in range(16):
        _wbuf_row_handle(dut, off + r).value = _pack_i8_row(mat[r])


def _write_abuf_tiled_32x32(dut, mat, off=0):
    for mt in range(2):
        for kt in range(2):
            tile = mt * 2 + kt
            for r in range(16):
                # ABUF tiles are stored transposed per 16x16 tile.
                vals = [mat[mt * 16 + c][kt * 16 + r] for c in range(16)]
                _abuf_row_handle(dut, off + tile * 16 + r).value = _pack_i8_row(vals)


def _write_wbuf_tiled_32x32(dut, mat, off=0):
    for kt in range(2):
        for nt in range(2):
            tile = kt * 2 + nt
            for r in range(16):
                vals = [mat[kt * 16 + r][nt * 16 + c] for c in range(16)]
                _wbuf_row_handle(dut, off + tile * 16 + r).value = _pack_i8_row(vals)


def _read_accum_32x32(dut, dst_off=0):
    out = [[0 for _ in range(32)] for _ in range(32)]
    for i in range(32):
        for j in range(32):
            mt, nt = i // 16, j // 16
            li, lj = i % 16, j % 16
            tile = mt * 2 + nt
            grp, lane = lj // 4, lj % 4
            row = dst_off + tile * 64 + li * 4 + grp
            lanes = _accum_row_u32x4(dut, row)
            u = lanes[lane]
            out[i][j] = u if u < (1 << 31) else (u - (1 << 32))
    return out


def _write_abuf_ktiles_16x64(dut, mat, off=0):
    # Tile order consumed by controller for m=1,n=1,k=4: ktile in [0..3].
    for kt in range(4):
        tile = kt
        for r in range(16):
            # ABUF tiles stored transposed per 16x16 tile.
            vals = [mat[c][kt * 16 + r] for c in range(16)]
            _abuf_row_handle(dut, off + tile * 16 + r).value = _pack_i8_row(vals)


def _write_wbuf_ktiles_64x16(dut, mat, off=0):
    # Tile order consumed by controller for m=1,n=1,k=4: ktile in [0..3].
    for kt in range(4):
        tile = kt
        for r in range(16):
            vals = [mat[kt * 16 + r][c] for c in range(16)]
            _wbuf_row_handle(dut, off + tile * 16 + r).value = _pack_i8_row(vals)


@cocotb.test()
async def test_matmul_identity(dut):
    a = [[(i * 3 + j) & 0x7F for j in range(16)] for i in range(16)]
    eye = [[1 if i == j else 0 for j in range(16)] for i in range(16)]
    exp = _matmul_ref(a, eye)

    prog = [
        CONFIG_TILE(1, 1, 1),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ]
    await _setup(dut, prog)
    _write_abuf_matrix_transposed(dut, a)
    _write_wbuf_matrix(dut, eye)
    await _wait_halt(dut)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = _read_accum_16x16(dut)
    assert got == exp, "identity matmul mismatch"


@cocotb.test()
async def test_matmul_accumulate_flag(dut):
    a = [[1 for _ in range(16)] for _ in range(16)]
    b = [[2 if i == j else 0 for j in range(16)] for i in range(16)]
    exp = _matmul_ref(a, b)

    prog = [
        CONFIG_TILE(1, 1, 1),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=1),
        SYNC(0b010),
        HALT(),
    ]
    await _setup(dut, prog)
    _write_abuf_matrix_transposed(dut, a)
    _write_wbuf_matrix(dut, b)
    await _wait_halt(dut)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = _read_accum_16x16(dut)
    exp2 = [[v * 2 for v in row] for row in exp]
    assert got == exp2, "accumulate flag mismatch"


@cocotb.test()
async def test_load_matmul_sync_integration(dut):
    """Integration: LOAD ABUF/WBUF from DRAM, then MATMUL, SYNC and verify ACCUM."""
    a = [[_to_i8((i * 7 + j * 3) & 0xFF) for j in range(16)] for i in range(16)]
    b = [[_to_i8((i * 5 + j * 11) & 0xFF) for j in range(16)] for i in range(16)]
    exp = _matmul_ref(a, b)

    # Prepare DRAM payloads: ABUF receives A^T layout expected by current controller.
    a_t = _transpose_16x16(a)
    a_bytes = b"".join(bytes((v & 0xFF) for v in row) for row in a_t)
    b_bytes = b"".join(bytes((v & 0xFF) for v in row) for row in b)

    src_a = 0x120000
    src_b = 0x121000

    prog = [
        SET_ADDR_LO(0, src_a), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 16, 0, 0),
        SYNC(0b001),
        SET_ADDR_LO(1, src_b), SET_ADDR_HI(1, 0),
        LOAD(BUF_WBUF, 0, 16, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ]

    dram = await _setup(dut, prog)
    dram.write_bytes(src_a, a_bytes)
    dram.write_bytes(src_b, b_bytes)
    await _wait_halt(dut, max_cycles=500_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = _read_accum_16x16(dut)
    assert got == exp, "LOAD->MATMUL integration mismatch"


@cocotb.test()
async def test_matmul_multitile_2x2x2(dut):
    a = [[((i * 7 + j * 5 + 3) % 11) - 5 for j in range(32)] for i in range(32)]
    b = [[((i * 3 + j * 9 + 1) % 13) - 6 for j in range(32)] for i in range(32)]
    exp = _matmul_ref_32(a, b)

    prog = [
        CONFIG_TILE(2, 2, 2),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ]
    await _setup(dut, prog)
    _write_abuf_tiled_32x32(dut, a)
    _write_wbuf_tiled_32x32(dut, b)
    await _wait_halt(dut, max_cycles=600_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = _read_accum_32x32(dut)
    assert got == exp, "multitile 2x2x2 matmul mismatch"


@cocotb.test()
async def test_matmul_signed_extremes(dut):
    a = [[-128 if ((i + j) & 1) else 127 for j in range(16)] for i in range(16)]
    b = [[127 if ((i * 3 + j * 5) & 1) else -128 for j in range(16)] for i in range(16)]
    exp = _matmul_ref(a, b)

    prog = [
        CONFIG_TILE(1, 1, 1),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ]
    await _setup(dut, prog)
    _write_abuf_matrix_transposed(dut, a)
    _write_wbuf_matrix(dut, b)
    await _wait_halt(dut)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = _read_accum_16x16(dut)
    assert got == exp, "signed extremes matmul mismatch"


@cocotb.test()
async def test_matmul_random_regression(dut):
    rng = random.Random(12345)

    for tc in range(8):
        a = [[rng.randint(-128, 127) for _ in range(16)] for _ in range(16)]
        b = [[rng.randint(-128, 127) for _ in range(16)] for _ in range(16)]
        exp = _matmul_ref(a, b)

        prog = [
            CONFIG_TILE(1, 1, 1),
            MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
            SYNC(0b010),
            HALT(),
        ]
        await _setup(dut, prog)
        _write_abuf_matrix_transposed(dut, a)
        _write_wbuf_matrix(dut, b)
        await _wait_halt(dut)

        assert int(dut.done.value) == 1
        assert int(dut.fault.value) == 0
        got = _read_accum_16x16(dut)
        assert got == exp, f"random regression mismatch on case {tc}"


@cocotb.test()
async def test_matmul_k4_boundary_stress(dut):
    a = [[127 if ((i + k) & 1) else -128 for k in range(64)] for i in range(16)]
    b = [[-128 if ((k * 7 + j) & 1) else 127 for j in range(16)] for k in range(64)]
    exp = _matmul_ref_16x64x16(a, b)

    prog = [
        CONFIG_TILE(1, 1, 4),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ]
    await _setup(dut, prog)
    _write_abuf_ktiles_16x64(dut, a)
    _write_wbuf_ktiles_64x16(dut, b)
    await _wait_halt(dut)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = _read_accum_16x16(dut)
    assert got == exp, "k=4 boundary stress mismatch"


@cocotb.test()
async def test_matmul_multitile_random_regression_2x2x2(dut):
    rng = random.Random(67890)

    for tc in range(4):
        a = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(32)]
        b = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(32)]
        exp = _matmul_ref_32(a, b)

        prog = [
            CONFIG_TILE(2, 2, 2),
            MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
            SYNC(0b010),
            HALT(),
        ]
        await _setup(dut, prog)
        _write_abuf_tiled_32x32(dut, a)
        _write_wbuf_tiled_32x32(dut, b)
        await _wait_halt(dut, max_cycles=700_000)

        assert int(dut.done.value) == 1
        assert int(dut.fault.value) == 0
        got = _read_accum_32x32(dut)
        assert got == exp, f"multitile random mismatch on case {tc}"
