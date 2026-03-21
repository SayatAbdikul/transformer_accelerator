"""cocotb DMA tests for TACCEL Phase 2.

Tests LOAD (DRAM→SRAM) and STORE (SRAM→DRAM) via AXI4 master.
Verification uses LOAD+STORE roundtrips (data appears in DRAM after STORE).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from utils.insn_builder import (
    NOP, HALT, SYNC, SET_ADDR_LO, SET_ADDR_HI, LOAD, STORE,
    BUF_ABUF, BUF_WBUF, BUF_ACCUM,
)
from utils.dram_model import DramModel


# ---------------------------------------------------------------------------
# Common setup helper
# ---------------------------------------------------------------------------

async def _setup(
    dut,
    insns,
    dram_writes=None,
    write_slave_kwargs=None,
):
    """Clock, reset, load program+data, start, return dram model."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dram = DramModel()
    dram.write_program(insns)
    if dram_writes:
        for addr, data in dram_writes.items():
            dram.write_bytes(addr, bytes(data))
    cocotb.start_soon(dram.axi_slave(dut))
    cocotb.start_soon(dram.axi_write_slave(dut, **(write_slave_kwargs or {})))

    dut.rst_n.value = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    return dram


async def _wait_halt(dut, max_cycles=100_000):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1 or dut.fault.value == 1:
            return
    raise TimeoutError("DUT did not halt")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_store_is_store_decode_signal(dut):
    """Sanity check: STORE instruction must assert dma_is_store/is_store_q."""
    DST = 0x20000
    prog = [
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_ABUF, 0, 1, 1, 0),
        SYNC(0b001),
        HALT(),
    ]

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dram = DramModel()
    dram.write_program(prog)
    cocotb.start_soon(dram.axi_slave(dut))

    dut.rst_n.value = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    saw_dispatch = False
    for _ in range(500):
        await RisingEdge(dut.clk)
        dma_dispatch = int(dut.dma_dispatch.value) if dut.dma_dispatch.value.is_resolvable else 0
        if dma_dispatch:
            saw_dispatch = True
            assert int(dut.dma_is_store.value) == 1, "dma_is_store was not asserted on STORE dispatch"
            # is_store_q is a sequential latch updated on this edge; verify on
            # the next clock to avoid delta-cycle sampling ambiguity.
            await RisingEdge(dut.clk)
            assert int(dut.u_dma.is_store_q.value) == 1, "u_dma.is_store_q did not latch STORE"
            break

    assert saw_dispatch, "did not observe dma_dispatch for STORE within 500 cycles"


@cocotb.test()
async def test_load_store_roundtrip(dut):
    """LOAD 1x16 bytes from DRAM, STORE back to different address, verify."""
    SRC = 0x10000
    DST = 0x20000
    src_data = bytes(range(16))
    prog = [
        SET_ADDR_LO(0, SRC), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 1, 0, 0),
        SYNC(0b001),
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_ABUF, 0, 1, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {SRC: src_data})
    await _wait_halt(dut)

    assert dut.done.value == 1,  "expected done"
    assert dut.fault.value == 0, "unexpected fault"
    assert bytes(dram.mem[DST:DST+16]) == src_data, "data mismatch"
    dut._log.info("load_store_roundtrip: OK")


@cocotb.test()
async def test_load_multi_beat(dut):
    """LOAD 16 beats (256 bytes) → STORE → verify all 256 bytes."""
    SRC = 0x30000
    DST = 0x40000
    src_data = bytes(i & 0xFF for i in range(256))
    prog = [
        SET_ADDR_LO(0, SRC), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 16, 0, 0),
        SYNC(0b001),
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_ABUF, 0, 16, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {SRC: src_data})
    await _wait_halt(dut)

    assert dut.done.value == 1,  "expected done"
    assert dut.fault.value == 0, "unexpected fault"
    got = bytes(dram.mem[DST:DST+256])
    assert got == src_data, f"data mismatch at byte {next(i for i,b in enumerate(got) if b!=src_data[i])}"
    dut._log.info("load_multi_beat: OK (256 bytes)")


@cocotb.test()
async def test_load_to_wbuf(dut):
    """LOAD to WBUF buffer, STORE back."""
    SRC = 0x50000
    DST = 0x60000
    src_data = bytes(0xBB ^ i for i in range(16))
    prog = [
        SET_ADDR_LO(0, SRC), SET_ADDR_HI(0, 0),
        LOAD(BUF_WBUF, 0, 1, 0, 0),
        SYNC(0b001),
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_WBUF, 0, 1, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {SRC: src_data})
    await _wait_halt(dut)

    assert dut.done.value == 1 and dut.fault.value == 0
    assert bytes(dram.mem[DST:DST+16]) == src_data
    dut._log.info("load_to_wbuf: OK")


@cocotb.test()
async def test_load_to_accum(dut):
    """LOAD to ACCUM buffer, STORE back."""
    SRC = 0x70000
    DST = 0x80000
    src_data = bytes(0xCC ^ i for i in range(16))
    prog = [
        SET_ADDR_LO(0, SRC), SET_ADDR_HI(0, 0),
        LOAD(BUF_ACCUM, 0, 1, 0, 0),
        SYNC(0b001),
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_ACCUM, 0, 1, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {SRC: src_data})
    await _wait_halt(dut)

    assert dut.done.value == 1 and dut.fault.value == 0
    assert bytes(dram.mem[DST:DST+16]) == src_data
    dut._log.info("load_to_accum: OK")


@cocotb.test()
async def test_addr_reg_independence(dut):
    """R2 and R3 used for LOADs; R0/R1 used for STOREs — all independent."""
    SRC_A = 0x90000
    SRC_B = 0xA0000
    DST_A = 0xB0000
    DST_B = 0xC0000
    data_a = bytes(0x11 ^ i for i in range(16))
    data_b = bytes(0x88 ^ i for i in range(16))

    prog = [
        SET_ADDR_LO(2, SRC_A), SET_ADDR_HI(2, 0),
        SET_ADDR_LO(3, SRC_B), SET_ADDR_HI(3, 0),
        LOAD(BUF_ABUF, 0, 1, 2, 0),
        SYNC(0b001),
        LOAD(BUF_WBUF, 0, 1, 3, 0),
        SYNC(0b001),
        SET_ADDR_LO(0, DST_A), SET_ADDR_HI(0, 0),
        SET_ADDR_LO(1, DST_B), SET_ADDR_HI(1, 0),
        STORE(BUF_ABUF, 0, 1, 0, 0),
        SYNC(0b001),
        STORE(BUF_WBUF, 0, 1, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {SRC_A: data_a, SRC_B: data_b})
    await _wait_halt(dut)

    assert dut.done.value == 1 and dut.fault.value == 0
    assert bytes(dram.mem[DST_A:DST_A+16]) == data_a, "A mismatch"
    assert bytes(dram.mem[DST_B:DST_B+16]) == data_b, "B mismatch"
    dut._log.info("addr_reg_independence: OK")


@cocotb.test()
async def test_dram_oob_fault(dut):
    """LOAD past DRAM end → fault_code = 2 (FAULT_DRAM_OOB)."""
    # DRAM_SIZE = 16 MB = 0x1000000
    # addr = 0xFFFFF0; end = 0xFFFFF0 + 2*16 = 0x1000010 > 0x1000000
    NEAR_END = 0xFFFFF0
    prog = [
        SET_ADDR_LO(0, NEAR_END), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 2, 0, 0),   # OOB
        SYNC(0b001),                   # detects fault in SYNC_WAIT
        HALT(),                        # unreachable
    ]
    await _setup(dut, prog)
    await _wait_halt(dut)

    assert dut.fault.value == 1, "expected fault"
    assert int(dut.fault_code.value) == 2, \
        f"expected fault_code=2, got {int(dut.fault_code.value)}"
    dut._log.info(f"dram_oob_fault: fault_code={int(dut.fault_code.value)}")


@cocotb.test()
async def test_load_dispatch_async(dut):
    """LOAD is non-blocking; SYNC(001) stalls until DMA completes."""
    SRC = 0xD0000
    prog = [
        SET_ADDR_LO(0, SRC), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 1, 0, 0),
        NOP(),          # pipeline continues after dispatch
        SYNC(0b001),    # stalls until DMA done
        HALT(),
    ]
    dram = await _setup(dut, prog, {SRC: bytes(16)})
    await _wait_halt(dut)

    assert dut.done.value == 1, "expected done"
    assert dut.fault.value == 0, "unexpected fault"
    dut._log.info("load_dispatch_async: OK")


@cocotb.test()
async def test_dram_offset_field(dut):
    """dram_off=4 shifts effective address by 4×16=64 bytes."""
    BASE   = 0x50000
    OFFSET = 4        # dram_off field value → +64 bytes
    SRC    = BASE + OFFSET * 16   # 0x50040
    DST    = 0xE0000
    src_data = bytes(0xD0 ^ i for i in range(16))

    prog = [
        SET_ADDR_LO(0, BASE), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 1, 0, OFFSET),  # effective = BASE + 64
        SYNC(0b001),
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_ABUF, 0, 1, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {SRC: src_data})
    await _wait_halt(dut)

    assert dut.done.value == 1 and dut.fault.value == 0
    assert bytes(dram.mem[DST:DST+16]) == src_data, "data mismatch"
    dut._log.info("dram_offset_field: OK")


@cocotb.test()
async def test_store_oob_fault(dut):
    """STORE past DRAM end should raise FAULT_DRAM_OOB."""
    NEAR_END = 0xFFFFF0
    prog = [
        SET_ADDR_LO(0, NEAR_END), SET_ADDR_HI(0, 0),
        STORE(BUF_ABUF, 0, 2, 0, 0),   # OOB: 2 beats from 0xFFFFF0 crosses 16MB
        SYNC(0b001),
        HALT(),
    ]
    await _setup(dut, prog)
    await _wait_halt(dut)

    assert dut.fault.value == 1, "expected fault"
    assert int(dut.fault_code.value) == 2, (
        f"expected fault_code=2, got {int(dut.fault_code.value)}"
    )
    dut._log.info("store_oob_fault: OK")


@cocotb.test()
async def test_load_store_max_len_256(dut):
    """Boundary: xfer_len=256 beats (AXI len=255) should complete and match."""
    SRC = 0x120000
    DST = 0x140000
    nbytes = 256 * 16
    src_data = bytes((i * 7) & 0xFF for i in range(nbytes))
    prog = [
        SET_ADDR_LO(0, SRC), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 256, 0, 0),
        SYNC(0b001),
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_ABUF, 0, 256, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(dut, prog, {SRC: src_data})
    await _wait_halt(dut, max_cycles=300_000)

    assert dut.done.value == 1, "expected done"
    assert dut.fault.value == 0, "unexpected fault"
    assert bytes(dram.mem[DST:DST+nbytes]) == src_data, "data mismatch"
    dut._log.info("load_store_max_len_256: OK")


@cocotb.test()
async def test_store_backpressure_aw_w(dut):
    """STORE should complete when AW/W are stalled for several cycles."""
    SRC = 0x160000
    DST = 0x180000
    src_data = bytes((0x3A + i) & 0xFF for i in range(64))
    prog = [
        SET_ADDR_LO(0, SRC), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 4, 0, 0),
        SYNC(0b001),
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_ABUF, 0, 4, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(
        dut,
        prog,
        {SRC: src_data},
        write_slave_kwargs={
            "w_stall_cycles": 2,
            "b_valid_delay_cycles": 2,
        },
    )
    await _wait_halt(dut)

    assert dut.done.value == 1, "expected done"
    assert dut.fault.value == 0, "unexpected fault"
    assert bytes(dram.mem[DST:DST+64]) == src_data, "data mismatch"
    dut._log.info("store_backpressure_aw_w: OK")


@cocotb.test()
async def test_store_bresp_non_okay_path(dut):
    """Exercise DMA B channel when slave returns non-OKAY BRESP."""
    SRC = 0x1A0000
    DST = 0x1C0000
    src_data = bytes((0x90 ^ i) & 0xFF for i in range(16))
    prog = [
        SET_ADDR_LO(0, SRC), SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 1, 0, 0),
        SYNC(0b001),
        SET_ADDR_LO(1, DST), SET_ADDR_HI(1, 0),
        STORE(BUF_ABUF, 0, 1, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await _setup(
        dut,
        prog,
        {SRC: src_data},
        write_slave_kwargs={"b_resp": 2},
    )
    await _wait_halt(dut)

    # Current RTL does not fault on BRESP and only waits for BVALID handshake.
    assert dut.done.value == 1, "expected done"
    assert dut.fault.value == 0, "unexpected fault"
    assert bytes(dram.mem[DST:DST+16]) == src_data, "data mismatch"
    dut._log.info("store_bresp_non_okay_path: exercised")
