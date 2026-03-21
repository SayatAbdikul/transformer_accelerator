// Verilator DMA tests for TACCEL Phase 2.
//
// Tests:
//   load_single_beat   — LOAD 1×16 bytes; verify via STORE roundtrip
//   load_multi_beat    — LOAD 16×16 bytes (256 B); verify via STORE roundtrip
//   store_roundtrip    — LOAD src→SRAM, STORE SRAM→dst; src==dst bytes
//   load_to_wbuf       — LOAD to WBUF buffer
//   load_to_accum      — LOAD to ACCUM buffer
//   addr_reg_r1        — R1 used for LOAD, R0 untouched
//   dram_oob_fault     — STORE with end_addr > DRAM_SIZE → fault_code=2
//   load_dispatch_async — LOAD dispatched (non-blocking) → SYNC(001) waits → HALT

#include "Vtaccel_top.h"
#include "verilated.h"
#include "include/testbench.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

static int tests_run  = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
    printf("PASS: %s\n", name); tests_pass++; tests_run++; } while(0)
#define TEST_FAIL(name, msg) do { \
    fprintf(stderr, "FAIL: %s — %s\n", name, msg); std::exit(1); } while(0)

// ============================================================================
// Simulation helper
// ============================================================================
struct Sim {
    std::unique_ptr<Vtaccel_top> dut;
    AXI4SlaveModel dram;

    explicit Sim() : dut(std::make_unique<Vtaccel_top>()), dram(16*1024*1024) {
        do_reset(dut.get());
    }

    void load(const std::vector<uint64_t>& prog) {
        dram.write_program(prog);
    }

    // Start execution; run until halt or fault
    int run(int timeout = 100000) {
        dut->start = 1;
        tick(dut.get(), dram);
        dut->start = 0;
        return run_until_halt(dut.get(), dram, timeout);
    }
};

// ============================================================================
// test: LOAD 1 beat → STORE roundtrip (ABUF)
// ============================================================================
static void test_load_single_beat() {
    const char* name = "load_single_beat";
    Sim s;

    // Write 16 known bytes at DRAM address 0x10000 (src)
    constexpr uint64_t SRC_ADDR  = 0x10000;
    constexpr uint64_t DST_ADDR  = 0x20000;
    uint8_t src_data[16];
    for (int i = 0; i < 16; i++) src_data[i] = (uint8_t)(0xA0 + i);
    s.dram.write_bytes(SRC_ADDR, src_data, 16);

    // Program:
    //   SET_ADDR_LO R0, SRC_ADDR   ; R0 = SRC
    //   LOAD ABUF[0], xfer=1, R0, dram_off=0
    //   SYNC 001
    //   SET_ADDR_LO R1, DST_ADDR   ; R1 = DST
    //   STORE ABUF[0], xfer=1, R1, dram_off=0
    //   SYNC 001
    //   HALT
    s.load({
        insn::SET_ADDR_LO(0, SRC_ADDR),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD (0/*ABUF*/, 0, 1, 0, 0),
        insn::SYNC (0b001),
        insn::SET_ADDR_LO(1, DST_ADDR),
        insn::SET_ADDR_HI(1, 0),
        insn::STORE(0/*ABUF*/, 0, 1, 1, 0),
        insn::SYNC (0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done) TEST_FAIL(name, "DUT did not halt");
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    // Verify DRAM[DST] == src_data
    for (int i = 0; i < 16; i++) {
        uint8_t got = s.dram.read_byte(DST_ADDR + i);
        if (got != src_data[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD 16 beats (256 bytes) → STORE roundtrip
// ============================================================================
static void test_load_multi_beat() {
    const char* name = "load_multi_beat";
    Sim s;

    constexpr uint64_t SRC = 0x30000;
    constexpr uint64_t DST = 0x40000;
    constexpr int NBYTES = 256;   // 16 rows × 16 bytes

    uint8_t src[NBYTES];
    for (int i = 0; i < NBYTES; i++) src[i] = (uint8_t)i;
    s.dram.write_bytes(SRC, src, NBYTES);

    s.load({
        insn::SET_ADDR_LO(0, SRC),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD (0, 0, 16, 0, 0),   // xfer_len=16 → 256 bytes
        insn::SYNC (0b001),
        insn::SET_ADDR_LO(1, DST),
        insn::SET_ADDR_HI(1, 0),
        insn::STORE(0, 0, 16, 1, 0),
        insn::SYNC (0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done) TEST_FAIL(name, "DUT did not halt");
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    for (int i = 0; i < NBYTES; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD to WBUF
// ============================================================================
static void test_load_to_wbuf() {
    const char* name = "load_to_wbuf";
    Sim s;

    constexpr uint64_t SRC = 0x50000;
    constexpr uint64_t DST = 0x60000;
    uint8_t src[16];
    for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0xB0 + i);
    s.dram.write_bytes(SRC, src, 16);

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD (1/*WBUF*/, 0, 1, 0, 0),
        insn::SYNC (0b001),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(1/*WBUF*/, 0, 1, 1, 0),
        insn::SYNC (0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD to ACCUM
// ============================================================================
static void test_load_to_accum() {
    const char* name = "load_to_accum";
    Sim s;

    constexpr uint64_t SRC = 0x70000;
    constexpr uint64_t DST = 0x80000;
    uint8_t src[16];
    for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0xC0 + i);
    s.dram.write_bytes(SRC, src, 16);

    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD (2/*ACCUM*/, 0, 1, 0, 0),
        insn::SYNC (0b001),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(2/*ACCUM*/, 0, 1, 1, 0),
        insn::SYNC (0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: R1 used independently from R0
// ============================================================================
static void test_addr_reg_independence() {
    const char* name = "addr_reg_independence";
    Sim s;

    // Data at two separate DRAM regions
    constexpr uint64_t SRC_A = 0x90000;
    constexpr uint64_t DST_A = 0xA0000;
    constexpr uint64_t SRC_B = 0xB0000;
    constexpr uint64_t DST_B = 0xC0000;

    uint8_t pa[16], pb[16];
    for (int i = 0; i < 16; i++) { pa[i] = (uint8_t)(0x11 + i); pb[i] = (uint8_t)(0x88 + i); }
    s.dram.write_bytes(SRC_A, pa, 16);
    s.dram.write_bytes(SRC_B, pb, 16);

    // Use R2 for A (src), R3 for B (src), R0 for A (dst), R1 for B (dst)
    s.load({
        insn::SET_ADDR_LO(2, SRC_A), insn::SET_ADDR_HI(2, 0),
        insn::SET_ADDR_LO(3, SRC_B), insn::SET_ADDR_HI(3, 0),
        insn::LOAD(0/*ABUF*/, 0, 1, 2, 0),   // load A to ABUF[0]
        insn::SYNC(0b001),
        insn::LOAD(1/*WBUF*/, 0, 1, 3, 0),   // load B to WBUF[0]
        insn::SYNC(0b001),
        insn::SET_ADDR_LO(0, DST_A), insn::SET_ADDR_HI(0, 0),
        insn::SET_ADDR_LO(1, DST_B), insn::SET_ADDR_HI(1, 0),
        insn::STORE(0/*ABUF*/, 0, 1, 0, 0),
        insn::SYNC(0b001),
        insn::STORE(1/*WBUF*/, 0, 1, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST_A + i) != pa[i]) TEST_FAIL(name, "A data mismatch");
        if (s.dram.read_byte(DST_B + i) != pb[i]) TEST_FAIL(name, "B data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// test: DRAM OOB → FAULT_DRAM_OOB (code 2)
// ============================================================================
static void test_dram_oob_fault() {
    const char* name = "dram_oob_fault";
    Sim s;

    // Set R0 to near end of 16 MB DRAM; xfer_len=2 → 32 bytes past end
    // DRAM_SIZE = 16 MB = 0x1000000
    // addr = 0xFFFFF0 = 16777200; end = 16777200 + 32 = 0x1000010 > 0x1000000
    constexpr uint64_t NEAR_END = 0xFFFFF0;

    s.load({
        insn::SET_ADDR_LO(0, NEAR_END),
        insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0, 0, 2, 0, 0),   // OOB: 2 × 16 bytes past NEAR_END
        insn::SYNC(0b001),            // will see dma_fault in SYNC_WAIT
        insn::HALT(),                 // unreachable
    });

    s.run(200000);
    if (!s.dut->fault)      TEST_FAIL(name, "expected fault=1");
    if (s.dut->fault_code != 2) TEST_FAIL(name, "expected fault_code=2 (DRAM_OOB)");
    TEST_PASS(name);
}

// ============================================================================
// test: LOAD dispatch is non-blocking; SYNC(001) waits correctly
// ============================================================================
static void test_load_dispatch_async() {
    const char* name = "load_dispatch_async";
    Sim s;

    constexpr uint64_t SRC = 0xD0000;
    uint8_t src[16] = {};
    s.dram.write_bytes(SRC, src, 16);

    // LOAD dispatched → NOP → SYNC(001) → HALT
    // SYNC should stall until DMA completes, then HALT.
    s.load({
        insn::SET_ADDR_LO(0, SRC), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0, 0, 1, 0, 0),
        insn::NOP(),                  // prove pipeline continues after dispatch
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done) TEST_FAIL(name, "DUT did not halt");
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");
    TEST_PASS(name);
}

// ============================================================================
// test: dram_off field shifts address correctly
// ============================================================================
static void test_dram_offset() {
    const char* name = "dram_offset";
    Sim s;

    // R0 = 0x50000 (base); dram_off=4 → effective = 0x50000 + 4*16 = 0x50040
    constexpr uint64_t BASE = 0x50040;   // effective address
    constexpr uint64_t DST  = 0xE0000;
    uint8_t src[16];
    for (int i = 0; i < 16; i++) src[i] = (uint8_t)(0xD0 + i);
    s.dram.write_bytes(BASE, src, 16);   // write at effective address

    s.load({
        // base address is 0x50000; dram_off=4 → 0x50000 + 64 = 0x50040
        insn::SET_ADDR_LO(0, 0x50000), insn::SET_ADDR_HI(0, 0),
        insn::LOAD(0, 0, 1, 0, 4),   // dram_off=4 → byte +64
        insn::SYNC(0b001),
        insn::SET_ADDR_LO(1, DST), insn::SET_ADDR_HI(1, 0),
        insn::STORE(0, 0, 1, 1, 0),
        insn::SYNC(0b001),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");
    for (int i = 0; i < 16; i++) {
        if (s.dram.read_byte(DST + i) != src[i]) TEST_FAIL(name, "data mismatch");
    }
    TEST_PASS(name);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    test_load_single_beat();
    test_load_multi_beat();
    test_load_to_wbuf();
    test_load_to_accum();
    test_addr_reg_independence();
    test_dram_oob_fault();
    test_load_dispatch_async();
    test_dram_offset();

    printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
