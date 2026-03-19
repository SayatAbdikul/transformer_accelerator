// Common Verilator testbench infrastructure for TACCEL Phase 1.
//
// Provides:
//   - ClockGen: drives clk + rst_n
//   - AXI4SlaveModel: behavioral DRAM with instruction-write helpers
//   - run_until_halt(): drives simulation until done or fault or timeout
//
// Usage pattern:
//   auto tb = std::make_unique<Vtaccel_top>();
//   AXI4SlaveModel dram(1 << 24);  // 16 MB
//   dram.write_insn(tb, 0, NOP_WORD);
//   dram.write_insn(tb, 1, HALT_WORD);
//   reset(tb);
//   run_until_halt(tb, dram, /*timeout=*/1000);
//   assert(tb->done == 1);

#pragma once

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>

// ============================================================================
// ISA helper: build 64-bit big-endian instruction words
// (Matches software/taccel/isa/encoding.py exactly)
// ============================================================================
namespace insn {

constexpr int OPCODE_SHIFT      = 59;
constexpr int R_SRC1_BUF_SHIFT  = 57;
constexpr int R_SRC1_OFF_SHIFT  = 41;
constexpr int R_SRC2_BUF_SHIFT  = 39;
constexpr int R_SRC2_OFF_SHIFT  = 23;
constexpr int R_DST_BUF_SHIFT   = 21;
constexpr int R_DST_OFF_SHIFT   = 5;
constexpr int R_SREG_SHIFT      = 1;
constexpr int R_FLAGS_SHIFT     = 0;

constexpr int M_BUF_ID_SHIFT    = 57;
constexpr int M_SRAM_OFF_SHIFT  = 41;
constexpr int M_XFER_LEN_SHIFT  = 25;
constexpr int M_ADDR_REG_SHIFT  = 23;
constexpr int M_DRAM_OFF_SHIFT  = 7;

constexpr int B_SRC_BUF_SHIFT   = 57;
constexpr int B_SRC_OFF_SHIFT   = 41;
constexpr int B_DST_BUF_SHIFT   = 39;
constexpr int B_DST_OFF_SHIFT   = 23;
constexpr int B_LENGTH_SHIFT    = 7;
constexpr int B_SRC_ROWS_SHIFT  = 1;
constexpr int B_TRANSPOSE_SHIFT = 0;

constexpr int A_ADDR_REG_SHIFT  = 57;
constexpr int A_IMM28_SHIFT     = 29;

constexpr int C_M_SHIFT         = 49;
constexpr int C_N_SHIFT         = 39;
constexpr int C_K_SHIFT         = 29;

constexpr int SS_SREG_SHIFT     = 55;
constexpr int SS_SRC_MODE_SHIFT = 53;
constexpr int SS_IMM16_SHIFT    = 37;
constexpr int SYNC_MASK_SHIFT   = 56;

constexpr uint64_t NOP()         { return uint64_t(0x00) << OPCODE_SHIFT; }
constexpr uint64_t HALT()        { return uint64_t(0x01) << OPCODE_SHIFT; }
inline    uint64_t SYNC(uint8_t mask) {
    return (uint64_t(0x02) << OPCODE_SHIFT) |
           (uint64_t(mask & 0x7) << SYNC_MASK_SHIFT);
}
inline uint64_t CONFIG_TILE(int M, int N, int K) {
    assert(M >= 1 && M <= 1024 && N >= 1 && N <= 1024 && K >= 1 && K <= 1024);
    return (uint64_t(0x03) << OPCODE_SHIFT)   |
           (uint64_t(M-1)  << C_M_SHIFT)      |
           (uint64_t(N-1)  << C_N_SHIFT)      |
           (uint64_t(K-1)  << C_K_SHIFT);
}
inline uint64_t SET_SCALE(int sreg, uint16_t fp16_val, int src_mode = 0) {
    return (uint64_t(0x04)        << OPCODE_SHIFT)   |
           (uint64_t(sreg & 0xF)  << SS_SREG_SHIFT)  |
           (uint64_t(src_mode&3)  << SS_SRC_MODE_SHIFT) |
           (uint64_t(fp16_val)    << SS_IMM16_SHIFT);
}
inline uint64_t SET_ADDR_LO(int reg, uint32_t imm28) {
    return (uint64_t(0x05) << OPCODE_SHIFT) |
           (uint64_t(reg & 3) << A_ADDR_REG_SHIFT) |
           (uint64_t(imm28 & 0xFFFFFFF) << A_IMM28_SHIFT);
}
inline uint64_t SET_ADDR_HI(int reg, uint32_t imm28) {
    return (uint64_t(0x06) << OPCODE_SHIFT) |
           (uint64_t(reg & 3) << A_ADDR_REG_SHIFT) |
           (uint64_t(imm28 & 0xFFFFFFF) << A_IMM28_SHIFT);
}
inline uint64_t LOAD(int buf_id, int sram_off, int xfer_len,
                     int addr_reg, int dram_off) {
    return (uint64_t(0x07)       << OPCODE_SHIFT)  |
           (uint64_t(buf_id&3)   << M_BUF_ID_SHIFT)  |
           (uint64_t(sram_off)   << M_SRAM_OFF_SHIFT) |
           (uint64_t(xfer_len)   << M_XFER_LEN_SHIFT) |
           (uint64_t(addr_reg&3) << M_ADDR_REG_SHIFT) |
           (uint64_t(dram_off)   << M_DRAM_OFF_SHIFT);
}
inline uint64_t STORE(int buf_id, int sram_off, int xfer_len,
                      int addr_reg, int dram_off) {
    return (uint64_t(0x08)       << OPCODE_SHIFT)    |
           (uint64_t(buf_id&3)   << M_BUF_ID_SHIFT)  |
           (uint64_t(sram_off)   << M_SRAM_OFF_SHIFT) |
           (uint64_t(xfer_len)   << M_XFER_LEN_SHIFT) |
           (uint64_t(addr_reg&3) << M_ADDR_REG_SHIFT) |
           (uint64_t(dram_off)   << M_DRAM_OFF_SHIFT);
}
inline uint64_t MATMUL(int src1_buf, int src1_off, int src2_buf, int src2_off,
                       int dst_buf, int dst_off, int sreg, int flags = 0) {
    return (uint64_t(0x0A)         << OPCODE_SHIFT) |
           (uint64_t(src1_buf & 3) << R_SRC1_BUF_SHIFT) |
           (uint64_t(src1_off)     << R_SRC1_OFF_SHIFT) |
           (uint64_t(src2_buf & 3) << R_SRC2_BUF_SHIFT) |
           (uint64_t(src2_off)     << R_SRC2_OFF_SHIFT) |
           (uint64_t(dst_buf & 3)  << R_DST_BUF_SHIFT) |
           (uint64_t(dst_off)      << R_DST_OFF_SHIFT) |
           (uint64_t(sreg & 0xF)   << R_SREG_SHIFT) |
           (uint64_t(flags & 1)    << R_FLAGS_SHIFT);
}

// Illegal opcode for fault tests
constexpr uint64_t ILLEGAL_OP() { return uint64_t(0x11) << OPCODE_SHIFT; }

} // namespace insn

// ============================================================================
// AXI4 slave model: behavioral DRAM
// ============================================================================
class AXI4SlaveModel {
public:
    explicit AXI4SlaveModel(size_t size_bytes)
        : mem_(size_bytes, 0), pending_valid_(false), pending_addr_(0),
          pending_timer_(0), r_valid_(false),
          aw_pending_(false), aw_addr_(0), aw_len_(0), w_beat_cnt_(0),
          b_pending_(false)
    {}

    // Write a big-endian 64-bit instruction word at instruction index pc_idx
    void write_insn(int pc_idx, uint64_t word) {
        size_t base = (size_t)pc_idx * 8;
        assert(base + 7 < mem_.size());
        mem_[base+0] = (word >> 56) & 0xFF;
        mem_[base+1] = (word >> 48) & 0xFF;
        mem_[base+2] = (word >> 40) & 0xFF;
        mem_[base+3] = (word >> 32) & 0xFF;
        mem_[base+4] = (word >> 24) & 0xFF;
        mem_[base+5] = (word >> 16) & 0xFF;
        mem_[base+6] = (word >>  8) & 0xFF;
        mem_[base+7] = (word >>  0) & 0xFF;
    }

    // Write a program sequence starting at instruction 0
    void write_program(const std::vector<uint64_t>& insns) {
        for (size_t i = 0; i < insns.size(); i++)
            write_insn((int)i, insns[i]);
    }

    // Write raw bytes at byte address
    void write_bytes(size_t addr, const uint8_t* data, size_t len) {
        assert(addr + len <= mem_.size());
        std::memcpy(mem_.data() + addr, data, len);
    }

    uint8_t read_byte(size_t addr) const {
        assert(addr < mem_.size());
        return mem_[addr];
    }

    // Drive AXI slave outputs given DUT's master outputs (call every cycle)
    void tick(Vtaccel_top* dut, int latency = 2) {
        // AR channel: always ready; accept new request
        dut->m_axi_ar_ready = 1;
        if (dut->m_axi_ar_valid && !pending_valid_) {
            uint64_t aligned = (uint64_t)dut->m_axi_ar_addr & ~uint64_t(0xF);
            pending_addr_  = aligned;
            pending_timer_ = latency;
            pending_valid_ = true;
        }

        // R channel
        if (r_valid_) {
            if (dut->m_axi_r_ready) {
                r_valid_       = false;
                dut->m_axi_r_valid = 0;
            }
            // hold r_valid + r_data until accepted
        } else if (pending_valid_) {
            if (pending_timer_ > 0) {
                pending_timer_--;
            } else {
                // Build 128-bit (16-byte) response, little-endian
                // rdata[7:0] = mem[addr+0], rdata[15:8] = mem[addr+1], ...
                // We drive the 128-bit field as two 64-bit halves:
                //   rdata_lo = mem[addr+0..7]  (insn at even PC)
                //   rdata_hi = mem[addr+8..15] (insn at odd PC)
                // Note: we store instructions big-endian so mem[addr] = MSByte
                // The fetch_unit does a byte-swap after extraction.
                uint64_t lo = 0, hi = 0;
                for (int b = 0; b < 8 && pending_addr_+b < mem_.size(); b++)
                    lo |= (uint64_t)mem_[pending_addr_+b] << (b*8);
                for (int b = 0; b < 8 && pending_addr_+8+b < mem_.size(); b++)
                    hi |= (uint64_t)mem_[pending_addr_+8+b] << (b*8);

                // Drive r_data as a byte array (128 bits)
                // Verilator represents [127:0] as WData[3] (4×32-bit words)
                uint32_t* rdata = dut->m_axi_r_data.data();
                rdata[0] = lo & 0xFFFFFFFF;
                rdata[1] = (lo >> 32) & 0xFFFFFFFF;
                rdata[2] = hi & 0xFFFFFFFF;
                rdata[3] = (hi >> 32) & 0xFFFFFFFF;

                dut->m_axi_r_valid = 1;
                dut->m_axi_r_last  = 1;
                dut->m_axi_r_resp  = 0;
                r_valid_      = true;
                pending_valid_ = false;
            }
        }
    }

private:
    std::vector<uint8_t> mem_;
    bool     pending_valid_;
    uint64_t pending_addr_;
    int      pending_timer_;
    bool     r_valid_;
};

// ============================================================================
// Simulation helpers
// ============================================================================

// Apply reset for n cycles
inline void do_reset(Vtaccel_top* dut, int cycles = 10) {
    dut->rst_n = 0;
    dut->start = 0;
    for (int i = 0; i < cycles; i++) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
    }
    dut->rst_n = 1;
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

// Tick one clock cycle, driving AXI slave
inline void tick(Vtaccel_top* dut, AXI4SlaveModel& dram, int latency = 2) {
    // Sample outputs on negedge (after posedge registered)
    dut->clk = 0;
    dut->eval();
    dram.tick(dut, latency);  // update slave combinational outputs
    dut->eval();

    dut->clk = 1;
    dut->eval();
}

// Run until done or fault; returns cycle count
inline int run_until_halt(Vtaccel_top* dut, AXI4SlaveModel& dram,
                           int timeout = 100000, int latency = 2) {
    for (int i = 0; i < timeout; i++) {
        tick(dut, dram, latency);
        if (dut->done || dut->fault)
            return i + 1;
    }
    throw std::runtime_error("Simulation timeout: DUT did not halt");
}

// ============================================================================
// Test assertion macro
// ============================================================================
#define EXPECT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s  (at %s:%d)\n", msg, __FILE__, __LINE__); \
            std::exit(1); \
        } \
    } while(0)

#define PASS(name) \
    fprintf(stdout, "PASS: %s\n", name)
