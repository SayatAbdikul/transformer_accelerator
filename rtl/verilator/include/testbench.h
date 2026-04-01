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
inline uint64_t BUF_COPY(int src_buf, int src_off, int dst_buf, int dst_off,
                         int length, int src_rows, int transpose = 0) {
    return (uint64_t(0x09)        << OPCODE_SHIFT)      |
           (uint64_t(src_buf & 3) << B_SRC_BUF_SHIFT)   |
           (uint64_t(src_off)     << B_SRC_OFF_SHIFT)   |
           (uint64_t(dst_buf & 3) << B_DST_BUF_SHIFT)   |
           (uint64_t(dst_off)     << B_DST_OFF_SHIFT)   |
           (uint64_t(length)      << B_LENGTH_SHIFT)    |
           (uint64_t(src_rows)    << B_SRC_ROWS_SHIFT)  |
           (uint64_t(transpose & 1) << B_TRANSPOSE_SHIFT);
}
inline uint64_t R_TYPE(int opcode, int src1_buf, int src1_off, int src2_buf,
                       int src2_off, int dst_buf, int dst_off, int sreg,
                       int flags = 0) {
    return (uint64_t(opcode & 0x1F) << OPCODE_SHIFT)   |
           (uint64_t(src1_buf & 3)  << R_SRC1_BUF_SHIFT) |
           (uint64_t(src1_off)      << R_SRC1_OFF_SHIFT) |
           (uint64_t(src2_buf & 3)  << R_SRC2_BUF_SHIFT) |
           (uint64_t(src2_off)      << R_SRC2_OFF_SHIFT) |
           (uint64_t(dst_buf & 3)   << R_DST_BUF_SHIFT) |
           (uint64_t(dst_off)       << R_DST_OFF_SHIFT) |
           (uint64_t(sreg & 0xF)    << R_SREG_SHIFT) |
           (uint64_t(flags & 1)     << R_FLAGS_SHIFT);
}
inline uint64_t MATMUL(int src1_buf, int src1_off, int src2_buf, int src2_off,
                       int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x0A, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t REQUANT(int src1_buf, int src1_off, int dst_buf, int dst_off,
                        int sreg, int flags = 0) {
    return R_TYPE(0x0B, src1_buf, src1_off, 0, 0, dst_buf, dst_off, sreg, flags);
}
inline uint64_t REQUANT_PC(int src1_buf, int src1_off, int src2_buf, int src2_off,
                           int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x11, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t SCALE_MUL(int src1_buf, int src1_off, int dst_buf, int dst_off,
                          int sreg, int flags = 0) {
    return R_TYPE(0x0C, src1_buf, src1_off, 0, 0, dst_buf, dst_off, sreg, flags);
}
inline uint64_t VADD(int src1_buf, int src1_off, int src2_buf, int src2_off,
                     int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x0D, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t SOFTMAX(int src1_buf, int src1_off, int dst_buf, int dst_off,
                        int sreg, int flags = 0) {
    return R_TYPE(0x0E, src1_buf, src1_off, 0, 0, dst_buf, dst_off, sreg, flags);
}
inline uint64_t LAYERNORM(int src1_buf, int src1_off, int src2_buf, int src2_off,
                          int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x0F, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t GELU(int src1_buf, int src1_off, int dst_buf, int dst_off,
                     int sreg, int flags = 0) {
    return R_TYPE(0x10, src1_buf, src1_off, 0, 0, dst_buf, dst_off, sreg, flags);
}
inline uint64_t SOFTMAX_ATTNV(int src1_buf, int src1_off, int src2_buf, int src2_off,
                              int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x12, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}
inline uint64_t DEQUANT_ADD(int src1_buf, int src1_off, int src2_buf, int src2_off,
                            int dst_buf, int dst_off, int sreg, int flags = 0) {
    return R_TYPE(0x13, src1_buf, src1_off, src2_buf, src2_off, dst_buf, dst_off, sreg, flags);
}

// Illegal opcode for fault tests
constexpr uint64_t ILLEGAL_OP() { return uint64_t(0x14) << OPCODE_SHIFT; }

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
          b_pending_(false), next_r_resp_(0), next_r_last_override_(-1)
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

    void inject_next_read(int resp, int force_last = -1) {
        next_r_resp_ = resp & 0x3;
        next_r_last_override_ = force_last;
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

    // Drive AXI slave outputs given DUT's master outputs (call every cycle).
    // Handles AR/R (reads, single-beat model) and AW/W/B (writes, multi-beat).
    void tick(Vtaccel_top* dut, int latency = 2) {
        // ----------------------------------------------------------------
        // AR channel: accept one outstanding read request at a time
        // ----------------------------------------------------------------
        dut->m_axi_ar_ready = 1;
        if (dut->m_axi_ar_valid && !pending_valid_) {
            uint64_t aligned = (uint64_t)dut->m_axi_ar_addr & ~uint64_t(0xF);
            pending_addr_    = aligned;
            pending_ar_len_  = dut->m_axi_ar_len;   // burst beat count - 1
            pending_beat_    = 0;
            pending_timer_   = latency;
            pending_valid_   = true;
        }

        // ----------------------------------------------------------------
        // R channel: serve one beat per call when timer fires
        // ----------------------------------------------------------------
        if (r_valid_) {
            if (dut->m_axi_r_ready) {
                r_valid_           = false;
                dut->m_axi_r_valid = 0;
                dut->m_axi_r_last  = 0;
                // If more beats remain in this burst, queue next beat
                if (pending_beat_ <= (int)pending_ar_len_) {
                    pending_timer_ = 1;  // next beat next cycle
                }
            }
        } else if (pending_valid_) {
            if (pending_timer_ > 0) {
                pending_timer_--;
            } else {
                // Build 16-byte response for current beat
                uint64_t beat_addr = pending_addr_ + (uint64_t)pending_beat_ * 16;
                uint64_t lo = 0, hi = 0;
                for (int b = 0; b < 8 && beat_addr+b < mem_.size(); b++)
                    lo |= (uint64_t)mem_[beat_addr+b] << (b*8);
                for (int b = 0; b < 8 && beat_addr+8+b < mem_.size(); b++)
                    hi |= (uint64_t)mem_[beat_addr+8+b] << (b*8);

                uint32_t* rdata = dut->m_axi_r_data.data();
                rdata[0] = lo & 0xFFFFFFFF;
                rdata[1] = (lo >> 32) & 0xFFFFFFFF;
                rdata[2] = hi & 0xFFFFFFFF;
                rdata[3] = (hi >> 32) & 0xFFFFFFFF;

                bool is_last = (pending_beat_ >= (int)pending_ar_len_);
                if (next_r_last_override_ >= 0)
                    is_last = (next_r_last_override_ != 0);
                dut->m_axi_r_valid = 1;
                dut->m_axi_r_last  = is_last ? 1 : 0;
                dut->m_axi_r_resp  = next_r_resp_;
                r_valid_           = true;
                pending_beat_++;
                next_r_resp_ = 0;
                next_r_last_override_ = -1;
                if (is_last)
                    pending_valid_ = false;
            }
        }

        // ----------------------------------------------------------------
        // AW channel: accept write address
        // ----------------------------------------------------------------
        dut->m_axi_aw_ready = 1;
        if (dut->m_axi_aw_valid && !aw_pending_) {
            aw_addr_    = (uint64_t)dut->m_axi_aw_addr;
            aw_len_     = dut->m_axi_aw_len;
            w_beat_cnt_ = 0;
            aw_pending_ = true;
        }

        // ----------------------------------------------------------------
        // W channel: accept write data beats
        // ----------------------------------------------------------------
        if (aw_pending_) {
            dut->m_axi_w_ready = 1;
            if (dut->m_axi_w_valid) {
                // Write 16 bytes to memory
                uint64_t beat_addr = aw_addr_ + (uint64_t)w_beat_cnt_ * 16;
                uint32_t* wdata = dut->m_axi_w_data.data();
                for (int b = 0; b < 16 && beat_addr+b < mem_.size(); b++)
                    mem_[beat_addr+b] = (uint8_t)((wdata[b/4] >> ((b%4)*8)) & 0xFF);
                bool last = dut->m_axi_w_last || (w_beat_cnt_ >= (int)aw_len_);
                w_beat_cnt_++;
                if (last) {
                    aw_pending_ = false;
                    b_pending_  = true;
                    // Keep w_ready=1 this cycle so DUT sees it on posedge;
                    // the else branch below will clear it next cycle.
                }
            }
        } else {
            dut->m_axi_w_ready = 0;
        }

        // ----------------------------------------------------------------
        // B channel: send write response
        // ----------------------------------------------------------------
        if (b_pending_) {
            dut->m_axi_b_valid = 1;
            dut->m_axi_b_resp  = 0;
            if (dut->m_axi_b_ready) {
                b_pending_ = false;
                // Keep b_valid=1 this cycle so DUT sees it on posedge;
                // the else branch below will clear it next cycle.
            }
        } else {
            dut->m_axi_b_valid = 0;
            dut->m_axi_b_resp  = 0;
        }
    }

private:
    std::vector<uint8_t> mem_;
    // Read state
    bool     pending_valid_;
    uint64_t pending_addr_;
    int      pending_ar_len_;
    int      pending_beat_;
    int      pending_timer_;
    bool     r_valid_;
    // Write state
    bool     aw_pending_;
    uint64_t aw_addr_;
    int      aw_len_;
    int      w_beat_cnt_;
    bool     b_pending_;
    int      next_r_resp_;
    int      next_r_last_override_;
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
