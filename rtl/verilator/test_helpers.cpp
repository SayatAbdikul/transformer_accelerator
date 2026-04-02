// Verilator helper-engine tests for Phase C.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/testbench.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

static int tests_run  = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
    printf("PASS: %s\n", name); tests_pass++; tests_run++; } while(0)
#define TEST_FAIL(name, msg) do { \
    fprintf(stderr, "FAIL: %s — %s\n", name, msg); std::exit(1); } while(0)

struct Sim {
    std::unique_ptr<Vtaccel_top> dut;
    AXI4SlaveModel dram;

    explicit Sim() : dut(std::make_unique<Vtaccel_top>()), dram(16 * 1024 * 1024) {
        do_reset(dut.get());
    }

    void load(const std::vector<uint64_t>& prog) {
        dram.write_program(prog);
    }

    void run(int timeout = 200000) {
        dut->start = 1;
        tick(dut.get(), dram);
        dut->start = 0;
        run_until_halt(dut.get(), dram, timeout);
    }
};

enum BufId : int {
    BUF_ABUF_ID  = 0,
    BUF_WBUF_ID  = 1,
    BUF_ACCUM_ID = 2,
};

static VlWide<4>* row_ptr(Vtaccel_top* dut, int buf_id, int row) {
    auto* r = dut->rootp;
    switch (buf_id) {
        case BUF_ABUF_ID:  return &r->taccel_top__DOT__u_sram__DOT__u_abuf__DOT__mem[row];
        case BUF_WBUF_ID:  return &r->taccel_top__DOT__u_sram__DOT__u_wbuf__DOT__mem[row];
        case BUF_ACCUM_ID: return &r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row];
        default: std::abort();
    }
}

static const VlWide<4>* row_ptr_const(Vtaccel_top* dut, int buf_id, int row) {
    return row_ptr(dut, buf_id, row);
}

static void sram_write_row(Vtaccel_top* dut, int buf_id, int row, const uint8_t data[16]) {
    VlWide<4>* mem = row_ptr(dut, buf_id, row);
    for (int w = 0; w < 4; ++w) {
        (*mem)[w] = (uint32_t(data[w * 4 + 0])      ) |
                    (uint32_t(data[w * 4 + 1]) <<  8) |
                    (uint32_t(data[w * 4 + 2]) << 16) |
                    (uint32_t(data[w * 4 + 3]) << 24);
    }
}

static void sram_read_row(Vtaccel_top* dut, int buf_id, int row, uint8_t out[16]) {
    const VlWide<4>* mem = row_ptr_const(dut, buf_id, row);
    for (int w = 0; w < 4; ++w) {
        uint32_t word = (*mem)[w];
        out[w * 4 + 0] = uint8_t((word >> 0) & 0xFF);
        out[w * 4 + 1] = uint8_t((word >> 8) & 0xFF);
        out[w * 4 + 2] = uint8_t((word >> 16) & 0xFF);
        out[w * 4 + 3] = uint8_t((word >> 24) & 0xFF);
    }
}

static void sram_write_bytes(Vtaccel_top* dut, int buf_id, size_t byte_off,
                             const std::vector<uint8_t>& data) {
    size_t pos = 0;
    while (pos < data.size()) {
        int row = int((byte_off + pos) / 16);
        int lane = int((byte_off + pos) % 16);
        int take = int(std::min<size_t>(16 - lane, data.size() - pos));
        uint8_t tmp[16];
        sram_read_row(dut, buf_id, row, tmp);
        for (int i = 0; i < take; ++i)
            tmp[lane + i] = data[pos + size_t(i)];
        sram_write_row(dut, buf_id, row, tmp);
        pos += size_t(take);
    }
}

static std::vector<uint8_t> sram_read_bytes(Vtaccel_top* dut, int buf_id, size_t byte_off,
                                            size_t len) {
    std::vector<uint8_t> out(len);
    size_t pos = 0;
    while (pos < len) {
        int row = int((byte_off + pos) / 16);
        int lane = int((byte_off + pos) % 16);
        int take = int(std::min<size_t>(16 - lane, len - pos));
        uint8_t tmp[16];
        sram_read_row(dut, buf_id, row, tmp);
        for (int i = 0; i < take; ++i)
            out[pos + size_t(i)] = tmp[lane + i];
        pos += size_t(take);
    }
    return out;
}

static std::vector<uint8_t> pack_i32_le(const std::vector<int32_t>& vals) {
    std::vector<uint8_t> out(vals.size() * 4);
    for (size_t i = 0; i < vals.size(); ++i) {
        uint32_t word = uint32_t(vals[i]);
        out[i * 4 + 0] = uint8_t((word >> 0) & 0xFF);
        out[i * 4 + 1] = uint8_t((word >> 8) & 0xFF);
        out[i * 4 + 2] = uint8_t((word >> 16) & 0xFF);
        out[i * 4 + 3] = uint8_t((word >> 24) & 0xFF);
    }
    return out;
}

static std::vector<int32_t> unpack_i32_le(const std::vector<uint8_t>& bytes) {
    std::vector<int32_t> out(bytes.size() / 4);
    for (size_t i = 0; i < out.size(); ++i) {
        uint32_t word = uint32_t(bytes[i * 4 + 0]) |
                        (uint32_t(bytes[i * 4 + 1]) << 8) |
                        (uint32_t(bytes[i * 4 + 2]) << 16) |
                        (uint32_t(bytes[i * 4 + 3]) << 24);
        out[i] = int32_t(word);
    }
    return out;
}

static int8_t sat_add_ref(int8_t a, int8_t b) {
    int sum = int(a) + int(b);
    if (sum > 127) return int8_t(127);
    if (sum < -128) return int8_t(-128);
    return int8_t(sum);
}

static int64_t fp16_mul_round_even_ref(int32_t src, uint16_t fp16_val) {
    bool sign = (fp16_val >> 15) & 1;
    uint32_t exp = (fp16_val >> 10) & 0x1F;
    uint32_t frac = fp16_val & 0x3FF;
    if (exp == 0 && frac == 0)
        return 0;

    int32_t shift = 0;
    int64_t mant = 0;
    if (exp == 0) {
        mant = int64_t(frac);
        shift = -24;
    } else {
        mant = int64_t(1024 + frac);
        shift = int32_t(exp) - 25;
    }

    int64_t prod = int64_t(src) * mant;
    if (sign)
        prod = -prod;

    if (shift >= 0)
        return prod << shift;

    int rshift = -shift;
    int64_t abs_prod = prod < 0 ? -prod : prod;
    int64_t q = abs_prod >> rshift;
    int64_t rem = abs_prod & ((int64_t(1) << rshift) - 1);
    int64_t half = int64_t(1) << (rshift - 1);
    if (rem > half || (rem == half && (q & 1)))
        q += 1;
    return prod < 0 ? -q : q;
}

static int8_t requant_ref(int32_t src, uint16_t scale) {
    int64_t scaled = fp16_mul_round_even_ref(src, scale);
    if (scaled > 127) return int8_t(127);
    if (scaled < -128) return int8_t(-128);
    return int8_t(scaled);
}

static void expect_fault_program(const char* name,
                                 const std::vector<uint64_t>& prog,
                                 uint32_t expected_fault_code,
                                 int timeout = 5000) {
    Sim s;
    s.load(prog);
    s.run(timeout);
    EXPECT(s.dut->fault == 1, "fault should assert");
    EXPECT(s.dut->done == 0, "done should remain low");
    EXPECT(s.dut->fault_code == expected_fault_code, "unexpected fault code");
    TEST_PASS(name);
}

static void write_abuf_row_i8(Vtaccel_top* dut, int row, const int8_t vals[16]) {
    uint8_t raw[16];
    for (int i = 0; i < 16; ++i) raw[i] = uint8_t(vals[i]);
    sram_write_row(dut, BUF_ABUF_ID, row, raw);
}

static void write_wbuf_row_i8(Vtaccel_top* dut, int row, const int8_t vals[16]) {
    uint8_t raw[16];
    for (int i = 0; i < 16; ++i) raw[i] = uint8_t(vals[i]);
    sram_write_row(dut, BUF_WBUF_ID, row, raw);
}

static int32_t read_accum_ij(Vtaccel_top* dut, int dst_off, int i, int j) {
    auto* r = dut->rootp;
    int grp = j / 4;
    int lane = j % 4;
    int row = dst_off + i * 4 + grp;
    uint32_t word = r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
    return int32_t(word);
}

static void load_matrix_abuf_for_systolic(Vtaccel_top* dut, int off, const int8_t m[16][16]) {
    int8_t t[16][16] = {};
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            t[r][c] = m[c][r];
    for (int r = 0; r < 16; ++r)
        write_abuf_row_i8(dut, off + r, t[r]);
}

static void load_matrix_wbuf(Vtaccel_top* dut, int off, const int8_t m[16][16]) {
    for (int r = 0; r < 16; ++r)
        write_wbuf_row_i8(dut, off + r, m[r]);
}

static void matmul_ref(const int8_t a[16][16], const int8_t b[16][16], int32_t c[16][16]) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < 16; ++k)
                acc += int32_t(a[i][k]) * int32_t(b[k][j]);
            c[i][j] = acc;
        }
    }
}

static void test_buf_copy_flat_interbuffer() {
    const char* name = "buf_copy_flat_interbuffer";
    Sim s;
    std::vector<uint8_t> src(48);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = uint8_t((0x20 + 7 * i) & 0xFF);
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 10, 3, 0, 0),
        insn::HALT(),
    });
    s.run();

    EXPECT(s.dut->done == 1 && s.dut->fault == 0, "copy should halt cleanly");
    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 10 * 16, src.size());
    if (got != src) TEST_FAIL(name, "flat copy mismatch");
    TEST_PASS(name);
}

static void test_buf_copy_overlap_compaction() {
    const char* name = "buf_copy_overlap_compaction";
    Sim s;
    std::vector<uint8_t> bytes(6 * 16);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = uint8_t((0x51 + 11 * i) & 0xFF);
    auto expected = bytes;
    std::memmove(expected.data() + 16, expected.data() + 32, 48);
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, bytes);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 2, BUF_ABUF_ID, 1, 3, 0, 0),
        insn::HALT(),
    });
    s.run();

    EXPECT(s.dut->done == 1 && s.dut->fault == 0, "overlap copy should halt cleanly");
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 0, bytes.size());
    if (got != expected) TEST_FAIL(name, "overlap compaction mismatch");
    TEST_PASS(name);
}

static void test_buf_copy_zero_length() {
    const char* name = "buf_copy_zero_length";
    Sim s;
    std::vector<uint8_t> before(64);
    for (size_t i = 0; i < before.size(); ++i)
        before[i] = uint8_t((0x91 + 13 * i) & 0xFF);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 8 * 16, before);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 8, 0, 0, 0),
        insn::HALT(),
    });
    s.run();

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 8 * 16, before.size());
    if (got != before) TEST_FAIL(name, "zero-length copy modified memory");
    TEST_PASS(name);
}

static void test_buf_copy_transpose_unaligned_source() {
    const char* name = "buf_copy_transpose_unaligned_source";
    Sim s;
    constexpr int rows = 16;
    constexpr int cols = 18;
    std::vector<uint8_t> src(rows * cols);
    std::vector<uint8_t> expected(cols * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            src[r * cols + c] = uint8_t((r * 19 + c * 7 + 3) & 0xFF);
            expected[c * rows + r] = src[r * cols + c];
        }

    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src);
    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, 18, 1, 1),
        insn::HALT(),
    });
    s.run(100000);

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 0, expected.size());
    if (got != expected) TEST_FAIL(name, "transpose mismatch");
    TEST_PASS(name);
}

static void test_buf_copy_transpose_same_buffer_fault() {
    expect_fault_program("buf_copy_transpose_same_buffer_fault",
                         { insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_ABUF_ID, 32, 16, 1, 1) }, 6, 5000);
}

static void test_vadd_int8_saturating() {
    const char* name = "vadd_int8_saturating";
    Sim s;
    std::vector<uint8_t> src_a(256), src_b(256), expected(256);
    for (int i = 0; i < 256; ++i) {
        int8_t a = (i % 5 == 0) ? int8_t(120) :
                   (i % 7 == 0) ? int8_t(-120) : int8_t((i % 17) - 8);
        int8_t b = (i % 5 == 0) ? int8_t(30) :
                   (i % 7 == 0) ? int8_t(-30) : int8_t((i % 11) - 5);
        src_a[i] = uint8_t(a);
        src_b[i] = uint8_t(b);
        expected[i] = uint8_t(sat_add_ref(a, b));
    }

    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src_a);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, src_b);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::VADD(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ABUF_ID, 32, 0),
        insn::HALT(),
    });
    s.run(100000);

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 32 * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "INT8 VADD mismatch");
    TEST_PASS(name);
}

static void test_vadd_bias_int32_wrap() {
    const char* name = "vadd_bias_int32_wrap";
    Sim s;
    std::vector<int32_t> accum(16 * 16), bias(16), expected(16 * 16);
    for (int i = 0; i < 16 * 16; ++i)
        accum[i] = (i % 9 == 0) ? int32_t(0x7FFFFFF0u) :
                   (i % 13 == 0) ? int32_t(0x80000010u) :
                   int32_t((i * 97) - 3000);
    for (int j = 0; j < 16; ++j)
        bias[j] = (j % 4 == 0) ? int32_t(0x30u) :
                  (j % 5 == 0) ? int32_t(0xFFFFFFE0u) :
                  int32_t(j * 11 - 40);

    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            expected[r * 16 + c] = int32_t(uint32_t(accum[r * 16 + c]) + uint32_t(bias[c]));

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(accum));
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 200 * 16, pack_i32_le(bias));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, 200, BUF_ACCUM_ID, 128, 0),
        insn::HALT(),
    });
    s.run(120000);

    auto got = unpack_i32_le(sram_read_bytes(s.dut.get(), BUF_ACCUM_ID, 128 * 16, expected.size() * 4ULL));
    if (got != expected) TEST_FAIL(name, "bias VADD mismatch");
    TEST_PASS(name);
}

static void test_requant_rounding_and_clipping() {
    const char* name = "requant_rounding_and_clipping";
    Sim s;
    constexpr uint16_t SCALE_HALF = 0x3800; // 0.5
    const int32_t pattern[16] = {
        1, 3, 5, -1, -3, -5, 255, 257,
        -255, -257, 300, -300, 0, 2, -2, 7
    };
    std::vector<int32_t> src(16 * 16);
    std::vector<uint8_t> expected(16 * 16);
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            int32_t v = pattern[c] + r;
            src[r * 16 + c] = v;
            expected[r * 16 + c] = uint8_t(requant_ref(v, SCALE_HALF));
        }
    }

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(0, SCALE_HALF),
        insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 64, 0),
        insn::HALT(),
    });
    s.run(120000);

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 64 * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "requant rounding mismatch");
    TEST_PASS(name);
}

static void test_requant_subnormal_negative_zero() {
    const char* name = "requant_subnormal_negative_zero";
    Sim s;
    constexpr uint16_t SCALE_SUBN = 0x0001;
    constexpr uint16_t SCALE_NEG1 = 0xBC00;
    std::vector<int32_t> src(16 * 16, 0);
    std::vector<uint8_t> expected_subn(16 * 16, 0);
    std::vector<uint8_t> expected_neg(16, 0);

    const int32_t row0[16] = {
        1 << 23, 3 << 23, -(1 << 23), -(3 << 23),
        1, -1, 0, 4 << 23,
        -(4 << 23), 5 << 23, -(5 << 23), 127,
        -127, 128, -128, 1000
    };
    for (int c = 0; c < 16; ++c) {
        src[c] = row0[c];
        expected_subn[c] = uint8_t(requant_ref(row0[c], SCALE_SUBN));
    }
    for (int c = 0; c < 16; ++c) {
        src[16 + c] = c - 8;
        expected_subn[16 + c] = uint8_t(requant_ref(src[16 + c], SCALE_SUBN));
        expected_neg[c] = uint8_t(requant_ref(src[16 + c], SCALE_NEG1));
    }

    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(0, SCALE_SUBN),
        insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 128, 0),
        insn::SET_SCALE(0, SCALE_NEG1),
        insn::REQUANT(BUF_ACCUM_ID, 4, BUF_WBUF_ID, 256, 0),
        insn::SET_SCALE(0, 0x0000),
        insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 160, 0),
        insn::HALT(),
    });
    s.run(200000);

    auto got_subn = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 128 * 16, 256);
    if (!std::equal(expected_subn.begin(), expected_subn.end(), got_subn.begin())) {
        for (size_t i = 0; i < expected_subn.size(); ++i) {
            if (expected_subn[i] != got_subn[i]) {
                fprintf(stderr, "subnormal mismatch idx=%zu got=%d exp=%d\n",
                        i, int(got_subn[i]), int(expected_subn[i]));
                break;
            }
        }
        TEST_FAIL(name, "subnormal requant mismatch");
    }
    auto got_neg = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 256 * 16, 256);
    if (!std::equal(expected_neg.begin(), expected_neg.end(), got_neg.begin())) {
        for (size_t i = 0; i < 16; ++i) {
            if (expected_neg[i] != got_neg[i]) {
                fprintf(stderr, "negative mismatch idx=%zu got=%d exp=%d\n",
                        i, int(got_neg[i]), int(expected_neg[i]));
                break;
            }
        }
        TEST_FAIL(name, "negative requant mismatch");
    }
    auto got_zero = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 160 * 16, 256);
    if (!std::all_of(got_zero.begin(), got_zero.end(), [](uint8_t v) { return v == 0; }))
        TEST_FAIL(name, "zero-scale requant mismatch");
    TEST_PASS(name);
}

static void test_helper_oob_faults() {
    expect_fault_program("helper_buf_copy_oob_fault",
                         { insn::BUF_COPY(BUF_ABUF_ID, 8191, BUF_WBUF_ID, 0, 2, 0, 0) }, 3, 5000);
    expect_fault_program("helper_vadd_bad_mode_fault",
                         { insn::CONFIG_TILE(1, 1, 1),
                           insn::VADD(BUF_WBUF_ID, 0, BUF_WBUF_ID, 0, BUF_ABUF_ID, 0, 0) }, 6, 5000);
    expect_fault_program("helper_requant_bad_mode_fault",
                         { insn::CONFIG_TILE(1, 1, 1),
                           insn::REQUANT(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, 0) }, 6, 5000);
}

static void test_matmul_then_requant() {
    const char* name = "matmul_then_requant";
    Sim s;
    int8_t a[16][16] = {};
    int8_t eye[16][16] = {};
    int32_t exp_acc[16][16] = {};
    std::vector<uint8_t> expected(256);

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            a[i][j] = int8_t((i * 5 + j) - 20);
            eye[i][j] = (i == j) ? 1 : 0;
        }
    }
    matmul_ref(a, eye, exp_acc);
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            expected[i * 16 + j] = uint8_t(requant_ref(exp_acc[i][j], 0x3C00));

    load_matrix_abuf_for_systolic(s.dut.get(), 128, a);
    load_matrix_wbuf(s.dut.get(), 0, eye);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::MATMUL(BUF_ABUF_ID, 128, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0),
        insn::SYNC(0b010),
        insn::SET_SCALE(0, 0x3C00),
        insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 256, 0),
        insn::HALT(),
    });
    s.run(200000);

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 256 * 16, expected.size());
    if (got != expected) TEST_FAIL(name, "matmul->requant mismatch");
    TEST_PASS(name);
}

static void test_transpose_then_matmul() {
    const char* name = "transpose_then_matmul";
    Sim s;
    int8_t a[16][16] = {};
    int8_t k[16][16] = {};
    int8_t kt[16][16] = {};
    int32_t exp[16][16] = {};
    std::vector<uint8_t> flat_k(256);

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            a[i][j] = int8_t(((i * 3 + j * 5) % 11) - 5);
            k[i][j] = int8_t(((i * 7 + j * 2 + 1) % 13) - 6);
            kt[j][i] = k[i][j];
            flat_k[i * 16 + j] = uint8_t(k[i][j]);
        }
    }

    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, flat_k);
    load_matrix_abuf_for_systolic(s.dut.get(), 128, a);
    matmul_ref(a, kt, exp);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, 16, 1, 1),
        insn::CONFIG_TILE(1, 1, 1),
        insn::MATMUL(BUF_ABUF_ID, 128, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0),
        insn::SYNC(0b010),
        insn::HALT(),
    });
    s.run(200000);

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            int32_t got = read_accum_ij(s.dut.get(), 0, i, j);
            if (got != exp[i][j]) {
                fprintf(stderr, "transpose->matmul mismatch i=%d j=%d got=%d exp=%d\n",
                        i, j, got, exp[i][j]);
                TEST_FAIL(name, "transpose-fed MATMUL mismatch");
            }
        }
    }
    TEST_PASS(name);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    test_buf_copy_flat_interbuffer();
    test_buf_copy_overlap_compaction();
    test_buf_copy_zero_length();
    test_buf_copy_transpose_unaligned_source();
    test_buf_copy_transpose_same_buffer_fault();
    test_vadd_int8_saturating();
    test_vadd_bias_int32_wrap();
    test_requant_rounding_and_clipping();
    test_requant_subnormal_negative_zero();
    test_helper_oob_faults();
    test_matmul_then_requant();
    test_transpose_then_matmul();

    printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    if (tests_pass != tests_run) std::exit(1);
    return 0;
}
