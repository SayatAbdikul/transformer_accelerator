// Verilator tests for the Stage D SFU engine.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/testbench.h"

#include <algorithm>
#include <cmath>
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

using tbutil::SimHarness;
using tbutil::sram_write_row;
using tbutil::sram_read_row;
using tbutil::sram_write_bytes;
using tbutil::sram_read_bytes;
using tbutil::pack_i32_le;
using tbutil::pack_u16_le;
constexpr int BUF_ABUF_ID  = tbutil::BUF_ABUF_ID;
constexpr int BUF_WBUF_ID  = tbutil::BUF_WBUF_ID;
constexpr int BUF_ACCUM_ID = tbutil::BUF_ACCUM_ID;

static double fp16_to_double(uint16_t bits) {
    bool sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1F;
    int frac = bits & 0x3FF;
    double sign_v = sign ? -1.0 : 1.0;
    if (exp == 0 && frac == 0)
        return 0.0;
    if (exp == 0)
        return sign_v * (double(frac) / 1024.0) * std::ldexp(1.0, -14);
    if (exp == 31)
        return sign_v * 65504.0;
    return sign_v * (1.0 + double(frac) / 1024.0) * std::ldexp(1.0, exp - 15);
}

static int round_half_even(double x) {
    long long floor_i = static_cast<long long>(std::floor(x));
    double frac = x - static_cast<double>(floor_i);
    if (frac > 0.5)
        return int(floor_i + 1);
    if (frac < 0.5)
        return int(floor_i);
    if (floor_i & 1LL)
        return int(floor_i + 1);
    return int(floor_i);
}

static int8_t quantize_ref(double value, double out_scale) {
    if (out_scale == 0.0)
        return 0;
    int q = round_half_even(value / out_scale);
    if (q > 127) return int8_t(127);
    if (q < -128) return int8_t(-128);
    return int8_t(q);
}

static std::vector<int8_t> softmax_ref(const std::vector<int32_t>& src,
                                       int M, int N, double in_scale, double out_scale) {
    std::vector<int8_t> out(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r) {
        double row_max = double(src[size_t(r) * size_t(N)]) * in_scale;
        for (int c = 1; c < N; ++c) {
            double x = double(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            row_max = std::max(row_max, x);
        }

        double exp_sum = 0.0;
        for (int c = 0; c < N; ++c) {
            double x = double(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            exp_sum += std::exp(x - row_max);
        }

        for (int c = 0; c < N; ++c) {
            double x = double(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            out[size_t(r) * size_t(N) + size_t(c)] =
                quantize_ref(std::exp(x - row_max) / exp_sum, out_scale);
        }
    }
    return out;
}

static std::vector<int8_t> softmax_attnv_ref(const std::vector<int32_t>& qkt_i32,
                                             const std::vector<int8_t>& v_i8,
                                             int M, int K, int N,
                                             double qkt_scale, double v_scale, double out_scale) {
    std::vector<int8_t> out(size_t(M) * size_t(N), 0);
    for (int r = 0; r < M; ++r) {
        double row_max = double(qkt_i32[size_t(r) * size_t(K)]) * qkt_scale;
        for (int k = 1; k < K; ++k) {
            double q = double(qkt_i32[size_t(r) * size_t(K) + size_t(k)]) * qkt_scale;
            row_max = std::max(row_max, q);
        }

        std::vector<double> weights(K, 0.0);
        double exp_sum = 0.0;
        for (int k = 0; k < K; ++k) {
            double q = double(qkt_i32[size_t(r) * size_t(K) + size_t(k)]) * qkt_scale;
            weights[size_t(k)] = std::exp(q - row_max);
            exp_sum += weights[size_t(k)];
        }

        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                double prob = weights[size_t(k)] / exp_sum;
                double v = double(v_i8[size_t(k) * size_t(N) + size_t(n)]) * v_scale;
                acc += prob * v;
            }
            out[size_t(r) * size_t(N) + size_t(n)] = quantize_ref(acc, out_scale);
        }
    }
    return out;
}

static std::vector<int8_t> layernorm_ref(const std::vector<int8_t>& src,
                                         const std::vector<uint16_t>& gamma_bits,
                                         const std::vector<uint16_t>& beta_bits,
                                         int M, int N, double in_scale, double out_scale) {
    std::vector<double> gamma(N), beta(N);
    for (int i = 0; i < N; ++i) {
        gamma[i] = fp16_to_double(gamma_bits[size_t(i)]);
        beta[i] = fp16_to_double(beta_bits[size_t(i)]);
    }

    std::vector<int8_t> out(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r) {
        double sum = 0.0;
        for (int c = 0; c < N; ++c)
            sum += double(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
        double mean = sum / double(N);

        double var = 0.0;
        for (int c = 0; c < N; ++c) {
            double x = double(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            double d = x - mean;
            var += d * d;
        }
        var /= double(N);
        double denom = std::sqrt(var + 1.0e-6);

        for (int c = 0; c < N; ++c) {
            double x = double(src[size_t(r) * size_t(N) + size_t(c)]) * in_scale;
            double y = ((x - mean) / denom) * gamma[c] + beta[c];
            out[size_t(r) * size_t(N) + size_t(c)] = quantize_ref(y, out_scale);
        }
    }
    return out;
}

static std::vector<int8_t> gelu_ref_i8(const std::vector<int8_t>& src,
                                       double in_scale, double out_scale) {
    std::vector<int8_t> out(src.size());
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (size_t i = 0; i < src.size(); ++i) {
        double x = double(src[i]) * in_scale;
        double y = x * 0.5 * (1.0 + std::erf(x * inv_sqrt2));
        out[i] = quantize_ref(y, out_scale);
    }
    return out;
}

static std::vector<int8_t> gelu_ref_i32(const std::vector<int32_t>& src,
                                        double in_scale, double out_scale) {
    std::vector<int8_t> out(src.size());
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (size_t i = 0; i < src.size(); ++i) {
        double x = double(src[i]) * in_scale;
        double y = x * 0.5 * (1.0 + std::erf(x * inv_sqrt2));
        out[i] = quantize_ref(y, out_scale);
    }
    return out;
}

static void expect_equal_bytes(const char* name,
                               const std::vector<uint8_t>& got,
                               const std::vector<uint8_t>& exp) {
    if (got.size() != exp.size())
        TEST_FAIL(name, "length mismatch");
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != exp[i]) {
            std::fprintf(stderr, "%s mismatch at byte %zu: got=%d exp=%d\n",
                         name, i, int(got[i]), int(exp[i]));
            TEST_FAIL(name, "byte mismatch");
        }
    }
}

static void test_softmax_accum_large_row() {
    const char* name = "softmax_accum_large_row";
    constexpr int M = 16;
    constexpr int N = 208;
    constexpr int DST_OFF = 256;
    const double in_scale = fp16_to_double(0x3400);
    const double out_scale = fp16_to_double(0x3400);

    std::vector<int32_t> src(size_t(M) * size_t(N), 0);
    for (int r = 0; r < M; ++r)
        src[size_t(r) * size_t(N) + size_t(N - 1)] = 32;

    auto expected_i8 = softmax_ref(src, M, N, in_scale, out_scale);
    std::vector<uint8_t> expected(expected_i8.size());
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = uint8_t(expected_i8[i]);

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 13, 1),
        insn::SET_SCALE(2, 0x3400),
        insn::SET_SCALE(3, 0x3400),
        insn::SOFTMAX(BUF_ACCUM_ID, 0, BUF_ABUF_ID, DST_OFF, 2),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(200000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_layernorm_identity() {
    const char* name = "layernorm_identity";
    constexpr int M = 16;
    constexpr int N = 192;
    constexpr int DST_OFF = 512;
    const double in_scale = fp16_to_double(0x3400);
    const double out_scale = fp16_to_double(0x3400);

    std::vector<int8_t> src(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            src[size_t(r) * size_t(N) + size_t(c)] = int8_t(((r * 17 + c * 5) % 41) - 20);

    std::vector<uint16_t> gamma(N, 0x3C00);
    std::vector<uint16_t> beta(N, 0x0000);
    auto expected_i8 = layernorm_ref(src, gamma, beta, M, N, in_scale, out_scale);

    std::vector<uint8_t> src_bytes(src.size());
    std::vector<uint8_t> expected(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        src_bytes[i] = uint8_t(src[i]);
        expected[i] = uint8_t(expected_i8[i]);
    }

    std::vector<uint8_t> gb_bytes = pack_u16_le(gamma);
    auto beta_bytes = pack_u16_le(beta);
    gb_bytes.insert(gb_bytes.end(), beta_bytes.begin(), beta_bytes.end());

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src_bytes);
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, gb_bytes);
    s.load({
        insn::CONFIG_TILE(1, 12, 1),
        insn::SET_SCALE(3, 0x3400),
        insn::SET_SCALE(4, 0x3400),
        insn::LAYERNORM(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ABUF_ID, DST_OFF, 3),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_gelu_abuf_roundtrip() {
    const char* name = "gelu_abuf_roundtrip";
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int DST_OFF = 1024;
    const double in_scale = fp16_to_double(0x3400);
    const double out_scale = fp16_to_double(0x3400);

    std::vector<int8_t> src(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            src[size_t(r) * size_t(N) + size_t(c)] = int8_t((c - 8) + r);

    auto expected_i8 = gelu_ref_i8(src, in_scale, out_scale);
    std::vector<uint8_t> src_bytes(src.size()), expected(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        src_bytes[i] = uint8_t(src[i]);
        expected[i] = uint8_t(expected_i8[i]);
    }

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src_bytes);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(5, 0x3400),
        insn::SET_SCALE(6, 0x3400),
        insn::GELU(BUF_ABUF_ID, 0, BUF_WBUF_ID, DST_OFF, 5),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(100000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_gelu_accum_roundtrip() {
    const char* name = "gelu_accum_roundtrip";
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int DST_OFF = 1280;
    const double in_scale = fp16_to_double(0x3400);
    const double out_scale = fp16_to_double(0x3400);

    std::vector<int32_t> src(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            src[size_t(r) * size_t(N) + size_t(c)] = (c - 8) * 2 + r;

    auto expected_i8 = gelu_ref_i32(src, in_scale, out_scale);
    std::vector<uint8_t> expected(expected_i8.size());
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = uint8_t(expected_i8[i]);

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(src));
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(8, 0x3400),
        insn::SET_SCALE(9, 0x3400),
        insn::GELU(BUF_ACCUM_ID, 0, BUF_ABUF_ID, DST_OFF, 8),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(100000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_softmax_attnv_roundtrip() {
    const char* name = "softmax_attnv_roundtrip";
    constexpr int M = 16;
    constexpr int K = 16;
    constexpr int N = 16;
    constexpr int DST_OFF = 1536;
    constexpr uint16_t QKT_SCALE_BITS = 0x2C00;   // 0.0625
    constexpr uint16_t V_SCALE_BITS   = 0x3400;   // 0.25
    constexpr uint16_t OUT_SCALE_BITS = 0x3400;   // 0.25
    constexpr uint16_t TRACE_BITS     = 0x3000;   // trace-only
    const double qkt_scale = fp16_to_double(QKT_SCALE_BITS);
    const double v_scale = fp16_to_double(V_SCALE_BITS);
    const double out_scale = fp16_to_double(OUT_SCALE_BITS);

    std::vector<int32_t> qkt(size_t(M) * size_t(K));
    std::vector<int8_t> v(size_t(K) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k)
            qkt[size_t(r) * size_t(K) + size_t(k)] = ((r * 3 + k * 5) % 19) - 9;
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            v[size_t(k) * size_t(N) + size_t(n)] = int8_t(((k * 7 + n * 11) % 23) - 11);

    auto expected_i8 = softmax_attnv_ref(qkt, v, M, K, N, qkt_scale, v_scale, out_scale);
    std::vector<uint8_t> v_bytes(v.size()), expected(expected_i8.size());
    for (size_t i = 0; i < v.size(); ++i)
        v_bytes[i] = uint8_t(v[i]);
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = uint8_t(expected_i8[i]);

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(qkt));
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, v_bytes);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(4, QKT_SCALE_BITS),
        insn::SET_SCALE(5, V_SCALE_BITS),
        insn::SET_SCALE(6, OUT_SCALE_BITS),
        insn::SET_SCALE(7, TRACE_BITS),
        insn::SOFTMAX_ATTNV(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0, BUF_WBUF_ID, DST_OFF, 4),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, size_t(DST_OFF) * 16, expected.size());
    expect_equal_bytes(name, got, expected);
    TEST_PASS(name);
}

static void test_softmax_attnv_zero_out_scale() {
    const char* name = "softmax_attnv_zero_out_scale";
    constexpr int DST_OFF = 1664;
    SimHarness s;
    std::vector<int32_t> qkt(16 * 16, 0);
    std::vector<uint8_t> v(16 * 16, 0x7F);
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_i32_le(qkt));
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, v);
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SET_SCALE(8, 0x3400),
        insn::SET_SCALE(9, 0x3400),
        insn::SET_SCALE(10, 0x0000),
        insn::SET_SCALE(11, 0x3000),
        insn::SOFTMAX_ATTNV(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0, BUF_WBUF_ID, DST_OFF, 8),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(250000);
    if (s.dut->fault) TEST_FAIL(name, "unexpected fault");

    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, size_t(DST_OFF) * 16, 256);
    if (!std::all_of(got.begin(), got.end(), [](uint8_t vbyte) { return vbyte == 0; }))
        TEST_FAIL(name, "zero out-scale should zero destination");
    TEST_PASS(name);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    test_softmax_accum_large_row();
    test_layernorm_identity();
    test_gelu_abuf_roundtrip();
    test_gelu_accum_roundtrip();
    test_softmax_attnv_roundtrip();
    test_softmax_attnv_zero_out_scale();

    std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    if (tests_pass != tests_run) std::exit(1);
    return 0;
}
