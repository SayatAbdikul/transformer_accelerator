#include "Vtb_fp32_prim.h"
#include "verilated.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

enum Op : uint8_t {
    OP_ROUND = 0,
    OP_ADD = 1,
    OP_SUB = 2,
    OP_MUL = 3,
};

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(uint32_t bits) {
    float value = 0.0f;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool is_nan_bits(uint32_t bits) {
    return ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0u);
}

bool bits_match(uint32_t got, uint32_t expected) {
    if (is_nan_bits(expected)) {
        return is_nan_bits(got);
    }
    return got == expected;
}

uint32_t eval(Vtb_fp32_prim &tb, Op op, uint32_t a, uint32_t b = 0) {
    tb.op = op;
    tb.a_bits = a;
    tb.b_bits = b;
    tb.eval();
    return tb.result_bits;
}

bool expect_bits(Vtb_fp32_prim &tb, const std::string &name, Op op, uint32_t a, uint32_t b,
                 uint32_t expected) {
    const uint32_t got = eval(tb, op, a, b);
    if (!bits_match(got, expected)) {
        std::cerr << "[FAIL] " << name << " op=" << static_cast<int>(op)
                  << " a=0x" << std::hex << a << " b=0x" << b
                  << " got=0x" << got << " expected=0x" << expected << std::dec
                  << "\n";
        return false;
    }
    return true;
}

bool expect_real_shims(Vtb_fp32_prim &tb, const std::string &name, double a, double b) {
    tb.a_real = a;
    tb.b_real = b;
    tb.eval();

    const uint32_t a_round = float_bits(static_cast<float>(a));
    const uint32_t add = float_bits(static_cast<float>(static_cast<float>(a) + static_cast<float>(b)));
    const uint32_t sub = float_bits(static_cast<float>(static_cast<float>(a) - static_cast<float>(b)));
    const uint32_t mul = float_bits(static_cast<float>(static_cast<float>(a) * static_cast<float>(b)));

    bool ok = true;
    if (!bits_match(tb.real_round_bits, a_round)) {
        std::cerr << "[FAIL] " << name << "_real_round got=0x" << std::hex
                  << tb.real_round_bits << " expected=0x" << a_round << std::dec << "\n";
        ok = false;
    }
    if (!bits_match(tb.real_add_bits, add)) {
        std::cerr << "[FAIL] " << name << "_real_add got=0x" << std::hex
                  << tb.real_add_bits << " expected=0x" << add << std::dec << "\n";
        ok = false;
    }
    if (!bits_match(tb.real_sub_bits, sub)) {
        std::cerr << "[FAIL] " << name << "_real_sub got=0x" << std::hex
                  << tb.real_sub_bits << " expected=0x" << sub << std::dec << "\n";
        ok = false;
    }
    if (!bits_match(tb.real_mul_bits, mul)) {
        std::cerr << "[FAIL] " << name << "_real_mul got=0x" << std::hex
                  << tb.real_mul_bits << " expected=0x" << mul << std::dec << "\n";
        ok = false;
    }
    return ok;
}

uint32_t add_ref(uint32_t a, uint32_t b) {
    return float_bits(bits_float(a) + bits_float(b));
}

uint32_t sub_ref(uint32_t a, uint32_t b) {
    return float_bits(bits_float(a) - bits_float(b));
}

uint32_t mul_ref(uint32_t a, uint32_t b) {
    return float_bits(bits_float(a) * bits_float(b));
}

} // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vtb_fp32_prim tb;

    int failures = 0;
    auto check = [&](bool ok) {
        if (!ok) {
            failures++;
        }
    };

    const std::vector<uint32_t> roundtrip_values = {
        float_bits(0.0f),
        float_bits(-0.0f),
        float_bits(1.0f),
        float_bits(-1.0f),
        float_bits(0.15625f),
        float_bits(-13.75f),
        float_bits(65504.0f),
        0x00000001u,
        0x007fffffu,
        0x00800000u,
        0x7f7fffffu,
        0x7f800000u,
        0xff800000u,
    };
    for (uint32_t value : roundtrip_values) {
        check(expect_bits(tb, "roundtrip", OP_ROUND, value, 0, value));
    }

    struct BinaryCase {
        const char *name;
        uint32_t a;
        uint32_t b;
    };

    const std::vector<BinaryCase> cases = {
        {"simple", float_bits(1.0f), float_bits(2.0f)},
        {"negative", float_bits(-1.5f), float_bits(0.25f)},
        {"cancel", float_bits(1.0e20f), float_bits(-1.0e20f)},
        {"tie-ish", float_bits(1.0f), 0x33800000u},
        {"subnormal", 0x00000001u, float_bits(2.0f)},
        {"normal_min", 0x00800000u, float_bits(0.5f)},
        {"signed_zero", float_bits(-0.0f), float_bits(3.0f)},
        {"overflow", float_bits(3.4e38f), float_bits(2.0f)},
        {"inf", 0x7f800000u, float_bits(1.0f)},
    };

    for (const auto &tc : cases) {
        check(expect_bits(tb, std::string(tc.name) + "_add", OP_ADD, tc.a, tc.b, add_ref(tc.a, tc.b)));
        check(expect_bits(tb, std::string(tc.name) + "_sub", OP_SUB, tc.a, tc.b, sub_ref(tc.a, tc.b)));
        check(expect_bits(tb, std::string(tc.name) + "_mul", OP_MUL, tc.a, tc.b, mul_ref(tc.a, tc.b)));
    }

    uint32_t rng = 0x12345678u;
    for (int i = 0; i < 256; ++i) {
        rng = 1664525u * rng + 1013904223u;
        uint32_t a = rng & 0x7effffffu; // finite, non-NaN stress values
        rng = 1664525u * rng + 1013904223u;
        uint32_t b = rng & 0x7effffffu;
        check(expect_bits(tb, "random_add", OP_ADD, a, b, add_ref(a, b)));
        check(expect_bits(tb, "random_sub", OP_SUB, a, b, sub_ref(a, b)));
        check(expect_bits(tb, "random_mul", OP_MUL, a, b, mul_ref(a, b)));
    }

    check(expect_bits(tb, "inf_minus_inf", OP_SUB, 0x7f800000u, 0x7f800000u,
                      float_bits(std::numeric_limits<float>::infinity() -
                                 std::numeric_limits<float>::infinity())));
    check(expect_bits(tb, "zero_times_inf", OP_MUL, float_bits(0.0f), 0x7f800000u,
                      float_bits(0.0f * std::numeric_limits<float>::infinity())));

    const std::vector<std::pair<double, double>> real_cases = {
        {0.0, -0.0},
        {1.0e-6, 1.0},
        {192.0, 1.0 / 192.0},
        {-128.0, 0.082244873046875},
        {127.0, 0.0363311767578125},
        {65504.0, -13.25},
        {2147483647.0, 0.0009765625},
        {-2147483648.0, -0.000244140625},
        {3.4028234663852886e38, 2.0},
        {1.4012984643248171e-45, 2.0},
    };
    for (size_t i = 0; i < real_cases.size(); ++i) {
        check(expect_real_shims(tb, "real_case_" + std::to_string(i),
                                real_cases[i].first, real_cases[i].second));
    }

    double da = 0.123456789;
    double db = -97.25;
    for (int i = 0; i < 512; ++i) {
        da = std::sin(da * 13.0 + static_cast<double>(i)) * std::exp(static_cast<double>(i % 63) - 31.0);
        db = std::cos(db * 7.0 - static_cast<double>(i)) * std::exp(static_cast<double>((i * 3) % 63) - 31.0);
        check(expect_real_shims(tb, "real_random_" + std::to_string(i), da, db));
    }

    if (failures != 0) {
        std::cerr << failures << " FP32 primitive checks failed\n";
        return 1;
    }

    std::cout << "FP32 primitive tests passed\n";
    return 0;
}
