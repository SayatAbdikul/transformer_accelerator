// Focused native regressions for the QK^T debug split.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/testbench.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using tbutil::SimHarness;
using tbutil::sram_read_bytes;
using tbutil::sram_write_bytes;
constexpr int BUF_ABUF_ID = tbutil::BUF_ABUF_ID;
constexpr int BUF_WBUF_ID = tbutil::BUF_WBUF_ID;
constexpr int BUF_ACCUM_ID = tbutil::BUF_ACCUM_ID;

static int tests_run = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
  std::printf("PASS: %s\n", name); tests_pass++; tests_run++; \
} while (0)

#define TEST_FAIL(name, msg) do { \
  std::fprintf(stderr, "FAIL: %s - %s\n", name, msg); std::exit(1); \
} while (0)

namespace {

using QStrip16x64 = std::array<std::array<int8_t, 64>, 16>;
using Key208x64 = std::array<std::array<int8_t, 64>, 208>;
using KeyT64x208 = std::array<std::array<int8_t, 208>, 64>;

std::vector<uint8_t> flatten_qstrip_row_major(const QStrip16x64& q) {
  std::vector<uint8_t> out(16 * 64);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 64; ++c)
      out[size_t(r) * 64 + size_t(c)] = static_cast<uint8_t>(q[r][c]);
  }
  return out;
}

std::vector<uint8_t> flatten_key_row_major(const Key208x64& k) {
  std::vector<uint8_t> out(208 * 64);
  for (int r = 0; r < 208; ++r) {
    for (int c = 0; c < 64; ++c)
      out[size_t(r) * 64 + size_t(c)] = static_cast<uint8_t>(k[r][c]);
  }
  return out;
}

std::vector<uint8_t> flatten_keyt_row_major(const KeyT64x208& kt) {
  std::vector<uint8_t> out(64 * 208);
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 208; ++c)
      out[size_t(r) * 208 + size_t(c)] = static_cast<uint8_t>(kt[r][c]);
  }
  return out;
}

int32_t read_accum_wide(Vtaccel_top* dut, int dst_off, int row_idx, int col_idx, int cols) {
  auto* root = dut->rootp;
  const int words_per_row = cols / 4;
  const int grp = col_idx / 4;
  const int lane = col_idx % 4;
  const int row = dst_off + row_idx * words_per_row + grp;
  uint32_t word = root->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
  return static_cast<int32_t>(word);
}

void matmul_ref_16x64x208(const QStrip16x64& q, const KeyT64x208& kt, int32_t (&acc)[16][208]) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 208; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < 64; ++k)
        sum += int32_t(q[i][k]) * int32_t(kt[k][j]);
      acc[i][j] = sum;
    }
  }
}

void expect_clean_halt(const char* name, Vtaccel_top* dut) {
  if (!dut->done || dut->fault)
    TEST_FAIL(name, "did not halt cleanly");
}

void test_qkt_key_transpose_208x64() {
  const char* name = "qkt_key_transpose_208x64";
  SimHarness s;
  Key208x64 key{};
  KeyT64x208 expected{};

  for (int r = 0; r < 208; ++r) {
    for (int c = 0; c < 64; ++c) {
      int8_t v = (r >= 197) ? int8_t(0) : int8_t(((r * 9 + c * 5 + 3) % 31) - 15);
      key[r][c] = v;
      expected[c][r] = v;
    }
  }

  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, flatten_key_row_major(key));
  s.load({
      insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, (208 * 64) / 16, 13, 1),
      insn::HALT(),
  });
  s.run(200000);
  expect_clean_halt(name, s.dut.get());

  auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 0, 64 * 208);
  auto exp = flatten_keyt_row_major(expected);
  if (got != exp)
    TEST_FAIL(name, "transposed K mismatch");
  TEST_PASS(name);
}

void test_qkt_matmul_pretransposed_16x64x208() {
  const char* name = "qkt_matmul_pretransposed_16x64x208";
  SimHarness s;
  QStrip16x64 query{};
  KeyT64x208 key_t{};
  int32_t expected[16][208] = {};

  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 64; ++c)
      query[r][c] = int8_t(((r * 11 + c * 7 + 1) % 27) - 13);
  }
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 208; ++c)
      key_t[r][c] = int8_t(((r * 5 + c * 3 + 2) % 29) - 14);
  }

  matmul_ref_16x64x208(query, key_t, expected);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, flatten_qstrip_row_major(query));
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, flatten_keyt_row_major(key_t));
  s.load({
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });
  s.run(1500000);
  expect_clean_halt(name, s.dut.get());

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 208; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      if (got != expected[i][j]) {
        std::fprintf(stderr,
                     "wide QK^T mismatch row=%d col=%d got=%d exp=%d\n",
                     i, j, got, expected[i][j]);
        TEST_FAIL(name, "pre-transposed QK^T MATMUL mismatch");
      }
    }
  }
  TEST_PASS(name);
}

void test_qkt_matmul_pretransposed_nonzero_qoff_16x64x208() {
  const char* name = "qkt_matmul_pretransposed_nonzero_qoff_16x64x208";
  SimHarness s;
  constexpr int Q_OFF_UNITS = 4992;
  QStrip16x64 query{};
  KeyT64x208 key_t{};
  int32_t expected[16][208] = {};

  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 64; ++c)
      query[r][c] = int8_t(((r * 11 + c * 7 + 1) % 27) - 13);
  }
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 208; ++c)
      key_t[r][c] = int8_t(((r * 5 + c * 3 + 2) % 29) - 14);
  }

  matmul_ref_16x64x208(query, key_t, expected);
  sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(Q_OFF_UNITS) * 16, flatten_qstrip_row_major(query));
  sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, flatten_keyt_row_major(key_t));
  s.load({
      insn::CONFIG_TILE(1, 13, 4),
      insn::MATMUL(BUF_ABUF_ID, Q_OFF_UNITS, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });
  s.run(1500000);
  expect_clean_halt(name, s.dut.get());

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 208; ++j) {
      int32_t got = read_accum_wide(s.dut.get(), 0, i, j, 208);
      if (got != expected[i][j]) {
        std::fprintf(stderr,
                     "wide nonzero-qoff QK^T mismatch row=%d col=%d got=%d exp=%d\n",
                     i, j, got, expected[i][j]);
        TEST_FAIL(name, "pre-transposed nonzero-qoff QK^T MATMUL mismatch");
      }
    }
  }
  TEST_PASS(name);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_qkt_key_transpose_208x64();
  test_qkt_matmul_pretransposed_16x64x208();
  test_qkt_matmul_pretransposed_nonzero_qoff_16x64x208();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
