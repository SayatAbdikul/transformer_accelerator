// Verilator systolic tests for TACCEL Phase 3.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/testbench.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

static int tests_run = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
  printf("PASS: %s\n", name); tests_pass++; tests_run++; \
} while (0)
#define TEST_FAIL(name, msg) do { \
  fprintf(stderr, "FAIL: %s - %s\n", name, msg); std::exit(1); \
} while (0)

struct Sim {
  std::unique_ptr<Vtaccel_top> dut;
  AXI4SlaveModel dram;

  Sim() : dut(std::make_unique<Vtaccel_top>()), dram(16 * 1024 * 1024) {
    do_reset(dut.get());
  }

  void load(const std::vector<uint64_t>& prog) {
    dram.write_program(prog);
  }

  void run(int timeout = 300000) {
    dut->start = 1;
    tick(dut.get(), dram);
    dut->start = 0;
    run_until_halt(dut.get(), dram, timeout);
  }
};

static void write_abuf_row(Vtaccel_top* dut, int row, const int8_t vals[16]) {
  auto* r = dut->rootp;
  for (int w = 0; w < 4; ++w) {
    uint32_t word = 0;
    for (int b = 0; b < 4; ++b) {
      uint8_t v = static_cast<uint8_t>(vals[w * 4 + b]);
      word |= uint32_t(v) << (8 * b);
    }
    r->taccel_top__DOT__u_sram__DOT__u_abuf__DOT__mem[row][w] = word;
  }
}

static void write_wbuf_row(Vtaccel_top* dut, int row, const int8_t vals[16]) {
  auto* r = dut->rootp;
  for (int w = 0; w < 4; ++w) {
    uint32_t word = 0;
    for (int b = 0; b < 4; ++b) {
      uint8_t v = static_cast<uint8_t>(vals[w * 4 + b]);
      word |= uint32_t(v) << (8 * b);
    }
    r->taccel_top__DOT__u_sram__DOT__u_wbuf__DOT__mem[row][w] = word;
  }
}

static int32_t read_accum_ij(Vtaccel_top* dut, int dst_off, int i, int j) {
  auto* r = dut->rootp;
  int grp = j / 4;
  int lane = j % 4;
  int row = dst_off + i * 4 + grp;
  uint32_t word = r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
  return static_cast<int32_t>(word);
}

static void matmul_ref(const int8_t a[16][16], const int8_t b[16][16], int32_t c[16][16]) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < 16; ++k) {
        acc += int32_t(a[i][k]) * int32_t(b[k][j]);
      }
      c[i][j] = acc;
    }
  }
}

static void matmul_ref_32(const int8_t a[32][32], const int8_t b[32][32], int32_t c[32][32]) {
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < 32; ++k) {
        acc += int32_t(a[i][k]) * int32_t(b[k][j]);
      }
      c[i][j] = acc;
    }
  }
}

static void matmul_ref_16x64x16(const int8_t a[16][64], const int8_t b[64][16], int32_t c[16][16]) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < 64; ++k) {
        acc += int32_t(a[i][k]) * int32_t(b[k][j]);
      }
      c[i][j] = acc;
    }
  }
}

static void load_matrix_abuf(Vtaccel_top* dut, int off, const int8_t m[16][16]) {
  // Current Phase 3 controller consumes ABUF as columns-per-cycle, so we
  // place A transposed in SRAM rows here.
  int8_t t[16][16] = {};
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 16; ++c)
      t[r][c] = m[c][r];
  for (int r = 0; r < 16; ++r) write_abuf_row(dut, off + r, t[r]);
}

static void load_matrix_wbuf(Vtaccel_top* dut, int off, const int8_t m[16][16]) {
  for (int r = 0; r < 16; ++r) write_wbuf_row(dut, off + r, m[r]);
}

static void load_abuf_tiled_32x32(Vtaccel_top* dut, int off, const int8_t m[32][32]) {
  for (int mt = 0; mt < 2; ++mt) {
    for (int kt = 0; kt < 2; ++kt) {
      int tile = mt * 2 + kt;
      for (int r = 0; r < 16; ++r) {
        int8_t packed[16];
        for (int c = 0; c < 16; ++c) {
          // ABUF tiles are stored transposed per 16x16 tile.
          packed[c] = m[mt * 16 + c][kt * 16 + r];
        }
        write_abuf_row(dut, off + tile * 16 + r, packed);
      }
    }
  }
}

static void load_wbuf_tiled_32x32(Vtaccel_top* dut, int off, const int8_t m[32][32]) {
  for (int kt = 0; kt < 2; ++kt) {
    for (int nt = 0; nt < 2; ++nt) {
      int tile = kt * 2 + nt;
      for (int r = 0; r < 16; ++r) {
        int8_t packed[16];
        for (int c = 0; c < 16; ++c) packed[c] = m[kt * 16 + r][nt * 16 + c];
        write_wbuf_row(dut, off + tile * 16 + r, packed);
      }
    }
  }
}

static int32_t read_accum_32x32(Vtaccel_top* dut, int off, int i, int j) {
  int mt = i / 16;
  int nt = j / 16;
  int li = i % 16;
  int lj = j % 16;

  int tile = mt * 2 + nt;
  int grp = lj / 4;
  int lane = lj % 4;
  int row = off + tile * 64 + li * 4 + grp;

  auto* r = dut->rootp;
  uint32_t word = r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
  return static_cast<int32_t>(word);
}

static void load_abuf_ktiles_16x64(Vtaccel_top* dut, int off, const int8_t a[16][64]) {
  // Tile order in controller: (mtile * k_tiles + ktile), with mtile=0 here.
  for (int kt = 0; kt < 4; ++kt) {
    int tile = kt;
    for (int r = 0; r < 16; ++r) {
      int8_t packed[16];
      for (int c = 0; c < 16; ++c) {
        // ABUF tiles stored transposed (column-major by cycle).
        packed[c] = a[c][kt * 16 + r];
      }
      write_abuf_row(dut, off + tile * 16 + r, packed);
    }
  }
}

static void load_wbuf_ktiles_64x16(Vtaccel_top* dut, int off, const int8_t b[64][16]) {
  // Tile order in controller: (ktile * n_tiles + ntile), with ntile=0 here.
  for (int kt = 0; kt < 4; ++kt) {
    int tile = kt;
    for (int r = 0; r < 16; ++r) {
      int8_t packed[16];
      for (int c = 0; c < 16; ++c) packed[c] = b[kt * 16 + r][c];
      write_wbuf_row(dut, off + tile * 16 + r, packed);
    }
  }
}

static void test_matmul_identity() {
  const char* name = "matmul_identity_16x16";
  Sim s;

  int8_t a[16][16] = {};
  int8_t eye[16][16] = {};
  int32_t exp[16][16] = {};

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = static_cast<int8_t>((i * 3 + j) & 0x7F);
      eye[i][j] = (i == j) ? 1 : 0;
    }
  }

  load_matrix_abuf(s.dut.get(), 0, a);
  load_matrix_wbuf(s.dut.get(), 0, eye);
  matmul_ref(a, eye, exp);

  s.load({
      insn::CONFIG_TILE(1, 1, 1),
      insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });

  s.run();
  if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t got = read_accum_ij(s.dut.get(), 0, i, j);
      if (got != exp[i][j]) {
        fprintf(stderr, "identity mismatch i=%d j=%d got=%d exp=%d\n", i, j, got, exp[i][j]);
        TEST_FAIL(name, "ACCUM mismatch");
      }
    }
  }

  TEST_PASS(name);
}

static void test_matmul_ones() {
  const char* name = "matmul_ones_16x16";
  Sim s;

  int8_t a[16][16] = {};
  int8_t b[16][16] = {};
  int32_t exp[16][16] = {};

  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      a[i][j] = 1;
      b[i][j] = 1;
    }

  load_matrix_abuf(s.dut.get(), 0, a);
  load_matrix_wbuf(s.dut.get(), 0, b);
  matmul_ref(a, b, exp);

  s.load({
      insn::CONFIG_TILE(1, 1, 1),
      insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });

  s.run();
  if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t got = read_accum_ij(s.dut.get(), 0, i, j);
      if (got != exp[i][j]) {
        fprintf(stderr, "ones mismatch i=%d j=%d got=%d exp=%d\n", i, j, got, exp[i][j]);
        TEST_FAIL(name, "ACCUM mismatch");
      }
    }
  }

  TEST_PASS(name);
}

static void test_matmul_accumulate_flag() {
  const char* name = "matmul_accumulate_flag";
  Sim s;

  int8_t a[16][16] = {};
  int8_t b[16][16] = {};
  int32_t exp[16][16] = {};

  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      a[i][j] = 1;
      b[i][j] = (i == j) ? 2 : 0;
    }

  load_matrix_abuf(s.dut.get(), 0, a);
  load_matrix_wbuf(s.dut.get(), 0, b);
  matmul_ref(a, b, exp);

  s.load({
      insn::CONFIG_TILE(1, 1, 1),
      insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 0),
      insn::SYNC(0b010),
      insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 1),
      insn::SYNC(0b010),
      insn::HALT(),
  });

  s.run();
  if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t got = read_accum_ij(s.dut.get(), 0, i, j);
      if (got != (exp[i][j] * 2)) {
        fprintf(stderr, "acc mismatch i=%d j=%d got=%d exp=%d\n", i, j, got, exp[i][j] * 2);
        TEST_FAIL(name, "accumulate mismatch");
      }
    }
  }

  TEST_PASS(name);
}

static void test_matmul_multitile_2x2x2() {
  const char* name = "matmul_multitile_2x2x2";
  Sim s;

  int8_t a[32][32] = {};
  int8_t b[32][32] = {};
  int32_t exp[32][32] = {};

  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      a[i][j] = static_cast<int8_t>(((i * 7 + j * 5 + 3) % 11) - 5);
      b[i][j] = static_cast<int8_t>(((i * 3 + j * 9 + 1) % 13) - 6);
    }
  }

  load_abuf_tiled_32x32(s.dut.get(), 0, a);
  load_wbuf_tiled_32x32(s.dut.get(), 0, b);
  matmul_ref_32(a, b, exp);

  s.load({
      insn::CONFIG_TILE(2, 2, 2),
      insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });

  s.run();
  if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      int32_t got = read_accum_32x32(s.dut.get(), 0, i, j);
      if (got != exp[i][j]) {
        fprintf(stderr, "multitile mismatch i=%d j=%d got=%d exp=%d\n", i, j, got, exp[i][j]);
        TEST_FAIL(name, "ACCUM mismatch");
      }
    }
  }

  TEST_PASS(name);
}

static void test_matmul_signed_extremes_16x16() {
  const char* name = "matmul_signed_extremes_16x16";
  Sim s;

  int8_t a[16][16] = {};
  int8_t b[16][16] = {};
  int32_t exp[16][16] = {};

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = ((i + j) & 1) ? int8_t(-128) : int8_t(127);
      b[i][j] = ((i * 3 + j * 5) & 1) ? int8_t(127) : int8_t(-128);
    }
  }

  load_matrix_abuf(s.dut.get(), 0, a);
  load_matrix_wbuf(s.dut.get(), 0, b);
  matmul_ref(a, b, exp);

  s.load({
      insn::CONFIG_TILE(1, 1, 1),
      insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });

  s.run();
  if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t got = read_accum_ij(s.dut.get(), 0, i, j);
      if (got != exp[i][j]) {
        fprintf(stderr, "signed-ext mismatch i=%d j=%d got=%d exp=%d\n", i, j, got, exp[i][j]);
        TEST_FAIL(name, "ACCUM mismatch");
      }
    }
  }

  TEST_PASS(name);
}

static void test_matmul_random_regression_16x16() {
  const char* name = "matmul_random_regression_16x16";
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(-128, 127);

  for (int tc = 0; tc < 8; ++tc) {
    Sim s;
    int8_t a[16][16] = {};
    int8_t b[16][16] = {};
    int32_t exp[16][16] = {};

    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        a[i][j] = static_cast<int8_t>(dist(rng));
        b[i][j] = static_cast<int8_t>(dist(rng));
      }
    }

    load_matrix_abuf(s.dut.get(), 0, a);
    load_matrix_wbuf(s.dut.get(), 0, b);
    matmul_ref(a, b, exp);

    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 0),
        insn::SYNC(0b010),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        int32_t got = read_accum_ij(s.dut.get(), 0, i, j);
        if (got != exp[i][j]) {
          fprintf(stderr, "random tc=%d mismatch i=%d j=%d got=%d exp=%d\n",
                  tc, i, j, got, exp[i][j]);
          TEST_FAIL(name, "ACCUM mismatch");
        }
      }
    }
  }

  TEST_PASS(name);
}

static void test_matmul_k4_boundary_stress() {
  const char* name = "matmul_k4_boundary_stress";
  Sim s;

  int8_t a[16][64] = {};
  int8_t b[64][16] = {};
  int32_t exp[16][16] = {};

  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 64; ++k) {
      a[i][k] = ((i + k) & 1) ? int8_t(127) : int8_t(-128);
    }
  }
  for (int k = 0; k < 64; ++k) {
    for (int j = 0; j < 16; ++j) {
      b[k][j] = ((k * 7 + j) & 1) ? int8_t(-128) : int8_t(127);
    }
  }

  load_abuf_ktiles_16x64(s.dut.get(), 0, a);
  load_wbuf_ktiles_64x16(s.dut.get(), 0, b);
  matmul_ref_16x64x16(a, b, exp);

  s.load({
      insn::CONFIG_TILE(1, 1, 4),
      insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 0),
      insn::SYNC(0b010),
      insn::HALT(),
  });

  s.run();
  if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      int32_t got = read_accum_ij(s.dut.get(), 0, i, j);
      if (got != exp[i][j]) {
        fprintf(stderr, "k4 mismatch i=%d j=%d got=%d exp=%d\n", i, j, got, exp[i][j]);
        TEST_FAIL(name, "ACCUM mismatch");
      }
    }
  }

  TEST_PASS(name);
}

static void test_matmul_multitile_random_regression_2x2x2() {
  const char* name = "matmul_multitile_random_regression_2x2x2";
  std::mt19937 rng(67890);
  std::uniform_int_distribution<int> dist(-128, 127);

  for (int tc = 0; tc < 4; ++tc) {
    Sim s;
    int8_t a[32][32] = {};
    int8_t b[32][32] = {};
    int32_t exp[32][32] = {};

    for (int i = 0; i < 32; ++i) {
      for (int j = 0; j < 32; ++j) {
        a[i][j] = static_cast<int8_t>(dist(rng));
        b[i][j] = static_cast<int8_t>(dist(rng));
      }
    }

    load_abuf_tiled_32x32(s.dut.get(), 0, a);
    load_wbuf_tiled_32x32(s.dut.get(), 0, b);
    matmul_ref_32(a, b, exp);

    s.load({
        insn::CONFIG_TILE(2, 2, 2),
        insn::MATMUL(0, 0, 1, 0, 2, 0, 0, 0),
        insn::SYNC(0b010),
        insn::HALT(),
    });

    s.run();
    if (!s.dut->done || s.dut->fault) TEST_FAIL(name, "did not halt cleanly");

    for (int i = 0; i < 32; ++i) {
      for (int j = 0; j < 32; ++j) {
        int32_t got = read_accum_32x32(s.dut.get(), 0, i, j);
        if (got != exp[i][j]) {
          fprintf(stderr, "multitile-rand tc=%d mismatch i=%d j=%d got=%d exp=%d\n",
                  tc, i, j, got, exp[i][j]);
          TEST_FAIL(name, "ACCUM mismatch");
        }
      }
    }
  }

  TEST_PASS(name);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_matmul_identity();
  test_matmul_ones();
  test_matmul_accumulate_flag();
  test_matmul_multitile_2x2x2();
  test_matmul_signed_extremes_16x16();
  test_matmul_random_regression_16x16();
  test_matmul_k4_boundary_stress();
  test_matmul_multitile_random_regression_2x2x2();

  printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
