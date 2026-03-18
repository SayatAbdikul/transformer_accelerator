# TACCEL RTL Implementation Plan

## 1. Top-Level Architecture

```
                          AXI4 Master
                              │
              ┌───────────────┴───────────────┐
              │         taccel_top             │
              │  ┌─────────────────────────┐   │
              │  │      fetch_unit         │   │  DRAM[PC×8] → 64-bit insn
              │  └──────────┬──────────────┘   │
              │  ┌──────────▼──────────────┐   │
              │  │      decode_unit        │   │  5-bit opcode → control signals
              │  └──────────┬──────────────┘   │
              │  ┌──────────▼──────────────┐   │
              │  │     control_unit        │   │  Issue stage + SYNC barriers
              │  │  (scoreboard + FSM)     │   │
              │  └───┬────────┬────────┬───┘   │
              │      │        │        │       │
              │  ┌───▼──┐ ┌──▼───┐ ┌──▼───┐   │
              │  │ DMA  │ │ SYS  │ │ SFU  │   │
              │  │engine│ │array │ │      │   │
              │  └───┬──┘ └──┬───┘ └──┬───┘   │
              │      │       │        │        │
              │  ┌───▼───────▼────────▼───┐    │
              │  │     sram_subsystem     │    │
              │  │  ABUF  WBUF  ACCUM     │    │
              │  └────────────────────────┘    │
              └────────────────────────────────┘
```

### Clock & Reset
- Single clock domain (`clk`, `rst_n` active-low)
- Target: 200 MHz (5 ns period) — FPGA-friendly
- Synchronous reset for all flops

### External Interface
- AXI4 master for DRAM access (instruction fetch + DMA)
- `start` signal to begin execution at PC=0
- `done` output (asserted on HALT)
- `fault` output + `fault_code[3:0]` status register
- Optional: interrupt output

---

## 2. Module Hierarchy

### 2.1 `taccel_top` — Top-Level Wrapper
```systemverilog
module taccel_top #(
    parameter DRAM_ADDR_W = 56,
    parameter DRAM_DATA_W = 128     // 16-byte aligned AXI transfers
)(
    input  logic clk, rst_n,
    input  logic start,
    output logic done, fault,
    output logic [3:0] fault_code,
    // AXI4 master interface
    ...
);
```

Instantiates all submodules, wires control signals, multiplexes SRAM ports.

### 2.2 `fetch_unit` — Instruction Fetch
- Reads 8 bytes from DRAM at address `PC × 8`
- Issues AXI4 read burst (single beat, 8 bytes within 16-byte alignment)
- Outputs `insn_valid`, `insn_data[63:0]`
- Stalls when `control_unit` is not ready (SYNC stall, unit busy)
- Optional: instruction cache (64-entry direct-mapped) for tight loops — defer to Phase 3

**Microarchitecture:**
```
PC_reg → AXI addr calc → AXI read request → insn_data latch → insn_valid
                                                  ↑
                                          AXI read response
```

### 2.3 `decode_unit` — Instruction Decode
- Combinational decode of `insn_data[63:0]`
- Extracts opcode `[63:59]`, then format-specific fields
- Outputs a decoded control struct:

```systemverilog
typedef struct packed {
    logic [4:0]  opcode;
    // R-type fields
    logic [1:0]  src1_buf, src2_buf, dst_buf;
    logic [15:0] src1_off, src2_off, dst_off;
    logic [3:0]  sreg;
    logic        flags;
    // M-type fields
    logic [1:0]  buf_id, addr_reg;
    logic [15:0] sram_off, xfer_len, dram_off;
    // B-type fields
    logic [15:0] length;
    logic [5:0]  src_rows;
    logic        transpose;
    // A-type fields
    logic [27:0] imm28;
    // C-type fields
    logic [9:0]  tile_m, tile_n, tile_k;
    // S-type SET_SCALE fields
    logic [1:0]  src_mode;
    logic [15:0] imm16;
    // S-type SYNC fields
    logic [2:0]  resource_mask;
    // Format tag
    logic [2:0]  format;  // R=0, M=1, B=2, A=3, C=4, S=5
    logic        valid;
    logic        illegal;
} decoded_insn_t;
```

- `illegal` asserted for opcodes 0x11–0x1F or reserved buffer ID 0b11

### 2.4 `control_unit` — Issue Stage & Scoreboard
Central FSM that manages instruction issue, SYNC barriers, and fault detection.

**States:** `IDLE → FETCH → DECODE → ISSUE → WAIT_SYNC → HALT → FAULT`

**Issue-stage instructions** (execute in 0–1 cycle, block the pipeline):
- `CONFIG_TILE` → writes tile_m/n/k registers
- `SET_SCALE` → writes scale_regs[sreg] (FP16)
- `SET_ADDR_LO` / `SET_ADDR_HI` → writes addr_regs[reg] bits
- `REQUANT` → dispatches to requant datapath (issue-stage, multi-cycle)
- `SCALE_MUL` → dispatches to scale_mul datapath
- `VADD` → dispatches to vadd datapath
- `BUF_COPY` → dispatches to buf_copy engine
- `NOP` → 1 cycle
- `HALT` → set `done`, enter HALT state

**Dispatched instructions** (non-blocking, overlap allowed):
- `LOAD` / `STORE` → dispatch to DMA engine
- `MATMUL` → dispatch to systolic array
- `SOFTMAX` / `LAYERNORM` / `GELU` → dispatch to SFU

**SYNC logic:**
```systemverilog
// 3-bit busy flags
logic dma_busy, sys_busy, sfu_busy;

// SYNC stalls until (resource_mask & {sfu_busy, sys_busy, dma_busy}) == 0
wire sync_clear = ~|(insn.resource_mask & {sfu_busy, sys_busy, dma_busy});
```

**Fault detection:**
- Illegal opcode → FAULT state, fault_code = 4'h1
- DRAM OOB (from DMA) → fault_code = 4'h2
- SRAM OOB → fault_code = 4'h3
- Missing CONFIG_TILE → fault_code = 4'h4
- Reserved buffer ID → fault_code = 4'h5

### 2.5 `dma_engine` — LOAD / STORE
- AXI4 master interface for DRAM reads (LOAD) and writes (STORE)
- Computes effective DRAM byte address: `addr_regs[ADDR_REG] + DRAM_OFF × 16`
- Transfers `xfer_len × 16` bytes between DRAM and SRAM buffer
- Burst-based: AXI4 INCR bursts, 16 bytes per beat
- Reports `dma_busy` / `dma_done` to control_unit
- **OOB check**: compare `dram_byte_addr + xfer_bytes` against DRAM size register (programmed at init)

```
Cycle count: xfer_len cycles (1 cycle per 16-byte unit)
```

**Microarchitecture:**
```
addr_calc → AXI burst setup → beat_counter → SRAM write/read port → done
```

### 2.6 `systolic_array` — 16x16 INT8 Matmul
- 16×16 array of PEs, each performing `acc += a × b` per cycle
- Weight-stationary dataflow: weights pre-loaded, activations streamed
- INT8 inputs (signed), INT32 accumulators
- Tiled: iterates over `(M+1) × (N+1) × (K+1)` tiles, 16 cycles per tile
- `flags[0]`: 0 = overwrite ACCUM, 1 = accumulate

**PE module:**
```systemverilog
module systolic_pe (
    input  logic        clk, rst_n, en,
    input  logic [7:0]  a_in,       // INT8 activation (from west)
    input  logic [7:0]  b_in,       // INT8 weight (from north)
    output logic [7:0]  a_out,      // pass east
    output logic [7:0]  b_out,      // pass south
    input  logic        acc_clear,  // overwrite mode
    output logic [31:0] acc         // INT32 accumulator
);
    always_ff @(posedge clk) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (en && acc_clear)
            acc <= $signed(a_in) * $signed(b_in);
        else if (en)
            acc <= acc + $signed(a_in) * $signed(b_in);
    end
    always_ff @(posedge clk) begin
        a_out <= a_in;  // systolic propagation
        b_out <= b_in;
    end
endmodule
```

**Array controller FSM:**
1. For each (m_tile, n_tile): load weights[n_tile×16..n_tile×16+15, :] into column registers
2. For each k_tile: stream 16 activation rows from ABUF, 16 cycles
3. After all k_tiles: drain accumulators to ACCUM buffer
4. Repeat for next (m_tile, n_tile)

```
Total cycles: (M+1) × (N+1) × (K+1) × 16
```

### 2.7 `sfu_top` — Special Function Unit
Contains three sub-engines:

#### 2.7.1 `sfu_softmax`
Per-row softmax: `x_fp32 = INT8 × in_scale → shift → exp → normalize → requant`
- **Pass 1**: find row max (comparator tree, M×N cycles)
- **Pass 2**: exp(x - max), accumulate sum, divide, requant (M×N cycles)
- Total: M × N × 2 cycles
- Requires: FP32 exp unit, FP32 divider

#### 2.7.2 `sfu_layernorm`
Per-row: `mean → variance → normalize → affine → requant`
- **Pass 1**: accumulate sum, sum-of-squares (M×N cycles)
- **Pass 2**: normalize, multiply gamma, add beta, requant (M×N cycles)
- Total: M × N × 2 cycles
- Gamma/beta read from WBUF at src2 (FP16, widened to FP32)
- Requires: FP32 reciprocal-sqrt, FP32 multiplier

#### 2.7.3 `sfu_gelu`
Element-wise: `GELU(x) = x × 0.5 × (1 + erf(x/√2))`
- erf via Abramowitz & Stegun 7.1.26 polynomial:
  ```
  t = 1 / (1 + 0.3275911 × |x|)
  erf ≈ sign(x) × (1 - (a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵) × exp(-x²))
  ```
- Per-element: 5 FMA + 1 exp + 1 reciprocal
- Total: M × N × 2 cycles
- Requires: FP32 exp unit (shared with softmax), FP32 reciprocal, FP32 FMA

#### 2.7.4 `fp32_alu` — Shared FP32 Datapath
All SFU operations share a common FP32 functional unit:
- FP32 multiplier (pipelined, 3-cycle latency)
- FP32 adder (pipelined, 2-cycle latency)
- FP32 exp (lookup + polynomial, ~8 cycles)
- FP32 reciprocal (Newton-Raphson, ~4 cycles)
- FP32 reciprocal-sqrt (Newton-Raphson, ~6 cycles)
- INT8↔FP32 converters (dequant/requant with round-half-to-even)

**Round-half-to-even (banker's rounding)** must be implemented in:
- REQUANT (INT32 → INT8)
- SCALE_MUL
- All SFU requant paths

### 2.8 `sram_subsystem` — Buffer Management
Three dual-port SRAMs:

| Buffer | Size | Width | Depth | Port A | Port B |
|--------|------|-------|-------|--------|--------|
| ABUF | 128 KB | 128 bit (16 bytes) | 8192 | R/W (DMA, BUF_COPY) | R (Systolic, SFU) |
| WBUF | 256 KB | 128 bit (16 bytes) | 16384 | R/W (DMA, BUF_COPY) | R (Systolic, SFU) |
| ACCUM | 64 KB | 128 bit (16 bytes) | 4096 | R/W (Systolic drain) | R/W (REQUANT, VADD) |

- Port arbitration: DMA has priority during LOAD/STORE; compute units access during MATMUL/SFU
- `BUF_COPY` uses port A of both src and dst buffers
- ACCUM stored as INT32 little-endian: each 16-byte row = 4 × INT32 values

### 2.9 `issue_stage_ops` — REQUANT / SCALE_MUL / VADD / BUF_COPY

These execute in the issue stage (stall the pipeline until done):

**REQUANT** (INT32 → INT8):
- Read INT32 tile from ACCUM, multiply by FP16 scale (widened to FP32), round-half-to-even, clip [-128,127], write INT8 to dst
- Cycles: M × N

**SCALE_MUL**:
- ACCUM path: INT32 × FP32(scale) → round → clip INT32
- ABUF path: INT8 × FP32(scale) → round → clip INT8
- Cycles: M × N

**VADD**:
- ABUF (INT8 saturating): `clip(src1 + src2, -128, 127)` — element-wise
- ACCUM (INT32 broadcast): `src1[M,N] + broadcast(src2[1,N], M)` — bias add
- Cycles: M × N

**BUF_COPY**:
- Flat: memcpy `length × 16` bytes between buffers
- Transpose: read `[src_rows×16, cols]`, write `[cols, src_rows×16]`
- Cycles: length

---

## 3. Implementation Phases

### Phase 1: Foundation (Weeks 1–3)

**Goal:** Fetch-decode-execute skeleton, NOP/HALT/SYNC, SRAM infrastructure.

| Module | Description | LOC est. |
|--------|-------------|----------|
| `taccel_top` | Top-level shell, clk/rst, AXI stub | 150 |
| `fetch_unit` | PC register, AXI read for 8 bytes | 200 |
| `decode_unit` | Combinational opcode → control struct | 300 |
| `control_unit` | FSM: IDLE→FETCH→DECODE→ISSUE→HALT, NOP/HALT/SYNC handling | 250 |
| `sram_subsystem` | 3 dual-port SRAM wrappers (ABUF/WBUF/ACCUM) | 200 |
| `register_file` | 16 × FP16 scale regs, 4 × 56-bit addr regs, tile config | 100 |

**Tests (Phase 1):**
- Verilator: decode all 17 opcodes, verify field extraction
- Verilator: NOP → HALT sequence, check `done` assertion
- Verilator: SYNC with various masks
- cocotb: load 8-byte instructions into DRAM model, verify fetch sequence
- cocotb: CONFIG_TILE → verify tile_config register

### Phase 2: DMA + Memory (Weeks 3–5)

**Goal:** LOAD/STORE working with AXI4 master, SET_ADDR_LO/HI.

| Module | Description | LOC est. |
|--------|-------------|----------|
| `dma_engine` | AXI4 master, burst read/write, address calc | 400 |
| `axi4_master` | AXI4 protocol: AR/R channels (read), AW/W/B channels (write) | 350 |

**Tests (Phase 2):**
- Verilator: SET_ADDR_LO + SET_ADDR_HI → verify 56-bit address composition
- Verilator: LOAD 16 bytes → verify SRAM contents match DRAM source
- Verilator: STORE → verify DRAM contents match SRAM source
- Verilator: LOAD then STORE roundtrip
- Verilator: STORE OOB → verify fault assertion
- cocotb: LOAD multi-unit transfer (xfer_len=16, 256 bytes)
- cocotb: LOAD to ABUF/WBUF/ACCUM, verify all three buffers
- cocotb: address register independence (R0 vs R1 vs R2 vs R3)

### Phase 3: Systolic Array (Weeks 5–8)

**Goal:** MATMUL producing correct INT32 results in ACCUM.

| Module | Description | LOC est. |
|--------|-------------|----------|
| `systolic_pe` | Single PE: a×b + acc | 40 |
| `systolic_array` | 16×16 PE mesh, data steering, weight loading | 500 |
| `systolic_controller` | Tile iteration FSM, SRAM read/write scheduling | 400 |

**Tests (Phase 3):**
- Verilator: single PE: verify `1×1 + 0 = 1`, `127×127 + acc`, signed multiply
- Verilator: 16×16 identity multiply: `A @ I = A`
- Verilator: `ones(16,16) @ ones(16,16) = 16*ones(16,16)`
- Verilator: accumulate flag: two MATMULs with flags=1 produce sum
- Verilator: multi-tile MATMUL (M=2, N=2, K=2) — 32×32 × 32×32 problem
- cocotb: golden model comparison — random 16×16 matrices, bitwise INT32 match
- cocotb: CONFIG_TILE → MATMUL → SYNC → verify ACCUM readback
- cocotb: overflow boundary test (max acc = ±644M, verify no wrap)

### Phase 4: Issue-Stage Compute (Weeks 8–10)

**Goal:** REQUANT, SCALE_MUL, VADD, BUF_COPY working.

| Module | Description | LOC est. |
|--------|-------------|----------|
| `requant_unit` | INT32 × FP32 → round-half-to-even → clip INT8 | 200 |
| `scale_mul_unit` | INT8/INT32 × FP32 scale → round → clip | 200 |
| `vadd_unit` | INT8 saturating / INT32 broadcast add | 150 |
| `buf_copy_unit` | Flat copy + transpose engine | 250 |
| `fp32_round_hte` | Round-half-to-even logic (shared) | 80 |

**Tests (Phase 4):**
- Verilator: REQUANT — scale=0.5, input=100 → 50; scale=1.0, input=200 → 127 (clip)
- Verilator: REQUANT round-half-to-even: 0.5→0, 1.5→2, 2.5→2, 3.5→4
- Verilator: SCALE_MUL INT8 and INT32 paths
- Verilator: VADD INT8 saturating: 100+50→127, -100+-50→-128, 10+20→30
- Verilator: VADD INT32 broadcast: bias vector added to all rows
- Verilator: BUF_COPY flat: ABUF→WBUF 256 bytes exact match
- Verilator: BUF_COPY transpose: [32,16]→[16,32] element-by-element verify
- cocotb: MATMUL → SYNC → REQUANT → verify INT8 output vs golden model
- cocotb: MATMUL → VADD(bias) → REQUANT pipeline

### Phase 5: SFU (Weeks 10–14)

**Goal:** SOFTMAX, LAYERNORM, GELU matching golden model output.

| Module | Description | LOC est. |
|--------|-------------|----------|
| `fp32_mul` | Pipelined FP32 multiplier | 200 |
| `fp32_add` | Pipelined FP32 adder | 200 |
| `fp32_exp` | Lookup + polynomial FP32 exp | 300 |
| `fp32_recip` | Newton-Raphson reciprocal | 150 |
| `fp32_rsqrt` | Newton-Raphson reciprocal sqrt | 150 |
| `fp32_fma` | Fused multiply-add | 250 |
| `int8_fp32_conv` | Dequant (INT8→FP32) and requant (FP32→INT8 w/ RTE) | 100 |
| `sfu_softmax` | 2-pass softmax controller | 350 |
| `sfu_layernorm` | 2-pass layernorm controller | 350 |
| `sfu_gelu` | erf polynomial + GELU pipeline | 300 |
| `sfu_top` | Arbitration, shared datapath mux | 200 |

**Tests (Phase 5):**
- Verilator: fp32_mul — exhaustive on special values (0, ±inf, NaN, denorms)
- Verilator: fp32_exp — compare vs software exp() over [-10, 10], max ULP error
- Verilator: fp32_recip — verify `x × recip(x) ≈ 1.0` for range
- Verilator: erf polynomial — compare vs scipy erf over [-8, 8], verify max error < 1e-6
- Verilator: GELU single element — known values (GELU(0)=0, GELU(1)≈0.8413, GELU(-1)≈-0.1587)
- Verilator: SOFTMAX — uniform input → uniform output; one-hot input → near-one-hot output
- Verilator: LAYERNORM — identity gamma/beta → zero-mean, unit-variance output
- cocotb: SOFTMAX 16×16 tile — bitwise INT8 match vs golden model (scipy path)
- cocotb: LAYERNORM 16×16 tile — bitwise match vs golden model
- cocotb: GELU 16×16 tile — bitwise match vs golden model
- cocotb: SFU with various scale register values

### Phase 6: Integration & Pipeline (Weeks 14–17)

**Goal:** Full instruction set working, multi-instruction programs, parallel execution with SYNC.

| Task | Description | LOC est. |
|------|-------------|----------|
| Pipeline hazard resolution | Ensure SYNC properly stalls; no RAW hazards on SRAM | 200 |
| SET_SCALE from buffer | src_mode 1/2/3 reading FP16 from ABUF/WBUF/ACCUM | 50 |
| Fault handler | All 5 fault types set fault_code and halt | 100 |
| Cycle counter | Optional performance counter register | 50 |

**Tests (Phase 6):**
- cocotb: DMA LOAD → SYNC → MATMUL → SYNC → REQUANT → SYNC → STORE (full datapath)
- cocotb: Parallel LOAD overlapping MATMUL (no SYNC between), verify correctness
- cocotb: SYNC with wrong mask → data corruption detected (negative test)
- cocotb: CONFIG_TILE required before compute — verify fault without it
- cocotb: illegal opcode (0x11) → verify fault_code and halt
- cocotb: reserved buffer ID (0b11) → verify fault
- cocotb: run assembler output through RTL, compare SRAM state vs golden model
- cocotb: multi-block attention: QKV → MATMUL → SOFTMAX → MATMUL → LAYERNORM

### Phase 7: Full Model Validation (Weeks 17–20)

**Goal:** Run DeiT-tiny compiled binary on RTL, bit-exact match vs golden model.

| Task | Description |
|------|-------------|
| DRAM image loading | Load `to_dram_image()` output into RTL testbench DRAM |
| Input injection | Write INT8 patches to `input_offset` in DRAM |
| Output extraction | Read classifier output from DRAM after HALT |
| Golden comparison | Compare ACCUM/ABUF snapshots at key checkpoints |
| Performance | Measure total cycles, compare vs golden model cycle_count |

**Tests (Phase 7):**
- cocotb: single transformer block (1 of 12) — full attention + MLP
- cocotb: full DeiT-tiny inference — 28,149 instructions, compare final logits
- cocotb: top-1 accuracy on validation images matches golden model (65%)

---

## 4. Directory Structure

```
rtl/
├── src/
│   ├── taccel_top.sv
│   ├── fetch_unit.sv
│   ├── decode_unit.sv
│   ├── control_unit.sv
│   ├── dma_engine.sv
│   ├── axi4_master.sv
│   ├── systolic/
│   │   ├── systolic_pe.sv
│   │   ├── systolic_array.sv
│   │   └── systolic_controller.sv
│   ├── sfu/
│   │   ├── sfu_top.sv
│   │   ├── sfu_softmax.sv
│   │   ├── sfu_layernorm.sv
│   │   └── sfu_gelu.sv
│   ├── fp32/
│   │   ├── fp32_mul.sv
│   │   ├── fp32_add.sv
│   │   ├── fp32_exp.sv
│   │   ├── fp32_recip.sv
│   │   ├── fp32_rsqrt.sv
│   │   ├── fp32_fma.sv
│   │   ├── fp32_round_hte.sv    # Round-half-to-even
│   │   └── int8_fp32_conv.sv
│   ├── compute/
│   │   ├── requant_unit.sv
│   │   ├── scale_mul_unit.sv
│   │   ├── vadd_unit.sv
│   │   └── buf_copy_unit.sv
│   ├── memory/
│   │   ├── sram_subsystem.sv
│   │   ├── sram_dp.sv           # Parameterized dual-port SRAM
│   │   └── register_file.sv     # Scale regs, addr regs, tile config
│   ├── include/
│   │   ├── taccel_pkg.sv        # Types, parameters, constants
│   │   └── axi4_if.sv           # AXI4 interface definition
│   └── tb/
│       └── axi4_slave_model.sv  # AXI4 DRAM model for simulation
│
├── verilator/
│   ├── Makefile
│   ├── include/
│   │   ├── golden_model.h       # C++ wrapper calling Python golden model
│   │   ├── testbench.h          # Common TB infrastructure
│   │   └── dram_model.h         # DRAM behavioral model
│   ├── test_decode.cpp
│   ├── test_systolic_pe.cpp
│   ├── test_systolic_array.cpp
│   ├── test_dma.cpp
│   ├── test_requant.cpp
│   ├── test_vadd.cpp
│   ├── test_buf_copy.cpp
│   ├── test_fp32_ops.cpp
│   ├── test_sfu_gelu.cpp
│   ├── test_sfu_softmax.cpp
│   ├── test_sfu_layernorm.cpp
│   └── test_control_unit.cpp
│
├── cocotb/
│   ├── Makefile
│   ├── conftest.py              # Shared fixtures (clock, reset, DRAM model)
│   ├── utils/
│   │   ├── axi4_driver.py       # AXI4 bus functional model
│   │   ├── dram_model.py        # Python DRAM behavioral model
│   │   └── golden_bridge.py     # Import taccel golden model for comparison
│   ├── test_fetch_decode.py
│   ├── test_dma.py
│   ├── test_systolic.py
│   ├── test_issue_ops.py        # REQUANT, SCALE_MUL, VADD, BUF_COPY
│   ├── test_sfu.py
│   ├── test_sync.py
│   ├── test_faults.py
│   ├── test_multi_insn.py       # Multi-instruction sequences
│   └── test_deit_inference.py   # Full model inference
│
├── synth/
│   ├── constraints.xdc          # FPGA timing constraints
│   └── Makefile                 # Synthesis flow (Vivado / Yosys)
│
└── docs/
    └── microarchitecture.md
```

---

## 5. `taccel_pkg.sv` — Shared Constants

```systemverilog
package taccel_pkg;

    // Opcodes
    typedef enum logic [4:0] {
        OP_NOP        = 5'h00,
        OP_HALT       = 5'h01,
        OP_SYNC       = 5'h02,
        OP_CONFIG_TILE= 5'h03,
        OP_SET_SCALE  = 5'h04,
        OP_SET_ADDR_LO= 5'h05,
        OP_SET_ADDR_HI= 5'h06,
        OP_LOAD       = 5'h07,
        OP_STORE      = 5'h08,
        OP_BUF_COPY   = 5'h09,
        OP_MATMUL     = 5'h0A,
        OP_REQUANT    = 5'h0B,
        OP_SCALE_MUL  = 5'h0C,
        OP_VADD       = 5'h0D,
        OP_SOFTMAX    = 5'h0E,
        OP_LAYERNORM  = 5'h0F,
        OP_GELU       = 5'h10
    } opcode_t;

    // Buffer IDs
    typedef enum logic [1:0] {
        BUF_ABUF  = 2'b00,
        BUF_WBUF  = 2'b01,
        BUF_ACCUM = 2'b10
        // 2'b11 = reserved (fault)
    } buf_id_t;

    // Buffer sizes
    parameter ABUF_BYTES  = 131072;  // 128 KB
    parameter WBUF_BYTES  = 262144;  // 256 KB
    parameter ACCUM_BYTES =  65536;  //  64 KB

    // Systolic array
    parameter SYS_DIM = 16;
    parameter TILE_BYTES = SYS_DIM * SYS_DIM;  // 256 bytes

    // AXI
    parameter AXI_DATA_W = 128;
    parameter AXI_ADDR_W = 56;

    // DRAM address registers
    parameter NUM_ADDR_REGS = 4;
    parameter ADDR_WIDTH = 56;

    // Scale registers
    parameter NUM_SCALE_REGS = 16;

    // Tile config
    parameter TILE_DIM_BITS = 10;

    // Instruction format
    typedef enum logic [2:0] {
        FMT_R = 3'd0,
        FMT_M = 3'd1,
        FMT_B = 3'd2,
        FMT_A = 3'd3,
        FMT_C = 3'd4,
        FMT_S = 3'd5
    } insn_fmt_t;

    // Execution units (for SYNC mask)
    parameter UNIT_DMA = 0;
    parameter UNIT_SYS = 1;
    parameter UNIT_SFU = 2;

    // Fault codes
    typedef enum logic [3:0] {
        FAULT_NONE        = 4'h0,
        FAULT_ILLEGAL_OP  = 4'h1,
        FAULT_DRAM_OOB    = 4'h2,
        FAULT_SRAM_OOB    = 4'h3,
        FAULT_NO_CONFIG   = 4'h4,
        FAULT_BAD_BUF     = 4'h5
    } fault_code_t;

    // Erf polynomial coefficients (A&S 7.1.26) — bit patterns for FP32
    parameter logic [31:0] ERF_A1 = 32'h3E82_7906;  //  0.254829592
    parameter logic [31:0] ERF_A2 = 32'hBE91_A98E;  // -0.284496736
    parameter logic [31:0] ERF_A3 = 32'h3FB5_D78E;  //  1.421413741
    parameter logic [31:0] ERF_A4 = 32'hBFBA_0005;  // -1.453152027
    parameter logic [31:0] ERF_A5 = 32'h3F87_DC22;  //  1.061405429
    parameter logic [31:0] ERF_P  = 32'h3EA7_B9D2;  //  0.3275911

endpackage
```

---

## 6. Verilator Test Strategy

### Infrastructure (`testbench.h`)
```cpp
class TestBench {
    std::unique_ptr<Vtaccel_top> dut;
    VerilatedVcdC* trace;
    uint64_t tick_count;
    std::vector<uint8_t> dram;  // 16 MB DRAM model

    void tick();           // posedge + negedge
    void reset();          // 10 cycles of rst_n=0
    void load_dram(const uint8_t* data, size_t len, uint64_t addr);
    void read_sram(buf_id_t buf, uint16_t offset, uint8_t* dst, size_t len);
    void run_until_halt(uint64_t max_cycles = 10'000'000);
    bool check_fault();
};
```

### Test Categories

**Unit tests** (per-module, small, fast):
- `test_decode.cpp`: Feed all 17 opcode encodings → verify decoded struct fields
- `test_systolic_pe.cpp`: Stimulate single PE with known a/b → check acc
- `test_fp32_ops.cpp`: fp32_mul, fp32_add, fp32_exp, fp32_recip against C++ reference
- `test_requant.cpp`: Known INT32 × scale → verify INT8 output with RTE

**Integration tests** (multi-module, medium):
- `test_systolic_array.cpp`: Identity matmul, random matmul vs C++ reference
- `test_dma.cpp`: LOAD/STORE with AXI DRAM model, verify byte-accurate transfer
- `test_sfu_gelu.cpp`: 16×16 tile GELU, compare vs `_erf_poly` reference in C++

**System tests** (full chip, slow):
- `test_control_unit.cpp`: Multi-instruction programs (CONFIG_TILE → MATMUL → SYNC → HALT)

### Golden Model Bridge (C++)
```cpp
// Calls Python golden model via pybind11 or subprocess
class GoldenModel {
    // Load assembled program, run golden model, extract state
    void load_program(const std::string& asm_source);
    void run();
    std::vector<int8_t> get_abuf(size_t offset, size_t len);
    std::vector<int32_t> get_accum(size_t offset, size_t len);
};
```

Alternatively: export golden model reference data as `.bin` files during test setup, load in C++.

---

## 7. cocotb Test Strategy

### Infrastructure (`conftest.py`)
```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.handle import SimHandleBase
import sys
sys.path.insert(0, "../software")
from taccel.assembler.assembler import Assembler
from taccel.golden_model.simulator import Simulator
from taccel.golden_model.state import MachineState

@cocotb.coroutine
async def reset_dut(dut, cycles=10):
    dut.rst_n.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.coroutine
async def load_program_to_dram(dut, asm_source, data=b""):
    """Assemble, build DRAM image, load into testbench DRAM model."""
    prog = Assembler().assemble(asm_source, data=data)
    image = prog.to_dram_image()
    # Write to AXI slave DRAM model
    for i, byte in enumerate(image):
        dut.dram_model.mem[i] = byte
    return prog

@cocotb.coroutine
async def run_until_halt(dut, max_cycles=100_000):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            return
    raise TimeoutError("DUT did not halt")
```

### Test Files

**`test_fetch_decode.py`** — Instruction fetch and decode:
```python
@cocotb.test()
async def test_nop_halt(dut):
    """NOP → HALT: verify done signal."""
    await reset_dut(dut)
    await load_program_to_dram(dut, "NOP\nHALT")
    dut.start.value = 1
    await run_until_halt(dut)
    assert dut.done.value == 1
    assert dut.fault.value == 0
```

**`test_dma.py`** — DMA operations:
```python
@cocotb.test()
async def test_load_store_roundtrip(dut):
    """LOAD from DRAM → ABUF → STORE to different DRAM address."""
    prog = await load_program_to_dram(dut, """
        SET_ADDR_LO R0, 0x0000000
        SET_ADDR_HI R0, 0x0000000
        LOAD buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=0, dram_off=0
        SYNC 0b001
        SET_ADDR_LO R1, 0x0100000
        SET_ADDR_HI R1, 0x0000000
        STORE buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=1, dram_off=0
        SYNC 0b001
        HALT
    """, data=bytes(range(256)))
    dut.start.value = 1
    await run_until_halt(dut)
    # Verify DRAM at 0x100000 matches original data
    for i in range(256):
        assert dut.dram_model.mem[0x100000 + i] == i
```

**`test_systolic.py`** — Matmul:
```python
@cocotb.test()
async def test_matmul_identity(dut):
    """A @ I = A (INT32)."""
    # Compare RTL ACCUM state vs golden model
    asm = """
        CONFIG_TILE M=1, N=1, K=1
        MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0
        SYNC 0b010
        HALT
    """
    golden_sim = make_golden(asm, abuf_data=A.tobytes(), wbuf_data=I.tobytes())
    golden_sim.run()
    # ... run RTL, compare ACCUM contents ...

@cocotb.test()
async def test_matmul_random_golden(dut):
    """Random 16x16 matmul, bitwise match vs golden model."""
    A = np.random.randint(-128, 127, (16, 16), dtype=np.int8)
    B = np.random.randint(-128, 127, (16, 16), dtype=np.int8)
    expected = A.astype(np.int32) @ B.astype(np.int32)
    # ... load to RTL, run, compare ...
```

**`test_sfu.py`** — Special functions:
```python
@cocotb.test()
async def test_gelu_vs_golden(dut):
    """GELU 16x16 tile — bitwise INT8 match vs golden model."""
    # ... assemble GELU program, run golden, run RTL, compare byte-for-byte ...

@cocotb.test()
async def test_softmax_row_sum(dut):
    """Softmax output rows approximately sum to 1.0 after dequant."""

@cocotb.test()
async def test_layernorm_zero_mean(dut):
    """LayerNorm output has approximately zero mean per row."""
```

**`test_faults.py`** — Error handling:
```python
@cocotb.test()
async def test_illegal_opcode(dut):
    """Opcode 0x11 → fault, halt."""
    # Manually inject 0x11 << 59 as raw instruction bytes

@cocotb.test()
async def test_missing_config_tile(dut):
    """MATMUL without CONFIG_TILE → fault."""

@cocotb.test()
async def test_dram_oob(dut):
    """STORE beyond DRAM boundary → fault."""
```

**`test_deit_inference.py`** — Full model (Phase 7):
```python
@cocotb.test(timeout_time=600, timeout_unit="sec")
async def test_deit_tiny_full(dut):
    """Run compiled DeiT-tiny binary, compare logits vs golden model."""
    from taccel.compiler.compiler import Compiler
    compiler = Compiler()
    program = compiler.compile()
    image = program.to_dram_image()
    # Load image, inject input patches, run RTL
    # Extract classifier output from DRAM
    # Compare top-1 prediction vs golden model
```

---

## 8. Critical Design Decisions for RTL

### 8.1 SRAM Implementation
- **FPGA**: use BRAM inference (`(* ram_style = "block" *)`), dual-port
- **ASIC**: use SRAM compiler (e.g., ARM Artisan, TSMC memory compiler)
- Dual-port needed for simultaneous DMA write + compute read

### 8.2 AXI4 Interface
- 128-bit data bus (16 bytes per beat, matches TACCEL's 16-byte transfer unit)
- INCR burst type, burst length = xfer_len
- Single outstanding transaction initially; add transaction pipelining in Phase 6 if needed
- Instruction fetch shares AXI port with DMA (arbitrated, fetch has lower priority during active DMA)

### 8.3 FP32 Datapath
- SFU is the only unit requiring FP32 — keep FP32 logic isolated in `fp32/`
- Share exp unit between softmax and GELU (time-multiplexed, never concurrent)
- FP16↔FP32 conversion: zero-extend mantissa, adjust exponent bias (no rounding needed for widening)
- Round-half-to-even: check guard/round/sticky bits after FP32 multiply

### 8.4 Pipeline Bubbles
- Instruction fetch latency: ~10 cycles (AXI read). Mitigate with:
  - 2-entry instruction prefetch buffer (fetch PC+1 while executing PC)
  - Optional: 64-entry direct-mapped instruction cache (Phase 3 optimization)
- MATMUL drains results to ACCUM after all tiles — ~16 cycles stall between last tile and ACCUM availability

### 8.5 Clock Domain
- Single clock domain for Phase 1–7
- Future: separate DRAM clock (AXI side) from compute clock (if frequency differs)

---

## 9. Resource Estimates (Xilinx UltraScale+)

| Module | LUTs | FFs | BRAM (36Kb) | DSP |
|--------|------|-----|-------------|-----|
| Systolic 16×16 | 8K | 10K | 0 | 256 (8-bit mult) |
| ABUF (128KB) | — | — | 32 | — |
| WBUF (256KB) | — | — | 64 | — |
| ACCUM (64KB) | — | — | 16 | — |
| DMA engine | 2K | 1K | 0 | 0 |
| SFU (FP32) | 15K | 8K | 4 (LUT) | 12 |
| Control + decode | 3K | 2K | 0 | 0 |
| **Total** | **~30K** | **~22K** | **~116** | **~268** |

Fits comfortably on a ZCU104 (XCZU7EV: 230K LUTs, 312 BRAM, 1728 DSP).

---

## 10. Performance Targets

| Metric | Target |
|--------|--------|
| Clock frequency | 200 MHz |
| MATMUL throughput | 256 INT8 MACs/cycle (16×16 array) |
| DMA bandwidth | 3.2 GB/s (128-bit @ 200 MHz) |
| DeiT-tiny inference | ~1.5M cycles (~7.5 ms @ 200 MHz) |
| Peak INT8 TOPS | 0.1 TOPS (256 MACs × 200 MHz × 2 ops/MAC) |

---

## 11. Milestone Checklist

| # | Milestone | Gate Criteria |
|---|-----------|---------------|
| M1 | Skeleton boots | NOP→HALT halts, waveform clean |
| M2 | DMA works | LOAD/STORE roundtrip, byte-accurate |
| M3 | MATMUL correct | 16×16 identity and random matmul match golden model |
| M4 | Issue ops work | REQUANT/VADD/SCALE_MUL/BUF_COPY all pass unit tests |
| M5 | SFU correct | SOFTMAX/LAYERNORM/GELU bitwise match golden model |
| M6 | Full ISA | All 17 opcodes + faults working, multi-instruction cocotb passes |
| M7 | DeiT-tiny runs | Full compiled binary produces correct top-1 predictions |
| M8 | FPGA demo | Bitstream generated, runs on ZCU104 at 200 MHz |
