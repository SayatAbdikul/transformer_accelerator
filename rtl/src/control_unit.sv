// Control unit — in-order issue stage FSM.
//
// Orchestrates instruction fetch, decode (via external decode_unit), and
// execution.  All issue-stage instructions execute here; long-latency
// instructions are dispatched to execution units via one-cycle dispatch pulses.
//
// FSM states:
//   IDLE        — awaiting start pulse
//   FETCH       — fetch_req asserted, waiting for insn_valid pulse
//   ISSUE       — decode + execute one instruction (1 cycle)
//   SYNC_WAIT   — barrier: stall until selected units drain
//   DISP_WAIT   — wait for in-flight ALU to finish (REQUANT/SCALE_MUL/VADD/BUF_COPY)
//   HALT        — terminal: done=1
//   FAULT       — terminal: fault=1
//
// Phase 1: dma_busy / sys_busy / sfu_busy / alu_busy tied to 0 externally.
// All dispatched instructions therefore complete in 0 cycles.

`ifndef CONTROL_UNIT_SV
`define CONTROL_UNIT_SV

`include "taccel_pkg.sv"

module control_unit
  import taccel_pkg::*;
(
  input  logic          clk,
  input  logic          rst_n,
  input  logic          start,

  // --- Fetch interface ---
  output logic [55:0]   pc,
  output logic          fetch_req,   // combinational, held in S_FETCH
  input  logic          insn_valid,  // 1-cycle pulse when instruction ready

  // --- Decoded instruction (stable after insn_valid; registered in top) ---
  input  decoded_insn_t insn,

  // --- Register file write ports ---
  output logic          scale_we,
  output logic [3:0]    scale_waddr,
  output logic [15:0]   scale_wdata,

  output logic          addr_lo_we,
  output logic          addr_hi_we,
  output logic [1:0]    addr_wsel,
  output logic [27:0]   addr_imm28,

  output logic          tile_we,
  output logic [9:0]    tile_m_in,
  output logic [9:0]    tile_n_in,
  output logic [9:0]    tile_k_in,

  // --- Register file read-back ---
  input  logic          tile_valid,

  // --- Dispatch to execution units (1-cycle pulses) ---
  output logic          dma_dispatch,
  output logic          sys_dispatch,
  output logic          sfu_dispatch,
  output logic          alu_dispatch,

  // --- Execution unit status ---
  input  logic          dma_busy,
  input  logic          sys_busy,
  input  logic          sfu_busy,
  input  logic          alu_busy,

  // --- Fault from execution units ---
  input  logic          dma_fault,
  input  logic [3:0]    dma_fault_code,

  // --- Status outputs ---
  output logic          done,
  output logic          fault,
  output logic [3:0]    fault_code
);

  // -------------------------------------------------------------------------
  // FSM state encoding
  // -------------------------------------------------------------------------
  typedef enum logic [2:0] {
    S_IDLE      = 3'd0,
    S_FETCH     = 3'd1,
    S_ISSUE     = 3'd2,
    S_SYNC_WAIT = 3'd3,
    S_DISP_WAIT = 3'd4,
    S_HALT      = 3'd5,
    S_FAULT     = 3'd6
  } ctrl_state_t;

  ctrl_state_t  state;
  logic [55:0]  pc_reg;
  logic [2:0]   sync_mask_q;  // latched resource_mask for SYNC_WAIT
  logic [3:0]   fault_code_r; // latched fault code

  assign pc         = pc_reg;
  assign fault_code = fault_code_r;

  // -------------------------------------------------------------------------
  // Pre-computed combinational helpers
  // -------------------------------------------------------------------------
  // Checks whether the units specified in mask are all idle
  logic sync_clear_q;   // for SYNC_WAIT: uses latched mask
  logic sync_clear_now; // for ISSUE: uses current insn mask

  assign sync_clear_q   = ~|(sync_mask_q       & {sfu_busy, sys_busy, dma_busy});
  assign sync_clear_now = ~|(insn.sync_mask & {sfu_busy, sys_busy, dma_busy});

  // -------------------------------------------------------------------------
  // Helper: does this opcode require CONFIG_TILE to have been set?
  // -------------------------------------------------------------------------
  function automatic logic needs_tile(input logic [4:0] op);
    return (op == OP_MATMUL   ||
            op == OP_REQUANT  ||
            op == OP_SCALE_MUL||
            op == OP_VADD     ||
            op == OP_SOFTMAX  ||
            op == OP_LAYERNORM||
            op == OP_GELU);
  endfunction

  // -------------------------------------------------------------------------
  // Sequential FSM + registers
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      pc_reg       <= 56'h0;
      sync_mask_q  <= 3'h0;
      fault_code_r <= 4'h0;

    end else begin
      case (state)

        // ------------------------------------------------------------------
        S_IDLE:
          if (start) begin
            pc_reg <= 56'h0;
            state  <= S_FETCH;
          end

        // ------------------------------------------------------------------
        // Stay in FETCH until insn_valid pulse.  The instruction is already
        // registered into insn_data_q in taccel_top on this same posedge, so
        // insn is valid on the NEXT cycle (S_ISSUE).
        S_FETCH:
          if (insn_valid)
            state <= S_ISSUE;

        // ------------------------------------------------------------------
        S_ISSUE: begin
          // DMA async fault takes priority
          if (dma_fault) begin
            fault_code_r <= dma_fault_code;
            state        <= S_FAULT;
          end else if (insn.illegal) begin
            fault_code_r <= (insn.opcode > 5'h10) ?
                             4'(FAULT_ILLEGAL_OP) : 4'(FAULT_BAD_BUF);
            state        <= S_FAULT;
          end else begin
            case (insn.opcode)
              OP_NOP: begin
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              OP_HALT:
                state <= S_HALT;

              OP_SYNC: begin
                if (sync_clear_now) begin
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end else begin
                  sync_mask_q <= insn.sync_mask;
                  state       <= S_SYNC_WAIT;
                end
              end

              OP_CONFIG_TILE: begin
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              OP_SET_SCALE: begin
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              OP_SET_ADDR_LO: begin
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              OP_SET_ADDR_HI: begin
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              // DMA: dispatched (non-blocking), pipeline continues
              OP_LOAD, OP_STORE: begin
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              // Systolic: dispatched (non-blocking)
              OP_MATMUL: begin
                if (!tile_valid) begin
                  fault_code_r <= 4'(FAULT_NO_CONFIG);
                  state        <= S_FAULT;
                end else begin
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end
              end

              // SFU: dispatched (non-blocking)
              OP_SOFTMAX, OP_LAYERNORM, OP_GELU: begin
                if (!tile_valid) begin
                  fault_code_r <= 4'(FAULT_NO_CONFIG);
                  state        <= S_FAULT;
                end else begin
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end
              end

              // Issue-stage ALU: stalls pipeline until done
              OP_REQUANT, OP_SCALE_MUL, OP_VADD: begin
                if (!tile_valid) begin
                  fault_code_r <= 4'(FAULT_NO_CONFIG);
                  state        <= S_FAULT;
                end else if (alu_busy) begin
                  state <= S_DISP_WAIT;
                end else begin
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end
              end

              OP_BUF_COPY: begin
                if (alu_busy) begin
                  state <= S_DISP_WAIT;
                end else begin
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end
              end

              default: begin
                fault_code_r <= 4'(FAULT_ILLEGAL_OP);
                state        <= S_FAULT;
              end
            endcase
          end
        end

        // ------------------------------------------------------------------
        S_SYNC_WAIT: begin
          if (dma_fault) begin
            fault_code_r <= dma_fault_code;
            state        <= S_FAULT;
          end else if (sync_clear_q) begin
            pc_reg <= pc_reg + 56'h1;
            state  <= S_FETCH;
          end
        end

        // ------------------------------------------------------------------
        S_DISP_WAIT: begin
          if (!alu_busy) begin
            pc_reg <= pc_reg + 56'h1;
            state  <= S_FETCH;
          end
        end

        // ------------------------------------------------------------------
        S_HALT, S_FAULT: ;  // terminal — held by done/fault outputs

        default: state <= S_IDLE;
      endcase
    end
  end

  // -------------------------------------------------------------------------
  // Combinational output logic
  // -------------------------------------------------------------------------
  always_comb begin
    // Defaults
    fetch_req    = 1'b0;
    done         = 1'b0;
    fault        = 1'b0;

    scale_we     = 1'b0;
    scale_waddr  = insn.s_sreg;
    scale_wdata  = insn.s_imm16;

    addr_lo_we   = 1'b0;
    addr_hi_we   = 1'b0;
    addr_wsel    = insn.a_addr_reg;
    addr_imm28   = insn.a_imm28;

    tile_we      = 1'b0;
    tile_m_in    = insn.c_tile_m;
    tile_n_in    = insn.c_tile_n;
    tile_k_in    = insn.c_tile_k;

    dma_dispatch = 1'b0;
    sys_dispatch = 1'b0;
    sfu_dispatch = 1'b0;
    alu_dispatch = 1'b0;

    case (state)
      S_FETCH: begin
        fetch_req = 1'b1;
      end

      S_ISSUE: begin
        if (!insn.illegal && !dma_fault) begin
          case (insn.opcode)
            OP_CONFIG_TILE:
              tile_we = 1'b1;

            OP_SET_SCALE:
              scale_we = (insn.s_src_mode == 2'b00);

            OP_SET_ADDR_LO:
              addr_lo_we = 1'b1;

            OP_SET_ADDR_HI:
              addr_hi_we = 1'b1;

            OP_LOAD, OP_STORE:
              dma_dispatch = 1'b1;

            OP_MATMUL:
              sys_dispatch = tile_valid;

            OP_SOFTMAX, OP_LAYERNORM, OP_GELU:
              sfu_dispatch = tile_valid;

            OP_REQUANT, OP_SCALE_MUL, OP_VADD, OP_BUF_COPY:
              alu_dispatch = !alu_busy &&
                             (insn.opcode == OP_BUF_COPY || tile_valid);

            default: ;
          endcase
        end
      end

      S_HALT:
        done = 1'b1;

      S_FAULT:
        fault = 1'b1;

      default: ;
    endcase
  end

endmodule

`endif // CONTROL_UNIT_SV
