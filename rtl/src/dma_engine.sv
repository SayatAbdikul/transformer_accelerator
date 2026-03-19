// DMA Engine — LOAD (DRAM→SRAM) and STORE (SRAM→DRAM) via AXI4 master.
//
// LOAD:  Issues one AXI4 multi-beat read burst (AR then R×xfer_len).
//        Each 16-byte beat is written to SRAM port A at consecutive rows.
//
// STORE: Issues an AXI4 write burst (AW, then W×xfer_len, then B).
//        SRAM port A is read 1 cycle ahead per beat (registered SRAM output).
//
// Effective DRAM byte address = base_addr + (dram_off × 16).
// OOB check: end_addr = dram_byte_addr + xfer_len×16; fault if > DRAM_SIZE.
//
// Phase 2 constraint: xfer_len ≤ 256 (fits in AXI4 8-bit burst length field).
// All parameters are latched on the dispatch pulse; insn fields may change after.

`ifndef DMA_ENGINE_SV
`define DMA_ENGINE_SV

`include "taccel_pkg.sv"

module dma_engine
  import taccel_pkg::*;
#(
  parameter int DRAM_SIZE = 1 << 24   // 16 MB default
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Dispatch (1-cycle pulse from control_unit) ---
  input  logic        dispatch,
  input  logic        is_store,        // 0=LOAD, 1=STORE
  input  logic [1:0]  buf_id,
  input  logic [15:0] sram_off,        // SRAM start row (16-byte units)
  input  logic [15:0] xfer_len,        // number of 16-byte beats (≤ 256)
  input  logic [55:0] base_addr,       // DRAM base address (from addr_reg)
  input  logic [15:0] dram_off,        // DRAM row offset (×16 = byte offset)

  // --- Status ---
  output logic        dma_busy,        // asserted while any state != IDLE
  output logic        dma_rd_busy,     // asserted during LOAD AR+R (for AXI arb)
  output logic        dma_fault,
  output logic [3:0]  dma_fault_code,

  // --- SRAM Port A ---
  output logic         sram_en,
  output logic         sram_we,
  output logic [1:0]   sram_buf,
  output logic [15:0]  sram_row,
  output logic [127:0] sram_wdata,
  input  logic [127:0] sram_rdata,     // valid 1 cycle after sram_en && !sram_we

  // --- AXI4 read channels (LOAD) ---
  output logic [AXI_ADDR_W-1:0] dma_ar_addr,
  output logic [7:0]             dma_ar_len,
  output logic                   dma_ar_valid,
  input  logic                   dma_ar_ready,
  input  logic [AXI_DATA_W-1:0]  dma_r_data,
  input  logic                   dma_r_valid,
  input  logic                   dma_r_last,
  output logic                   dma_r_ready,

  // --- AXI4 write channels (STORE) ---
  output logic [AXI_ADDR_W-1:0] dma_aw_addr,
  output logic [7:0]             dma_aw_len,
  output logic                   dma_aw_valid,
  input  logic                   dma_aw_ready,
  output logic [AXI_DATA_W-1:0]  dma_w_data,
  output logic [15:0]            dma_w_strb,
  output logic                   dma_w_valid,
  output logic                   dma_w_last,
  input  logic                   dma_w_ready,
  input  logic [1:0]             dma_b_resp,
  input  logic                   dma_b_valid,
  output logic                   dma_b_ready
);

  // -------------------------------------------------------------------------
  // FSM
  // -------------------------------------------------------------------------
  typedef enum logic [2:0] {
    D_IDLE           = 3'd0,
    D_LOAD_AR        = 3'd1,   // LOAD: issue AR (+ OOB check)
    D_LOAD_R         = 3'd2,   // LOAD: receive R beats, write SRAM
    D_STORE_AW       = 3'd3,   // STORE: issue AW (+ OOB check)
    D_STORE_SRAM_PRE = 3'd4,   // STORE: pre-read SRAM for current beat
    D_STORE_W        = 3'd5,   // STORE: drive W beat with sram_rdata
    D_STORE_B        = 3'd6,   // STORE: wait for B response
    D_FAULT          = 3'd7    // terminal fault
  } dma_state_t;

  dma_state_t  state;

  // Latched instruction parameters
  logic        is_store_q;
  logic [1:0]  buf_id_q;
  logic [15:0] sram_off_q;
  logic [15:0] xfer_len_q;
  logic [55:0] dram_byte_addr_q;  // base_addr + dram_off × 16
  logic [15:0] beat_cnt;
  logic [3:0]  fault_code_r;

  // OOB: end address (57-bit to detect overflow past 56-bit max)
  logic [56:0] dram_end;
  assign dram_end = {1'b0, dram_byte_addr_q} + {1'b0, xfer_len_q, 4'b0};
  logic oob;
  assign oob = (dram_end > 57'(DRAM_SIZE));

  // -------------------------------------------------------------------------
  // Sequential FSM
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state            <= D_IDLE;
      is_store_q       <= 1'b0;
      buf_id_q         <= 2'b00;
      sram_off_q       <= 16'h0;
      xfer_len_q       <= 16'h0;
      dram_byte_addr_q <= 56'h0;
      beat_cnt         <= 16'h0;
      fault_code_r     <= 4'h0;
    end else begin
      case (state)

        D_IDLE: begin
          if (dispatch) begin
            is_store_q       <= is_store;
            buf_id_q         <= buf_id;
            sram_off_q       <= sram_off;
            xfer_len_q       <= xfer_len;
            // dram_byte_addr = base_addr + dram_off × 16
            dram_byte_addr_q <= base_addr + ({40'h0, dram_off} << 4);
            beat_cnt         <= 16'h0;
            state            <= is_store ? D_STORE_AW : D_LOAD_AR;
          end
        end

        // ------------------------------------------------------------------
        // LOAD path
        D_LOAD_AR: begin
          if (oob) begin
            fault_code_r <= 4'(FAULT_DRAM_OOB);
            state        <= D_FAULT;
          end else if (dma_ar_ready) begin
            state <= D_LOAD_R;
          end
        end

        D_LOAD_R: begin
          if (dma_r_valid) begin
            beat_cnt <= beat_cnt + 16'h1;
            if (dma_r_last)
              state <= D_IDLE;
          end
        end

        // ------------------------------------------------------------------
        // STORE path
        D_STORE_AW: begin
          if (oob) begin
            fault_code_r <= 4'(FAULT_DRAM_OOB);
            state        <= D_FAULT;
          end else if (dma_aw_ready) begin
            state <= D_STORE_SRAM_PRE;
          end
        end

        D_STORE_SRAM_PRE: begin
          // sram_en=1, sram_we=0 driven combinationally below;
          // SRAM output will be valid on the next cycle (registered output).
          state <= D_STORE_W;
        end

        D_STORE_W: begin
          if (dma_w_ready) begin
            if (beat_cnt == xfer_len_q - 16'h1) begin
              state <= D_STORE_B;
            end else begin
              beat_cnt <= beat_cnt + 16'h1;
              state    <= D_STORE_SRAM_PRE;
            end
          end
        end

        D_STORE_B: begin
          if (dma_b_valid)
            state <= D_IDLE;
        end

        D_FAULT: ;  // terminal — cleared only by reset

        default: state <= D_IDLE;
      endcase
    end
  end

  // -------------------------------------------------------------------------
  // Combinational outputs
  // -------------------------------------------------------------------------
  always_comb begin
    dma_busy       = (state != D_IDLE);
    dma_rd_busy    = (state == D_LOAD_AR || state == D_LOAD_R);
    dma_fault      = (state == D_FAULT);
    dma_fault_code = fault_code_r;

    // AXI read defaults
    dma_ar_addr  = dram_byte_addr_q;
    dma_ar_len   = 8'(xfer_len_q - 16'h1);
    dma_ar_valid = 1'b0;
    dma_r_ready  = 1'b0;

    // AXI write defaults
    dma_aw_addr  = dram_byte_addr_q;
    dma_aw_len   = 8'(xfer_len_q - 16'h1);
    dma_aw_valid = 1'b0;
    dma_w_data   = 128'h0;
    dma_w_strb   = 16'hFFFF;  // all 16 bytes valid
    dma_w_valid  = 1'b0;
    dma_w_last   = 1'b0;
    dma_b_ready  = 1'b0;

    // SRAM defaults
    sram_en    = 1'b0;
    sram_we    = 1'b0;
    sram_buf   = buf_id_q;
    sram_row   = sram_off_q + beat_cnt;
    sram_wdata = dma_r_data;

    case (state)
      D_LOAD_AR: begin
        if (!oob) dma_ar_valid = 1'b1;
      end

      D_LOAD_R: begin
        dma_r_ready = 1'b1;
        if (dma_r_valid) begin
          sram_en    = 1'b1;
          sram_we    = 1'b1;
          sram_wdata = dma_r_data;
        end
      end

      D_STORE_AW: begin
        if (!oob) dma_aw_valid = 1'b1;
      end

      D_STORE_SRAM_PRE: begin
        sram_en = 1'b1;
        sram_we = 1'b0;
      end

      D_STORE_W: begin
        dma_w_valid = 1'b1;
        dma_w_data  = sram_rdata;
        dma_w_last  = (beat_cnt == xfer_len_q - 16'h1);
      end

      D_STORE_B: begin
        dma_b_ready = 1'b1;
      end

      default: ;
    endcase
  end

endmodule

`endif // DMA_ENGINE_SV
