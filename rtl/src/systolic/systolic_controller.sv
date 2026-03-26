`ifndef SYSTOLIC_CONTROLLER_SV
`define SYSTOLIC_CONTROLLER_SV

`include "taccel_pkg.sv"

module systolic_controller
  import taccel_pkg::*;
(
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 dispatch,

  input  logic [9:0]           tile_m,
  input  logic [9:0]           tile_n,
  input  logic [9:0]           tile_k,

  input  logic [1:0]           src1_buf,
  input  logic [15:0]          src1_off,
  input  logic [1:0]           src2_buf,
  input  logic [15:0]          src2_off,
  input  logic [1:0]           dst_buf,
  input  logic [15:0]          dst_off,
  input  logic                 flags_accumulate,

  output logic                 sys_busy,

  // SRAM port A (read/write): src2 read and dst writeback
  output logic                 sram_a_en,
  output logic                 sram_a_we,
  output logic [1:0]           sram_a_buf,
  output logic [15:0]          sram_a_row,
  output logic [127:0]         sram_a_wdata,
  input  logic [127:0]         sram_a_rdata,

  // SRAM port B (read-only): src1 read
  output logic                 sram_b_en,
  output logic [1:0]           sram_b_buf,
  output logic [15:0]          sram_b_row,
  input  logic [127:0]         sram_b_rdata
);

  typedef enum logic [3:0] {
    ST_IDLE       = 4'd0,
    ST_INIT_TILE  = 4'd1,
    ST_READ_REQ   = 4'd2,
    ST_READ_USE   = 4'd3,
    ST_DRAIN_PREP = 4'd4,
    ST_DRAIN_WR   = 4'd5
  } state_t;

  state_t state;

  logic [1:0]  src1_buf_q, src2_buf_q, dst_buf_q;
  logic [15:0] src1_off_q, src2_off_q, dst_off_q;
  logic        flags_accumulate_q;

  logic [10:0] m_tiles_q, n_tiles_q, k_tiles_q;
  logic [10:0] mtile_q, ntile_q, ktile_q;
  logic [5:0]  lane_q;
  logic [4:0]  drain_row_q;
  logic [1:0]  drain_grp_q;

  logic step_en;
  logic clear_acc;
  logic [0:0] arch_mode_q;
  logic       inject_zero_data;
  logic [15:0] lane_row_idx;
  logic [127:0] a_row_data_q, b_row_data_q;
  logic [SYS_DIM*SYS_DIM*32-1:0] acc_flat;

  localparam int CHAIN_PRELOAD_CYCLES = 1;
  localparam int CHAIN_FLUSH_CYCLES = (2 * (SYS_DIM - 1));
  localparam int CHAIN_TOTAL_STEPS  = CHAIN_PRELOAD_CYCLES + SYS_DIM + CHAIN_FLUSH_CYCLES;

  // Architecture default comes from package constant (currently broadcast).
  assign arch_mode_q = SYS_MODE_DEFAULT[0:0];

  systolic_array u_array (
    .clk      (clk),
    .rst_n    (rst_n),
    .step_en  (step_en),
    .clear_acc(clear_acc),
    .a_row_data(a_row_data_q),
    .b_row_data(b_row_data_q),
    .arch_mode(arch_mode_q),
    .acc_flat (acc_flat)
  );

  function automatic logic [31:0] acc_at(
    input logic [4:0] r,
    input logic [4:0] c
  );
    int idx;
    begin
      idx = ((int'(r) * SYS_DIM) + int'(c)) * 32;
      acc_at = acc_flat[idx +: 32];
    end
  endfunction

  logic [31:0] src1_tile_base, src2_tile_base, dst_tile_base;
  always_comb begin
    src1_tile_base = {16'h0, src1_off_q} + (((mtile_q * k_tiles_q) + {21'h0, ktile_q}) << 4);
    src2_tile_base = {16'h0, src2_off_q} + (((ktile_q * n_tiles_q) + {21'h0, ntile_q}) << 4);
    dst_tile_base  = {16'h0, dst_off_q}  + (((mtile_q * n_tiles_q) + {21'h0, ntile_q}) << 6);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      src1_buf_q <= 2'b0;
      src2_buf_q <= 2'b0;
      dst_buf_q <= 2'b0;
      src1_off_q <= 16'h0;
      src2_off_q <= 16'h0;
      dst_off_q <= 16'h0;
      flags_accumulate_q <= 1'b0;
      m_tiles_q <= 11'd0;
      n_tiles_q <= 11'd0;
      k_tiles_q <= 11'd0;
      mtile_q <= 11'd0;
      ntile_q <= 11'd0;
      ktile_q <= 11'd0;
      lane_q <= 6'd0;
      drain_row_q <= 5'd0;
      drain_grp_q <= 2'd0;
    end else begin
      case (state)
        ST_IDLE: begin
          if (dispatch) begin
            src1_buf_q <= src1_buf;
            src2_buf_q <= src2_buf;
            dst_buf_q <= dst_buf;
            src1_off_q <= src1_off;
            src2_off_q <= src2_off;
            dst_off_q <= dst_off;
            flags_accumulate_q <= flags_accumulate;
            m_tiles_q <= {1'b0, tile_m} + 11'd1;
            n_tiles_q <= {1'b0, tile_n} + 11'd1;
            k_tiles_q <= {1'b0, tile_k} + 11'd1;
            mtile_q <= 11'd0;
            ntile_q <= 11'd0;
            ktile_q <= 11'd0;
            lane_q <= 6'd0;
            state <= ST_INIT_TILE;
          end
        end

        ST_INIT_TILE: begin
          lane_q <= 6'd0;
          state <= ST_READ_REQ;
        end

        ST_READ_REQ: begin
          state <= ST_READ_USE;
        end

        ST_READ_USE: begin
          if (int'(lane_q) == ((arch_mode_q == SYS_MODE_CHAINED[0:0]) ? (CHAIN_TOTAL_STEPS - 1) : (SYS_DIM - 1))) begin
            lane_q <= 6'd0;
            if (ktile_q + 11'd1 < k_tiles_q) begin
              ktile_q <= ktile_q + 11'd1;
              state <= ST_READ_REQ;
            end else begin
              drain_row_q <= 5'd0;
              drain_grp_q <= 2'd0;
              state <= ST_DRAIN_PREP;
            end
          end else begin
            lane_q <= lane_q + 6'd1;
            state <= ST_READ_REQ;
          end
        end

        ST_DRAIN_PREP: begin
          state <= ST_DRAIN_WR;
        end

        ST_DRAIN_WR: begin
          if (drain_grp_q == 2'd3) begin
            drain_grp_q <= 2'd0;
            if (drain_row_q == 5'd15) begin
              // Tile complete. Move to next n/m tile.
              ktile_q <= 11'd0;
              if (ntile_q + 11'd1 < n_tiles_q) begin
                ntile_q <= ntile_q + 11'd1;
                state <= ST_INIT_TILE;
              end else if (mtile_q + 11'd1 < m_tiles_q) begin
                mtile_q <= mtile_q + 11'd1;
                ntile_q <= 11'd0;
                state <= ST_INIT_TILE;
              end else begin
                state <= ST_IDLE;
              end
            end else begin
              drain_row_q <= drain_row_q + 5'd1;
            end
          end else begin
            drain_grp_q <= drain_grp_q + 2'd1;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

  always_comb begin
    sys_busy = (state != ST_IDLE);

    // During chained flush cycles, inject zeros so only in-flight operands
    // continue propagating through the PE mesh.
    inject_zero_data = (arch_mode_q == SYS_MODE_CHAINED[0:0])
                    && ((lane_q == 6'd0) || (int'(lane_q) > SYS_DIM))
                    && ((state == ST_READ_REQ) || (state == ST_READ_USE));
    lane_row_idx = (arch_mode_q == SYS_MODE_CHAINED[0:0])
               ? ({10'h0, lane_q[5:0]} - 16'd1)
               : {10'h0, lane_q[5:0]};
    a_row_data_q = inject_zero_data ? 128'h0 : sram_b_rdata;
    b_row_data_q = inject_zero_data ? 128'h0 : sram_a_rdata;

    sram_a_en = 1'b0;
    sram_a_we = 1'b0;
    sram_a_buf = src2_buf_q;
    sram_a_row = 16'h0;
    sram_a_wdata = 128'h0;

    sram_b_en = 1'b0;
    sram_b_buf = src1_buf_q;
    sram_b_row = 16'h0;

    step_en = 1'b0;
    // Clear accumulator at tile start unless accumulate mode is requested.
    clear_acc = (state == ST_INIT_TILE) && !flags_accumulate_q;

    case (state)
      ST_READ_REQ: begin
        if (inject_zero_data) begin
          sram_b_en = 1'b0;
          sram_a_en = 1'b0;
        end else begin
          sram_b_en = 1'b1;
          sram_b_buf = src1_buf_q;
          sram_b_row = src1_tile_base[15:0] + lane_row_idx;

          sram_a_en = 1'b1;
          sram_a_we = 1'b0;
          sram_a_buf = src2_buf_q;
          sram_a_row = src2_tile_base[15:0] + lane_row_idx;
        end
      end

      ST_READ_USE: begin
        step_en = 1'b1;
      end

      ST_DRAIN_WR: begin
        logic [4:0] c0, c1, c2, c3;
        c0 = {1'b0, drain_grp_q, 2'b00};
        c1 = c0 + 5'd1;
        c2 = c0 + 5'd2;
        c3 = c0 + 5'd3;

        sram_a_en = 1'b1;
        sram_a_we = 1'b1;
        sram_a_buf = dst_buf_q;
        sram_a_row = dst_tile_base[15:0] + ({11'd0, drain_row_q} << 2) + {14'h0, drain_grp_q};
        sram_a_wdata[31:0]   = acc_at(drain_row_q, c0);
        sram_a_wdata[63:32]  = acc_at(drain_row_q, c1);
        sram_a_wdata[95:64]  = acc_at(drain_row_q, c2);
        sram_a_wdata[127:96] = acc_at(drain_row_q, c3);
      end

      default: ;
    endcase
  end

endmodule

`endif // SYSTOLIC_CONTROLLER_SV
