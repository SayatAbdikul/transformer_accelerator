`ifndef SYSTOLIC_ARRAY_SV
`define SYSTOLIC_ARRAY_SV

`include "taccel_pkg.sv"

module systolic_array
  import taccel_pkg::*;
(
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         step_en,
  input  logic                         clear_acc,
  input  logic [AXI_DATA_W-1:0]        a_row_data,
  input  logic [AXI_DATA_W-1:0]        b_row_data,
  output logic [SYS_DIM*SYS_DIM*32-1:0] acc_flat
);

  logic [7:0] a_vec [0:SYS_DIM-1];
  logic [7:0] b_vec [0:SYS_DIM-1];

  logic [31:0] pe_acc [0:SYS_DIM-1][0:SYS_DIM-1];

  genvar i, j;
  generate
    for (i = 0; i < SYS_DIM; i++) begin : GEN_A_B
      assign a_vec[i] = a_row_data[i*8 +: 8];
      assign b_vec[i] = b_row_data[i*8 +: 8];
    end
  endgenerate

  generate
    for (i = 0; i < SYS_DIM; i++) begin : GEN_ROW
      for (j = 0; j < SYS_DIM; j++) begin : GEN_COL
        logic [7:0] a_out_unused;
        logic [7:0] b_out_unused;
        systolic_pe u_pe (
          .clk       (clk),
          .rst_n     (rst_n),
          .en        (step_en),
          .acc_clear (clear_acc),
          .a_in      (a_vec[i]),
          .b_in      (b_vec[j]),
          .a_out     (a_out_unused),
          .b_out     (b_out_unused),
          .acc       (pe_acc[i][j])
        );

        localparam int FLAT = (i * SYS_DIM + j) * 32;
        assign acc_flat[FLAT +: 32] = pe_acc[i][j];
      end
    end
  endgenerate

endmodule

`endif // SYSTOLIC_ARRAY_SV
