`ifndef TB_FP32_PRIM_SV
`define TB_FP32_PRIM_SV

module tb_fp32_prim
  import fp32_prim_pkg::*;
(
  input  logic [1:0]  op,
  input  logic [31:0] a_bits,
  input  logic [31:0] b_bits,
  input  real         a_real,
  input  real         b_real,
  output logic [31:0] result_bits,
  output logic [31:0] real_round_bits,
  output logic [31:0] real_add_bits,
  output logic [31:0] real_sub_bits,
  output logic [31:0] real_mul_bits
);

  always_comb begin
    unique case (op)
      2'd0: result_bits = fp32_round_bits(a_bits);
      2'd1: result_bits = fp32_add_bits(a_bits, b_bits);
      2'd2: result_bits = fp32_sub_bits(a_bits, b_bits);
      2'd3: result_bits = fp32_mul_bits(a_bits, b_bits);
      default: result_bits = FP32_QNAN_BITS;
    endcase
  end

  always_comb begin
    real_round_bits = fp32_real_to_bits(fp32_round_real(a_real));
    real_add_bits = fp32_real_to_bits(fp32_add_real(a_real, b_real));
    real_sub_bits = fp32_real_to_bits(fp32_sub_real(a_real, b_real));
    real_mul_bits = fp32_real_to_bits(fp32_mul_real(a_real, b_real));
  end

endmodule

`endif
