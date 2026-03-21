`ifndef SYSTOLIC_PE_SV
`define SYSTOLIC_PE_SV

module systolic_pe (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        en,
  input  logic        acc_clear,
  input  logic [7:0]  a_in,
  input  logic [7:0]  b_in,
  output logic [7:0]  a_out,
  output logic [7:0]  b_out,
  output logic [31:0] acc
);

  logic signed [7:0]  a_s;
  logic signed [7:0]  b_s;
  logic signed [15:0] prod_s;

  assign a_s = a_in;
  assign b_s = b_in;
  assign prod_s = a_s * b_s;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out <= 8'h0;
      b_out <= 8'h0;
      acc   <= 32'h0;
    end else begin
      a_out <= a_in;
      b_out <= b_in;
      if (acc_clear)
        acc <= 32'h0;
      else if (en)
        acc <= $signed(acc) + $signed({{16{prod_s[15]}}, prod_s});
    end
  end

endmodule

`endif // SYSTOLIC_PE_SV
