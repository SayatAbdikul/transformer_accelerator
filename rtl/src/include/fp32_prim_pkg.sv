// Non-DPI FP32 primitives used while migrating the accelerator away from C
// math helpers. This first slice keeps the existing real-valued engine
// boundaries, but implements FP32 round/add/sub/mul in SystemVerilog by
// explicitly rounding IEEE-754 double-precision real values into packed
// single-precision bits with round-to-nearest-even semantics.

`ifndef FP32_PRIM_PKG_SV
`define FP32_PRIM_PKG_SV

package fp32_prim_pkg;
  typedef logic [31:0] fp32_t;
  typedef logic [63:0] fp64_t;

  localparam fp32_t FP32_QNAN_BITS = 32'h7fc0_0000;
  localparam fp32_t FP32_POS_INF   = 32'h7f80_0000;
  localparam fp32_t FP32_NEG_INF   = 32'hff80_0000;

  function automatic bit fp32_is_nan(input fp32_t bits);
    fp32_is_nan = (bits[30:23] == 8'hff) && (bits[22:0] != 23'd0);
  endfunction

  function automatic bit fp32_is_inf(input fp32_t bits);
    fp32_is_inf = (bits[30:23] == 8'hff) && (bits[22:0] == 23'd0);
  endfunction

  function automatic bit fp32_is_zero(input fp32_t bits);
    fp32_is_zero = (bits[30:0] == 31'd0);
  endfunction

  function automatic real fp32_pow2(input int exp);
    real value_r;
    int i;
    begin
      value_r = 1.0;
      if (exp >= 0) begin
        for (i = 0; i < exp; i++) begin
          value_r = value_r * 2.0;
        end
      end else begin
        for (i = 0; i < -exp; i++) begin
          value_r = value_r * 0.5;
        end
      end
      fp32_pow2 = value_r;
    end
  endfunction

  function automatic longint unsigned fp32_round_shift_right(
      input longint unsigned value,
      input int shift
  );
    longint unsigned quotient;
    longint unsigned remainder;
    longint unsigned half;
    longint unsigned mask;
    begin
      if (shift <= 0) begin
        fp32_round_shift_right = value << (-shift);
      end else if (shift >= 63) begin
        quotient = 0;
        remainder = value;
        half = 64'h8000_0000_0000_0000;
        fp32_round_shift_right =
            (remainder > half || (remainder == half && quotient[0])) ? 64'd1 : 64'd0;
      end else begin
        quotient = value >> shift;
        mask = (64'd1 << shift) - 64'd1;
        remainder = value & mask;
        half = 64'd1 << (shift - 1);
        if (remainder > half || (remainder == half && quotient[0])) begin
          fp32_round_shift_right = quotient + 64'd1;
        end else begin
          fp32_round_shift_right = quotient;
        end
      end
    end
  endfunction

  function automatic fp32_t fp32_from_fp64_bits(input fp64_t bits64);
    bit sign_b;
    int exp64;
    int exp_unbiased;
    int exp32;
    longint unsigned mant53;
    longint unsigned rounded;
    int sub_shift;
    begin
      sign_b = bits64[63];
      exp64 = int'(bits64[62:52]);
      mant53 = {11'd0, 1'b1, bits64[51:0]};

      if (exp64 == 2047) begin
        if (bits64[51:0] != 52'd0) begin
          fp32_from_fp64_bits = FP32_QNAN_BITS;
        end else begin
          fp32_from_fp64_bits = sign_b ? FP32_NEG_INF : FP32_POS_INF;
        end
      end else if (exp64 == 0) begin
        // Double subnormals are far below FP32 range for the values produced by
        // this accelerator path. Preserve the sign of underflowed zero.
        fp32_from_fp64_bits = {sign_b, 31'd0};
      end else begin
        exp_unbiased = exp64 - 1023;
        if (exp_unbiased > 127) begin
          fp32_from_fp64_bits = sign_b ? FP32_NEG_INF : FP32_POS_INF;
        end else if (exp_unbiased >= -126) begin
          rounded = fp32_round_shift_right(mant53, 29); // 53-bit mantissa -> 24 bits
          if (rounded >= 64'h0000_0000_0100_0000) begin
            rounded = 64'h0000_0000_0080_0000;
            exp_unbiased++;
          end
          exp32 = exp_unbiased + 127;
          if (exp32 >= 255) begin
            fp32_from_fp64_bits = sign_b ? FP32_NEG_INF : FP32_POS_INF;
          end else begin
            fp32_from_fp64_bits = {sign_b, exp32[7:0], rounded[22:0]};
          end
        end else begin
          // FP32 subnormal: mantissa = round(value / 2^-149).
          sub_shift = -(exp_unbiased + 97);
          rounded = fp32_round_shift_right(mant53, sub_shift);
          if (rounded == 0) begin
            fp32_from_fp64_bits = {sign_b, 31'd0};
          end else if (rounded >= 64'h0000_0000_0080_0000) begin
            fp32_from_fp64_bits = {sign_b, 8'd1, 23'd0};
          end else begin
            fp32_from_fp64_bits = {sign_b, 8'd0, rounded[22:0]};
          end
        end
      end
    end
  endfunction

  function automatic fp32_t fp32_real_to_bits(input real value_r);
    fp32_real_to_bits = fp32_from_fp64_bits($realtobits(value_r));
  endfunction

  function automatic real fp32_bits_to_real(input fp32_t bits);
    int exp32;
    int unsigned frac;
    real mag_r;
    real frac_r;
    fp64_t zero_bits;
    begin
      exp32 = int'(bits[30:23]);
      frac = int'(bits[22:0]);
      if (exp32 == 255) begin
        if (frac != 0) begin
          fp32_bits_to_real = $bitstoreal(64'h7ff8_0000_0000_0000);
        end else begin
          fp32_bits_to_real = bits[31]
              ? $bitstoreal(64'hfff0_0000_0000_0000)
              : $bitstoreal(64'h7ff0_0000_0000_0000);
        end
      end else if (exp32 == 0 && frac == 0) begin
        zero_bits = bits[31] ? 64'h8000_0000_0000_0000 : 64'h0000_0000_0000_0000;
        fp32_bits_to_real = $bitstoreal(zero_bits);
      end else begin
        if (exp32 == 0) begin
          frac_r = real'(frac) / 8388608.0;
          mag_r = frac_r * fp32_pow2(-126);
        end else begin
          frac_r = real'(frac) / 8388608.0;
          mag_r = (1.0 + frac_r) * fp32_pow2(exp32 - 127);
        end
        fp32_bits_to_real = bits[31] ? -mag_r : mag_r;
      end
    end
  endfunction

  function automatic longint unsigned fp32_shift_right_sticky(
      input longint unsigned value,
      input int shift
  );
    longint unsigned shifted;
    longint unsigned mask;
    begin
      if (shift <= 0) begin
        fp32_shift_right_sticky = value << (-shift);
      end else if (shift >= 64) begin
        fp32_shift_right_sticky = (value != 64'd0) ? 64'd1 : 64'd0;
      end else begin
        shifted = value >> shift;
        mask = (64'd1 << shift) - 64'd1;
        fp32_shift_right_sticky = shifted | (((value & mask) != 64'd0) ? 64'd1 : 64'd0);
      end
    end
  endfunction

  function automatic int fp32_msb_index(input longint unsigned value);
    int idx;
    begin
      fp32_msb_index = -1;
      for (idx = 63; idx >= 0; idx--) begin
        if (value[idx] && fp32_msb_index < 0) begin
          fp32_msb_index = idx;
        end
      end
    end
  endfunction

  function automatic fp32_t fp32_pack_rounded(
      input bit sign_b,
      input int exp_unbiased,
      input longint unsigned ext_sig
  );
    longint unsigned mant24_u;
    logic [22:0] frac_bits;
    logic [7:0] exp_bits;
    bit guard_b;
    bit round_b;
    bit sticky_b;
    begin
      if (ext_sig == 64'd0) begin
        fp32_pack_rounded = {sign_b, 31'd0};
      end else begin
        while ((exp_unbiased > -126) && (ext_sig < 64'h0000_0000_0400_0000)) begin
          ext_sig = ext_sig << 1;
          exp_unbiased--;
        end

        if (exp_unbiased < -126) begin
          ext_sig = fp32_shift_right_sticky(ext_sig, -126 - exp_unbiased);
          exp_unbiased = -126;
        end

        guard_b = ext_sig[2];
        round_b = ext_sig[1];
        sticky_b = ext_sig[0];
        mant24_u = ext_sig >> 3;
        if (guard_b && (round_b || sticky_b || mant24_u[0])) begin
          mant24_u = mant24_u + 64'd1;
        end

        if (mant24_u == 64'd0) begin
          fp32_pack_rounded = {sign_b, 31'd0};
        end else begin
          if (mant24_u >= 64'h0000_0000_0100_0000) begin
            mant24_u = fp32_shift_right_sticky(mant24_u, 1);
            exp_unbiased++;
          end

          if (exp_unbiased > 127) begin
            fp32_pack_rounded = sign_b ? FP32_NEG_INF : FP32_POS_INF;
          end else if (exp_unbiased <= -126) begin
            if (mant24_u >= 64'h0000_0000_0080_0000) begin
              exp_bits = 8'd1;
              frac_bits = mant24_u[22:0];
              fp32_pack_rounded = {sign_b, exp_bits, frac_bits};
            end else begin
              frac_bits = mant24_u[22:0];
              fp32_pack_rounded = {sign_b, 8'd0, frac_bits};
            end
          end else begin
            exp_bits = 8'(exp_unbiased + 127);
            frac_bits = mant24_u[22:0];
            fp32_pack_rounded = {sign_b, exp_bits, frac_bits};
          end
        end
      end
    end
  endfunction

  function automatic void fp32_decode_finite(
      input fp32_t bits,
      output bit sign_b,
      output int exp_unbiased,
      output longint unsigned mant24
  );
    begin
      sign_b = bits[31];
      if (bits[30:23] == 8'd0) begin
        exp_unbiased = -126;
        mant24 = {41'd0, bits[22:0]};
      end else begin
        exp_unbiased = int'(bits[30:23]) - 127;
        mant24 = {40'd0, 1'b1, bits[22:0]};
      end
    end
  endfunction

  function automatic fp32_t fp32_addsub_bits(
      input fp32_t lhs_bits,
      input fp32_t rhs_bits,
      input bit subtract_b
  );
    fp32_t rhs_eff_bits;
    bit sign_a;
    bit sign_b;
    int exp_a;
    int exp_b;
    int exp_r;
    longint unsigned mant_a;
    longint unsigned mant_b;
    longint unsigned ext_a;
    longint unsigned ext_b;
    longint unsigned mag_r;
    bit sign_r;
    begin
      rhs_eff_bits = rhs_bits ^ {subtract_b, 31'd0};

      if (fp32_is_nan(lhs_bits) || fp32_is_nan(rhs_eff_bits)) begin
        fp32_addsub_bits = FP32_QNAN_BITS;
      end else if (fp32_is_inf(lhs_bits) && fp32_is_inf(rhs_eff_bits)) begin
        fp32_addsub_bits = (lhs_bits[31] == rhs_eff_bits[31])
            ? {lhs_bits[31], 8'hff, 23'd0}
            : FP32_QNAN_BITS;
      end else if (fp32_is_inf(lhs_bits)) begin
        fp32_addsub_bits = {lhs_bits[31], 8'hff, 23'd0};
      end else if (fp32_is_inf(rhs_eff_bits)) begin
        fp32_addsub_bits = {rhs_eff_bits[31], 8'hff, 23'd0};
      end else if (fp32_is_zero(lhs_bits) && fp32_is_zero(rhs_eff_bits)) begin
        fp32_addsub_bits = {
          (lhs_bits[31] && rhs_eff_bits[31]),
          31'd0
        };
      end else begin
        fp32_decode_finite(lhs_bits, sign_a, exp_a, mant_a);
        fp32_decode_finite(rhs_eff_bits, sign_b, exp_b, mant_b);
        ext_a = mant_a << 3;
        ext_b = mant_b << 3;

        if (exp_a > exp_b) begin
          ext_b = fp32_shift_right_sticky(ext_b, exp_a - exp_b);
          exp_r = exp_a;
        end else if (exp_b > exp_a) begin
          ext_a = fp32_shift_right_sticky(ext_a, exp_b - exp_a);
          exp_r = exp_b;
        end else begin
          exp_r = exp_a;
        end

        if (sign_a == sign_b) begin
          mag_r = ext_a + ext_b;
          sign_r = sign_a;
          if (mag_r >= 64'h0000_0000_0800_0000) begin
            mag_r = fp32_shift_right_sticky(mag_r, 1);
            exp_r++;
          end
        end else if (ext_a > ext_b) begin
          mag_r = ext_a - ext_b;
          sign_r = sign_a;
        end else if (ext_b > ext_a) begin
          mag_r = ext_b - ext_a;
          sign_r = sign_b;
        end else begin
          mag_r = 64'd0;
          sign_r = 1'b0;
        end

        fp32_addsub_bits = fp32_pack_rounded(sign_r, exp_r, mag_r);
      end
    end
  endfunction

  function automatic fp32_t fp32_round_bits(input fp32_t value_bits);
    if (fp32_is_nan(value_bits)) begin
      fp32_round_bits = FP32_QNAN_BITS;
    end else begin
      fp32_round_bits = value_bits;
    end
  endfunction

  function automatic fp32_t fp32_add_bits(input fp32_t lhs_bits, input fp32_t rhs_bits);
    fp32_add_bits = fp32_addsub_bits(lhs_bits, rhs_bits, 1'b0);
  endfunction

  function automatic fp32_t fp32_sub_bits(input fp32_t lhs_bits, input fp32_t rhs_bits);
    fp32_sub_bits = fp32_addsub_bits(lhs_bits, rhs_bits, 1'b1);
  endfunction

  function automatic fp32_t fp32_mul_bits(input fp32_t lhs_bits, input fp32_t rhs_bits);
    bit sign_a;
    bit sign_b;
    bit sign_r;
    int exp_a;
    int exp_b;
    int exp_r;
    int msb_idx;
    longint unsigned mant_a;
    longint unsigned mant_b;
    longint unsigned product;
    longint unsigned ext_sig;
    begin
      sign_r = lhs_bits[31] ^ rhs_bits[31];

      if (fp32_is_nan(lhs_bits) || fp32_is_nan(rhs_bits)) begin
        fp32_mul_bits = FP32_QNAN_BITS;
      end else if ((fp32_is_inf(lhs_bits) && fp32_is_zero(rhs_bits))
          || (fp32_is_zero(lhs_bits) && fp32_is_inf(rhs_bits))) begin
        fp32_mul_bits = FP32_QNAN_BITS;
      end else if (fp32_is_inf(lhs_bits) || fp32_is_inf(rhs_bits)) begin
        fp32_mul_bits = sign_r ? FP32_NEG_INF : FP32_POS_INF;
      end else if (fp32_is_zero(lhs_bits) || fp32_is_zero(rhs_bits)) begin
        fp32_mul_bits = {sign_r, 31'd0};
      end else begin
        fp32_decode_finite(lhs_bits, sign_a, exp_a, mant_a);
        fp32_decode_finite(rhs_bits, sign_b, exp_b, mant_b);
        sign_r = sign_a ^ sign_b;
        product = mant_a * mant_b;
        msb_idx = fp32_msb_index(product);
        exp_r = exp_a + exp_b - 46 + msb_idx;
        if (msb_idx > 26) begin
          ext_sig = fp32_shift_right_sticky(product, msb_idx - 26);
        end else begin
          ext_sig = product << (26 - msb_idx);
        end
        fp32_mul_bits = fp32_pack_rounded(sign_r, exp_r, ext_sig);
      end
    end
  endfunction

  function automatic real fp32_round_real(input real value_r);
    fp32_round_real = fp32_bits_to_real(fp32_real_to_bits(value_r));
  endfunction

  function automatic real fp32_add_real(input real lhs_r, input real rhs_r);
    fp32_add_real = fp32_bits_to_real(
        fp32_add_bits(fp32_real_to_bits(lhs_r), fp32_real_to_bits(rhs_r)));
  endfunction

  function automatic real fp32_sub_real(input real lhs_r, input real rhs_r);
    fp32_sub_real = fp32_bits_to_real(
        fp32_sub_bits(fp32_real_to_bits(lhs_r), fp32_real_to_bits(rhs_r)));
  endfunction

  function automatic real fp32_mul_real(input real lhs_r, input real rhs_r);
    fp32_mul_real = fp32_bits_to_real(
        fp32_mul_bits(fp32_real_to_bits(lhs_r), fp32_real_to_bits(rhs_r)));
  endfunction

endpackage

`endif
