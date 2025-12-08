`ifdef SIMULATION
`timescale 1ns / 1ps
`endif

//////////////////////////////////////////////////////////////
// FP32 Multiplier (IEEE 754)
// Format: 1-bit sign, 8-bit exponent, 23-bit fraction
/////////////////////////////////////////////////////////////

module fp32_mult(
  input clk,
  input rst_n,
  input  [31:0] a,
  input  [31:0] b,
  output reg [31:0] result
);

  wire a_sign = a[31];
  wire b_sign = b[31];
  wire [7:0] a_exp = a[30:23];
  wire [7:0] b_exp = b[30:23];
  wire [23:0] a_frac = (a_exp == 0) ? {1'b0, a[22:0]} : {1'b1, a[22:0]};
  wire [23:0] b_frac = (b_exp == 0) ? {1'b0, b[22:0]} : {1'b1, b[22:0]};

  wire sign_out = a_sign ^ b_sign;
  wire [47:0] raw_prod = a_frac * b_frac;
  wire [9:0] exp_sum = a_exp + b_exp - 8'd127;

  wire [7:0] norm_exp;
  wire [47:0] norm_prod;
  wire [22:0] mantissa;

  fp32_multiplication_normaliser normaliser (
    .in_e(exp_sum),
    .in_m(raw_prod),
    .out_e(norm_exp),
    .out_m(norm_prod)
  );

  assign mantissa = norm_prod[46:24]; // Top 23 bits after normalization

  always @(*) begin
    // NaN check
    if ((a_exp == 8'hFF && a[22:0] != 0) || (b_exp == 8'hFF && b[22:0] != 0)) begin
      result = {1'b0, 8'hFF, 23'h400000}; // quiet NaN
    end
    // Zero check
    else if ((a_exp == 0 && a[22:0] == 0) || (b_exp == 0 && b[22:0] == 0)) begin
      result = {sign_out, 31'b0}; // signed zero
    end
    // Infinity
    else if ((a_exp == 8'hFF && a[22:0] == 0) || (b_exp == 8'hFF && b[22:0] == 0)) begin
      result = {sign_out, 8'hFF, 23'b0}; // signed infinity
    end
    // Normal case
    else begin
      result = {sign_out, norm_exp, mantissa};
    end
  end
endmodule

module fp32_multiplication_normaliser (
  input  [9:0] in_e,
  input  [47:0] in_m,
  output reg [7:0] out_e,
  output reg [47:0] out_m
);

  always @(*) begin
    if (in_m[47]) begin
      out_e = in_e + 1;
      out_m = in_m >> 1;
    end else if (in_m[46]) begin
      out_e = in_e;
      out_m = in_m;
    end else if (in_m[45]) begin
      out_e = in_e - 1; out_m = in_m << 1;
    end else if (in_m[44]) begin
      out_e = in_e - 2; out_m = in_m << 2;
    end else if (in_m[43]) begin
      out_e = in_e - 3; out_m = in_m << 3;
    end else if (in_m[42]) begin
      out_e = in_e - 4; out_m = in_m << 4;
    end else begin
      out_e = 0;
      out_m = 0;
    end
  end
endmodule

// module fp32_mult(clk, rst_n, a, b, result);
//   input clk;
//   input rst_n;
//   input [31:0] a, b;
//   output reg [31:0] result;

//   // Intermediate variables
//   reg sign_a, sign_b, sign_res;
//   reg [7:0] exp_a, exp_b, exp_res;
//   reg [23:0] mant_a, mant_b;
//   reg [47:0] mant_prod;
//   reg [22:0] mant_res;
//   reg [7:0] exp_sum;

//   always @(posedge clk) begin
//     // Decompose inputs
//     sign_a = a[31];
//     sign_b = b[31];
//     exp_a = (a[30:23] == 8'd0) ? 8'd1 : a[30:23];  // treat denormals as exponent 1
//     exp_b = (b[30:23] == 8'd0) ? 8'd1 : b[30:23];
//     mant_a = (a[30:23] == 8'd0) ? {1'b0, a[22:0]} : {1'b1, a[22:0]};
//     mant_b = (b[30:23] == 8'd0) ? {1'b0, b[22:0]} : {1'b1, b[22:0]};

//     // Compute result sign
//     sign_res = sign_a ^ sign_b;

//     // Multiply mantissas (24x24 = 48 bits)
//     mant_prod = mant_a * mant_b;

//     // Add exponents and subtract bias (127)
//     exp_sum = exp_a + exp_b - 8'd127;

//     // Normalize mantissa product
//     if (mant_prod[47] == 1) begin
//       mant_res = mant_prod[46:24];
//       exp_res = exp_sum + 1;
//     end else begin
//       mant_res = mant_prod[45:23];
//       exp_res = exp_sum;
//     end

//     // Assemble result
//     result = {sign_res, exp_res, mant_res};
//   end
// endmodule