`ifdef SIMULATION
`timescale 1ns / 1ps
`endif

// fp16_mult.v
// Floating point 16-bit multiplier (IEEE 754 half precision)


// This version is based on
// https://github.com/scalesim-project/scale-sim-v2/blob/main/code-examples/systolic-array-rtl/BF16_processing_element/bfp16_mult.v
// Revised to fp16. Still has slight precision loss. 

module fp16_mult(a, b, result);
  input [15:0] a, b;
  output reg [15:0] result;

  wire a_sign = a[15];
  wire b_sign = b[15];
  wire [4:0] a_exponent = a[14:10];
  wire [4:0] b_exponent = b[14:10];
  wire [10:0] a_mantissa = (a_exponent == 0) ? {1'b0, a[9:0]} : {1'b1, a[9:0]};
  wire [10:0] b_mantissa = (b_exponent == 0) ? {1'b0, b[9:0]} : {1'b1, b[9:0]};

  wire [15:0] multiplier_a_in = a;
  wire [15:0] multiplier_b_in = b;
  wire [15:0] multiplier_out;

  gMultiplier M1 (
    .a(multiplier_a_in),
    .b(multiplier_b_in),
    .out(multiplier_out)
  );

  reg o_sign;
  reg [4:0] o_exponent;
  reg [10:0] o_mantissa;

  always @(*) begin
    result = 16'd0;
    if (a_exponent == 5'b11111 && a_mantissa[9:0] != 0) begin
      o_sign = a_sign;
      o_exponent = 5'b11111;
      o_mantissa = a_mantissa;
    end else if (b_exponent == 5'b11111 && b_mantissa[9:0] != 0) begin
      o_sign = b_sign;
      o_exponent = 5'b11111;
      o_mantissa = b_mantissa;
    end else if ((a_exponent == 0 && a_mantissa[9:0] == 0) || (b_exponent == 0 && b_mantissa[9:0] == 0)) begin
      o_sign = a_sign ^ b_sign;
      o_exponent = 0;
      o_mantissa = 0;
    end else if (a_exponent == 5'b11111 || b_exponent == 5'b11111) begin
      o_sign = a_sign;
      o_exponent = 5'b11111;
      o_mantissa = 0;
    end else if (a == 16'd0 && b == 16'd0) begin
      o_sign = 0;
      o_exponent = 0;
      o_mantissa = 0;
    end else begin
      o_sign = multiplier_out[15];
      o_exponent = multiplier_out[14:10];
      o_mantissa = {1'b1, multiplier_out[9:0]};
    end
    result = {o_sign, o_exponent, o_mantissa[9:0]};
  end

endmodule

// module fp16_mult(a, b, result);
//   input [15:0] a, b;
//   output reg [15:0] result;


//   wire a_sign;
//   wire b_sign;
//   wire [4:0] a_exponent;
//   wire [4:0] b_exponent;
//   wire [10:0] a_mantissa;
//   wire [10:0] b_mantissa;
             
//   reg o_sign;
//   reg [4:0]  o_exponent;
//   reg [10:0] o_mantissa;  
	
//   reg [15:0] multiplier_a_in;
//   reg [15:0] multiplier_b_in;
//   wire [15:0] multiplier_out;

//   assign a_sign = a[15];
//   assign a_exponent = a[14:10];
//   assign a_mantissa = (a_exponent == 0) ? {1'b0, a[9:0]} : {1'b1, a[9:0]};

//   assign b_sign = b[15];
//   assign b_exponent = b[14:10];
//   assign b_mantissa = (b_exponent == 0) ? {1'b0, b[9:0]} : {1'b1, b[9:0]};

//   assign multiplier_a_in = a;
//   assign multiplier_b_in = b;

//   gMultiplier M1 (
//     .a(multiplier_a_in),
//     .b(multiplier_b_in),
//     .out(multiplier_out)
//   );

//   always @ (*) begin
//     result = 16'd0;
//     if (a_exponent == 5'b11111 && a_mantissa[9:0] != 0) begin
//         o_sign = a_sign;
//         o_exponent = 5'b11111;
//         o_mantissa = a_mantissa;
//         result = {o_sign, o_exponent, o_mantissa[9:0]};
//     end else if (b_exponent == 5'b11111 && b_mantissa[9:0] != 0) begin
//         o_sign = b_sign;
//         o_exponent = 5'b11111;
//         o_mantissa = b_mantissa;
//         result = {o_sign, o_exponent, o_mantissa[9:0]};
//     end else if ((a_exponent == 0 && a_mantissa[9:0] == 0) || (b_exponent == 0 && b_mantissa[9:0] == 0)) begin
//         o_sign = a_sign ^ b_sign;
//         o_exponent = 0;
//         o_mantissa = 0;
//         result = {o_sign, o_exponent, o_mantissa[9:0]};
//     end else if (a_exponent == 5'b11111 || b_exponent == 5'b11111) begin
//         o_sign = a_sign;
//         o_exponent = 5'b11111;
//         o_mantissa = 0;
//         result = {o_sign, o_exponent, o_mantissa[9:0]};
//     end else if (a == 16'd0 && b == 16'd0) begin
//         o_sign = 0;
//         o_exponent = 0;
//         o_mantissa = 0;
//         result = {o_sign, o_exponent, o_mantissa[9:0]};
//     end else begin
//         o_sign = multiplier_out[15];
//         o_exponent = multiplier_out[14:10];
//         o_mantissa = {1'b1, multiplier_out[9:0]}; // optional
//         result = {o_sign, o_exponent, multiplier_out[9:0]};
//     end
//   end
// endmodule


module gMultiplier(a, b, out);
  input [15:0] a, b;
  output [15:0] out;
  wire [15:0] out;

  reg a_sign, b_sign;
  reg [4:0] a_exponent, b_exponent;
  reg [10:0] a_mantissa, b_mantissa;

  reg o_sign;
  reg [4:0] o_exponent;
  reg [10:0] o_mantissa;

  reg [21:0] product;

  assign out[15] = o_sign;
  assign out[14:10] = o_exponent;
  assign out[9:0] = o_mantissa[9:0];

  reg [4:0] i_e;
  reg [21:0] i_m;
  wire [4:0] o_e;
  wire [21:0] o_m;

  multiplication_normaliser norm1 (
    .in_e(i_e),
    .in_m(i_m),
    .out_e(o_e),
    .out_m(o_m)
  );

  always @ (*) begin
    a_sign = a[15];
    a_exponent = a[14:10];
    a_mantissa = (a_exponent == 0) ? {1'b0, a[9:0]} : {1'b1, a[9:0]};

    b_sign = b[15];
    b_exponent = b[14:10];
    b_mantissa = (b_exponent == 0) ? {1'b0, b[9:0]} : {1'b1, b[9:0]};

    o_sign = a_sign ^ b_sign;
    o_exponent = a_exponent + b_exponent - 5'd15;
    product = a_mantissa * b_mantissa;

    if (product[21]) begin
      o_exponent = o_exponent + 1;
      product = product >> 1;
    end else if (!product[20] && o_exponent != 0) begin
      i_e = o_exponent;
      i_m = product;
      o_exponent = o_e;
      product = o_m;
    end

    o_mantissa = product[19:9];
  end
endmodule


module multiplication_normaliser(in_e, in_m, out_e, out_m);
  input [4:0] in_e;
  input [21:0] in_m;
  output reg [4:0] out_e;
  output reg [21:0] out_m;

  always @ (*) begin
    if (in_m[20:15] == 6'b000001) begin
      out_e = in_e - 5'd5;
      out_m = in_m << 5;
    end else if (in_m[20:16] == 5'b00001) begin
      out_e = in_e - 5'd4;
      out_m = in_m << 4;
    end else if (in_m[20:17] == 4'b0001) begin
      out_e = in_e - 5'd3;
      out_m = in_m << 3;
    end else if (in_m[20:18] == 3'b001) begin
      out_e = in_e - 5'd2;
      out_m = in_m << 2;
    end else if (in_m[20:19] == 2'b01) begin
      out_e = in_e - 5'd1;
      out_m = in_m << 1;
    end else begin
      out_e = in_e;
      out_m = in_m;
    end
  end
endmodule


// A simpler fp16 multiplier, but has more precision loss. Kept for reference.
// module fp16_mult (
//     input  [15:0] a,
//     input  [15:0] b,
//     output [15:0] result
// );

//     wire sign_a = a[15];
//     wire sign_b = b[15];
//     wire [4:0] exp_a = a[14:10];
//     wire [4:0] exp_b = b[14:10];
//     wire [9:0] mant_a = a[9:0];
//     wire [9:0] mant_b = b[9:0];

//     // Add implicit leading 1
//     wire [10:0] norm_mant_a = (exp_a == 0) ? {1'b0, mant_a} : {1'b1, mant_a};
//     wire [10:0] norm_mant_b = (exp_b == 0) ? {1'b0, mant_b} : {1'b1, mant_b};

//     // Sign of result
//     wire sign_res = sign_a ^ sign_b;

//     // Multiply mantissas (11x11 bits = 22 bits)
//     wire [21:0] mant_res = norm_mant_a * norm_mant_b;

//     // Exponent calculation
//     wire [6:0] exp_sum = exp_a + exp_b - 5'd15 + mant_res[21]; // bias = 15, add 1 if normalized mantissa shifted

//     // Normalize mantissa
//     wire [10:0] final_mant = mant_res[21] ? mant_res[20:10] : mant_res[19:9];

//     // Handle zero inputs
//     wire zero = (exp_a == 0 && mant_a == 0) || (exp_b == 0 && mant_b == 0);

//     assign result = zero ? 16'd0 :
//                     {sign_res, exp_sum[4:0], final_mant[9:0]};

// endmodule
