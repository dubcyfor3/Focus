`ifdef SIMULATION
`timescale 1ns / 1ps
`endif

// // fp32_add.v
// // Floating point 32-bit adder (IEEE 754 single precision)

//////////////////////////////////////////////////////////////
// FP32 Adder 

// https://github.com/danshanley/FPU/blob/master/fpu.v

// Format: 1-bit signed, 8-bit exponents, 23-bit fractions

// NOTE: MORE VERIFICATION NEEDED
/////////////////////////////////////////////////////////////

module fp32_add(a, b, result);

  input [31:0] a, b;
  output reg [31:0] result;

  wire a_sign = a[31];
  wire b_sign = b[31];
  wire [7:0] a_exponent = a[30:23];
  wire [7:0] b_exponent = b[30:23];
  wire [23:0] a_mantissa = {1'b1, a[22:0]};
  wire [23:0] b_mantissa = {1'b1, b[22:0]};

  reg o_sign;
  reg [7:0] o_exponent;
  reg [24:0] o_mantissa;

  wire [31:0] adder_a_in = a;
  wire [31:0] adder_b_in = b;
  wire [31:0] adder_out;

  generalAdder gAdder (
    .a(adder_a_in),
    .b(adder_b_in),
    .out(adder_out)
  );

  always @(*) begin
    result = 32'd0;
    if ((a_exponent == 255 && a_mantissa[22:0] != 0) || (b_exponent == 0 && b_mantissa[22:0] == 0)) begin
      o_sign = a_sign;
      o_exponent = a_exponent;
      o_mantissa = a_mantissa;
    end else if ((b_exponent == 255 && b_mantissa[22:0] != 0) || (a_exponent == 0 && a_mantissa[22:0] == 0)) begin
      o_sign = b_sign;
      o_exponent = b_exponent;
      o_mantissa = b_mantissa;
    end else if ((a_exponent == 255) || (b_exponent == 255)) begin
      o_sign = a_sign ^ b_sign;
      o_exponent = 255;
      o_mantissa = 0;
    end else begin
      o_sign = adder_out[31];
      o_exponent = adder_out[30:23];
      o_mantissa = adder_out[22:0];
    end
    result = {o_sign, o_exponent, o_mantissa[22:0]};
  end

endmodule


// module fp32_add(a, b, result);

//   input [31:0] a, b;
//   output reg [31:0] result;

//   wire a_sign;
//   wire b_sign;
//   wire [7:0] a_exponent;
//   wire [7:0] b_exponent; 
//   wire [23:0] a_mantissa; // plus one bit
//   wire [23:0] b_mantissa; // plus one bit 
             
//   reg o_sign;
//   reg [7:0] o_exponent;
//   reg [24:0] o_mantissa;  // plus two bits
	
//   reg [31:0] adder_a_in;
//   reg [31:0] adder_b_in;
//   wire [31:0] adder_out;
                   

//   assign a_sign = a[31];
//   assign a_exponent[7:0] = a[30:23];
//   assign a_mantissa[23:0] = {1'b1, a[22:0]};
//   assign b_sign = b[31];
//   assign b_exponent[7:0] = b[30:23];
//   assign b_mantissa[23:0] = {1'b1, b[22:0]};

//   generalAdder gAdder (
//     .a(adder_a_in),
//     .b(adder_b_in),
//     .out(adder_out)
//   );

//   assign adder_a_in = a;
//   assign adder_b_in = b;
  
//   //covers corner cases and uses general adder logic
//   //always @ ( posedge clk ) begin
//   always @ ( * ) begin
// 		result = 32'd0;
// 		//If a is NaN or b is zero return a
// 		if ((a_exponent == 255 && a_mantissa[22:0] != 0) || (b_exponent == 0) && (b_mantissa[22:0] == 0)) begin
// 			o_sign = a_sign;
// 			o_exponent = a_exponent;
// 			o_mantissa = a_mantissa;
// 			result = {o_sign, o_exponent, o_mantissa[22:0]};
// 		//If b is NaN or a is zero return b
// 		end else if ((b_exponent == 255 && b_mantissa[22:0] != 0) || (a_exponent == 0) && (a_mantissa[22:0] == 0)) begin
// 			o_sign = b_sign;
// 			o_exponent = b_exponent;
// 			o_mantissa = b_mantissa;
// 			result = {o_sign, o_exponent, o_mantissa[22:0]};
// 		//if a and b is inf return inf
// 		end else if ((a_exponent == 255) || (b_exponent == 255)) begin
// 			o_sign = a_sign ^ b_sign;
// 			o_exponent = 255;
// 			o_mantissa = 0;
// 			result = {o_sign, o_exponent, o_mantissa[22:0]};
// 		end else begin // Passed all corner cases
// 			//adder_a_in = A;
// 			//adder_b_in = B;
// 			o_sign = adder_out[31];
// 			o_exponent = adder_out[30:23];
// 			o_mantissa = adder_out[22:0];
// 			result = {o_sign, o_exponent, o_mantissa[22:0]};
// 		end
//   end  
        
// endmodule

//general adder logic whenever the inputs change
module generalAdder(a, b, out);
  input [31:0] a, b;
  output [31:0] out;   

  wire [31:0] out;
  
  reg a_sign;
  reg b_sign;
  reg [7:0] a_exponent;
  reg [7:0] b_exponent;
  reg [23:0] a_mantissa;
  reg [23:0] b_mantissa;   
  
  reg o_sign;
  reg [7:0] o_exponent;
  reg [24:0] o_mantissa; 


  reg [7:0] diff;
  reg [23:0] tmp_mantissa;

  reg [7:0] i_e;
  reg [24:0] i_m;
  wire [7:0] o_e;
  wire [24:0] o_m;

                       
  addition_normaliser norm1(
    .in_e(i_e),
    .in_m(i_m),
    .out_e(o_e),
    .out_m(o_m)
  );

  assign out[31] = o_sign;
  assign out[30:23] = o_exponent;
  assign out[22:0] = o_mantissa[22:0];

  always @ (*) begin
  
	  a_sign = a[31];
     
	  if(a[30:23] == 0) begin
		  a_exponent = 8'b00000001;
		  a_mantissa = {1'b0, a[22:0]};
	  end else begin
		  a_exponent = a[30:23];
		  a_mantissa = {1'b1, a[22:0]};
	  end
     
	  b_sign = b[31];
     
	  if(b[30:23] == 0) begin
		  b_exponent = 8'b00000001;
		  b_mantissa = {1'b0, b[22:0]};
	  end else begin
		  b_exponent = b[30:23];
		  b_mantissa = {1'b1, b[22:0]};
	  end
     
    if (a_exponent == b_exponent) begin // Equal exponents
      	o_exponent = a_exponent;
      	if (a_sign == b_sign) begin // Equal signs = add
        	o_mantissa = a_mantissa + b_mantissa;
        	//Signify to shift
        	o_mantissa[24] = 1;
        	o_sign = a_sign;
      	end else begin // Opposite signs = subtract
        	if(a_mantissa > b_mantissa) begin
          	o_mantissa = a_mantissa - b_mantissa;
          	o_sign = a_sign;
        	end else begin
       		  o_mantissa = b_mantissa - a_mantissa;
     		    o_sign = b_sign;
        	end
      	end
    end else begin //Unequal exponents
      	if (a_exponent > b_exponent) begin // A is bigger
        	o_exponent = a_exponent;
        	o_sign = a_sign;
			    diff = a_exponent - b_exponent;
        	tmp_mantissa = b_mantissa >> diff;
        	if (a_sign == b_sign)
          		o_mantissa = a_mantissa + tmp_mantissa;
        	else
          		o_mantissa = a_mantissa - tmp_mantissa;
     		end else if (a_exponent < b_exponent) begin // B is bigger
     		  o_exponent = b_exponent;
     		  o_sign = b_sign;
       		diff = b_exponent - a_exponent;
       		tmp_mantissa = a_mantissa >> diff;
        	if (a_sign == b_sign)
          		o_mantissa = b_mantissa + tmp_mantissa;
     		  else
				      o_mantissa = b_mantissa - tmp_mantissa;
      	end
    end

    if(o_mantissa[24] == 1) begin
      	o_exponent = o_exponent + 1;
      	o_mantissa = o_mantissa >> 1;
    end else if((o_mantissa[23] != 1) && (o_exponent != 0)) begin
      	i_e = o_exponent;
      	i_m = o_mantissa;
      	o_exponent = o_e;
      	o_mantissa = o_m;
    end
  end
endmodule 

module addition_normaliser(in_e, in_m, out_e, out_m);
  input [7:0] in_e;
  input [24:0] in_m;
  output [7:0] out_e;
  output [24:0] out_m;
  
  wire [7:0] in_e;
  wire [24:0] in_m;
  reg [7:0] out_e;
  reg [24:0] out_m;
  
  
  always @ ( * ) begin
    if (in_m[23:3] == 21'b000000000000000000001) begin
	  out_e = in_e - 20;
	  out_m = in_m << 20;
	end else if (in_m[23:4] == 20'b00000000000000000001) begin
	  out_e = in_e - 19;
	  out_m = in_m << 19;
	end else if (in_m[23:5] == 19'b0000000000000000001) begin
	  out_e = in_e - 18;
	  out_m = in_m << 18;
	end else if (in_m[23:6] == 18'b000000000000000001) begin
	  out_e = in_e - 17;
	  out_m = in_m << 17;
	end else if (in_m[23:7] == 17'b00000000000000001) begin
	  out_e = in_e - 16;
	  out_m = in_m << 16;
	end else if (in_m[23:8] == 16'b0000000000000001) begin
	  out_e = in_e - 15;
	  out_m = in_m << 15;
	end else if (in_m[23:9] == 15'b000000000000001) begin
	  out_e = in_e - 14;
	  out_m = in_m << 14;
	end else if (in_m[23:10] == 14'b00000000000001) begin
	  out_e = in_e - 13;
	  out_m = in_m << 13;
	end else if (in_m[23:11] == 13'b0000000000001) begin
	  out_e = in_e - 12;
	  out_m = in_m << 12;
	end else if (in_m[23:12] == 12'b000000000001) begin
	  out_e = in_e - 11;
	  out_m = in_m << 11;
	end else if (in_m[23:13] == 11'b00000000001) begin
	  out_e = in_e - 10;
      out_m = in_m << 10;
	end else if (in_m[23:14] == 10'b0000000001) begin
	  out_e = in_e - 9;
	  out_m = in_m << 9;
	end else if (in_m[23:15] == 9'b000000001) begin
	  out_e = in_e - 8;
	  out_m = in_m << 8;
	end else if (in_m[23:16] == 8'b00000001) begin
	  out_e = in_e - 7;
	  out_m = in_m << 7;
	end else if (in_m[23:17] == 7'b0000001) begin
	  out_e = in_e - 6;
      out_m = in_m << 6;
	end else if (in_m[23:18] == 6'b000001) begin
	  out_e = in_e - 5;
	  out_m = in_m << 5;
	end else if (in_m[23:19] == 5'b00001) begin
	  out_e = in_e - 4;
	  out_m = in_m << 4;
	end else if (in_m[23:20] == 4'b0001) begin
	  out_e = in_e - 3;
	  out_m = in_m << 3;
	end else if (in_m[23:21] == 3'b001) begin
	  out_e = in_e - 2;
	  out_m = in_m << 2;
	end else if (in_m[23:22] == 2'b01) begin
	  out_e = in_e - 1;
	  out_m = in_m << 1;
	end
  end
endmodule
  







// // Note: This is a simplified implementation and does not handle all edge cases.
// module fp32_add (
//     input  [31:0] a,
//     input  [31:0] b,
//     output [31:0] result
// );

//     // Extract fields
//     wire sign_a = a[31];
//     wire sign_b = b[31];
//     wire [7:0] exp_a = a[30:23];
//     wire [7:0] exp_b = b[30:23];
//     wire [22:0] frac_a = a[22:0];
//     wire [22:0] frac_b = b[22:0];

//     // Add implicit 1 to mantissa if normal
//     wire [24:0] mant_a = (exp_a == 0) ? {2'b00, frac_a} : {1'b1, frac_a, 1'b0};
//     wire [24:0] mant_b = (exp_b == 0) ? {2'b00, frac_b} : {1'b1, frac_b, 1'b0};

//     // Align exponents
//     wire [7:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
//     wire [24:0] mant_a_shifted = (exp_a > exp_b) ? mant_a : (mant_a >> exp_diff);
//     wire [24:0] mant_b_shifted = (exp_a > exp_b) ? (mant_b >> exp_diff) : mant_b;
//     wire [7:0] exp_common = (exp_a > exp_b) ? exp_a : exp_b;

//     // Add or subtract mantissas
//     wire same_sign = (sign_a == sign_b);
//     wire [25:0] mant_add = mant_a_shifted + mant_b_shifted;
//     wire [25:0] mant_sub = (mant_a_shifted >= mant_b_shifted)
//                            ? (mant_a_shifted - mant_b_shifted)
//                            : (mant_b_shifted - mant_a_shifted);
//     wire sign_res_sub = (mant_a_shifted >= mant_b_shifted) ? sign_a : sign_b;

//     wire [25:0] mant_raw = same_sign ? mant_add : mant_sub;
//     wire sign_res = same_sign ? sign_a : sign_res_sub;

//     // Normalize result
//     reg [7:0] exp_res;
//     reg [22:0] frac_res;
//     reg [25:0] mant_norm;
//     reg [4:0] shift;

//     always @(*) begin
//         mant_norm = mant_raw;
//         exp_res = exp_common;

//         // Normalize left if subtracted result
//         if (!same_sign) begin
//             shift = 0;
//             while (mant_norm[25] == 0 && mant_norm != 0 && shift < 25) begin
//                 mant_norm = mant_norm << 1;
//                 exp_res = exp_res - 1;
//                 shift = shift + 1;
//             end
//         end else if (mant_norm[25]) begin
//             // Right shift if overflow in addition
//             mant_norm = mant_norm >> 1;
//             exp_res = exp_res + 1;
//         end

//         frac_res = mant_norm[24:2]; // Truncate rounding
//     end

//     // Zero detection
//     wire is_zero = (mant_raw == 0);

//     assign result = is_zero ? 32'b0 :
//                     {sign_res, exp_res, frac_res};

// endmodule
