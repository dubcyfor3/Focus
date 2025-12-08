`ifdef SIMULATION
`timescale 1ns / 1ps
`endif

module fp16_add (
    input  [15:0] a,
    input  [15:0] b,
    output [15:0] result
);

    // Extract fields
    wire sign_a = a[15];
    wire sign_b = b[15];
    wire [4:0] exp_a = a[14:10];
    wire [4:0] exp_b = b[14:10];
    wire [9:0] frac_a = a[9:0];
    wire [9:0] frac_b = b[9:0];

    // Add implicit leading 1
    wire [11:0] mant_a = (exp_a == 0) ? {2'b00, frac_a} : {1'b1, frac_a};
    wire [11:0] mant_b = (exp_b == 0) ? {2'b00, frac_b} : {1'b1, frac_b};

    // Align exponents
    wire [4:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [11:0] mant_a_shifted = (exp_a > exp_b) ? mant_a : (mant_a >> exp_diff);
    wire [11:0] mant_b_shifted = (exp_b > exp_a) ? mant_b : (mant_b >> exp_diff);
    wire [4:0] exp_common = (exp_a > exp_b) ? exp_a : exp_b;

    // Add or subtract mantissas
    wire same_sign = (sign_a == sign_b);
    wire [12:0] mant_add = mant_a_shifted + mant_b_shifted;
    wire [12:0] mant_sub = (mant_a_shifted >= mant_b_shifted)
                         ? (mant_a_shifted - mant_b_shifted)
                         : (mant_b_shifted - mant_a_shifted);
    wire sign_res_sub = (mant_a_shifted >= mant_b_shifted) ? sign_a : sign_b;

    wire [12:0] mant_raw = same_sign ? mant_add : mant_sub;
    wire sign_res = same_sign ? sign_a : sign_res_sub;

    // Normalize result
    reg [4:0] exp_res;
    reg [9:0] frac_res;
    reg [12:0] mant_norm;
    integer i;

    always @(*) begin
        mant_norm = mant_raw;
        exp_res = exp_common;

        if (!same_sign) begin
            for (i = 12; i > 0 && mant_norm[i] == 0; i = i - 1) begin
                mant_norm = mant_norm << 1;
                exp_res = exp_res - 1;
            end
        end else if (mant_norm[12]) begin
            mant_norm = mant_norm >> 1;
            exp_res = exp_res + 1;
        end

        frac_res = mant_norm[9:0]; // truncate
    end

    // Detect zero result
    wire is_zero = (mant_raw == 0);

    assign result = is_zero ? 16'b0 : {sign_res, exp_res, frac_res};

endmodule
