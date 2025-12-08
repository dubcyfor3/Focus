module fp16_to_fp32 (
    input  [15:0] in_fp16,
    output [31:0] out_fp32
);

    wire sign = in_fp16[15];
    wire [4:0] exp16 = in_fp16[14:10];
    wire [9:0] frac16 = in_fp16[9:0];

    wire [7:0] exp32;
    wire [22:0] frac32;

    assign exp32 = (exp16 == 5'b00000) ? 8'd0 :  // subnormal or zero
                   (exp16 == 5'b11111) ? 8'hFF : // Inf or NaN
                   exp16 + 8'd112;               // 127 - 15 = 112

    // assign frac32 = (exp16 == 5'b00000) ? {13'b0, frac16} : // denormals
    //                 {13'b0, frac16};                        // normal numbers (pad to 23 bits)
    assign frac32 = {frac16, 13'b0};                        // (pad to 23 bits)

    assign out_fp32 = {sign, exp32, frac32};

endmodule
