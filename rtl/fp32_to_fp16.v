module fp32_to_fp16 (
    input  [31:0] in_fp32,
    output [15:0] out_fp16
);

    wire sign = in_fp32[31];
    wire [7:0] exp32 = in_fp32[30:23];
    wire [22:0] frac32 = in_fp32[22:0];

    wire [4:0] exp16;
    wire [9:0] frac16;

    wire overflow = (exp32 > 8'd142); // 142 = 127 + 15 => max FP16 exponent
    wire underflow = (exp32 < 8'd113); // 113 = 127 - 14 => min FP16 normal exp

    wire [7:0] exp_unbias = exp32 - 8'd127;

    assign exp16 = overflow     ? 5'b11111 : // Inf
                   underflow    ? 5'b00000 : // Zero or subnormal
                   exp_unbias + 5'd15;

    assign frac16 = overflow     ? 10'b0 :
                    underflow    ? 10'b0 :
                    frac32[22:13]; // truncate (no rounding)

    assign out_fp16 = {sign, exp16, frac16};

endmodule
