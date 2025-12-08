// ema_update_unit exponential moving average (EMA)
module average_update_unit (
    input  [15:0] alpha,       // FP16: smoothing factor (0 < alpha < 1)
    input  [15:0] one_minus_alpha,   // FP16: 1-alpha, precomputed for convenience
    input  [15:0] new_val,     // FP16: new input value
    input  [15:0] old_avg,     // FP16: previous average
    output [15:0] new_avg      // FP16: updated average
);

    // wire [15:0] one_minus_alpha;
    wire [15:0] alpha_new;
    wire [15:0] old_scaled;
    wire [15:0] sum;
    

    // Compute alpha * new_val
    fp16_mult u_mul1 (
        .a(alpha),
        .b(new_val),
        .result(alpha_new)
    );

    // Compute (1 - alpha) * old_avg
    fp16_mult u_mul2 (
        .a(one_minus_alpha),
        .b(old_avg),
        .result(old_scaled)
    );

    fp16_add u_add (
        .a(alpha_new),
        .b(old_scaled),
        .result(sum)
    );

    assign new_avg = sum;

    // logic [31:0] alpha_new_fp32, old_scaled_fp32, new_avg_fp32;
    // fp16_to_fp32 u_f16_to_f32_1 (.in_fp16(alpha_new), .out_fp32(alpha_new_fp32));
    // fp16_to_fp32 u_f16_to_f32_2 (.in_fp16(old_scaled), .out_fp32(old_scaled_fp32));
    // logic clk_dummy, rst_n_dummy;
    // assign clk_dummy = 1'b0;
    // assign rst_n_dummy = 1'b1;
    // fp32_add u_f32_add (.clk(clk_dummy), .rst_n(rst_n_dummy),
    //     .a(alpha_new_fp32), .b(old_scaled_fp32), .result(new_avg_fp32));
    // fp32_to_fp16 u_f32_to_f16 (.in_fp32(new_avg_fp32), .out_fp16(new_avg));

endmodule
