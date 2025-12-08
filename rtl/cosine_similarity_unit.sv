`ifdef SIMULATION
`timescale 1ns / 1ps
`endif

module cosine_similarity_unit (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    output logic         done,
    input  logic [15:0]  vec1[0:31],       // FP16 vector 1
    input  logic [15:0]  vec2[0:31],       // FP16 vector 2
    input  logic [15:0]  vec1_mag_inv,     // FP16 1/||vec1||
    input  logic [15:0]  vec2_mag_inv,     // FP16 1/||vec2||
    output logic [15:0]  similarity        // FP16 cosine similarity
);

    // Internal signals
    logic [15:0] products[0:31];
    logic [15:0] accum;
    logic [5:0]  sum_idx;
    logic        products_done;
    logic        dot_product_done;
    logic        final_mult_done;

    logic [15:0] add_result;

    // === Stage 1: Element-wise multiply vec1[i] * vec2[i] ===
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : PRODUCT_GEN
            fp16_mult u_mult (
                .a(vec1[i]),
                .b(vec2[i]),
                .result(products[i])
            );
        end
    endgenerate

    fp16_add u_add (
        .a(accum),
        .b(products[sum_idx]),
        .result(accum)
    );

    // === Stage 2: Sum all products using FP16 add ===
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_idx <= 0;
            accum <= 16'h0000;
            products_done <= 1'b0;
            dot_product_done <= 1'b0;
        end else if (start) begin
            sum_idx <= 0;
            accum <= 16'h0000;
            products_done <= 1'b1;
            dot_product_done <= 1'b0;
        end else if (products_done && sum_idx < 32) begin
            accum <= add_result;
            sum_idx <= sum_idx + 1;
            if (sum_idx == 31) begin
                dot_product_done <= 1'b1;
                products_done <= 1'b0;
            end
        end else begin
            dot_product_done <= 1'b0;
        end
    end

    // === Stage 3: Multiply dot_product * vec1_mag_inv * vec2_mag_inv ===
    logic [15:0] sim_tmp1;
    logic [15:0] sim_fp16;

    fp16_mult u_mult_final1 (
        .a(accum),
        .b(vec1_mag_inv),
        .result(sim_tmp1)
    );

    fp16_mult u_mult_final2 (
        .a(sim_tmp1),
        .b(vec2_mag_inv),
        .result(sim_fp16)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            final_mult_done <= 1'b0;
        end else if (dot_product_done) begin
            final_mult_done <= 1'b1;
        end else begin
            final_mult_done <= 1'b0;
        end
    end

    assign done = final_mult_done;
    assign similarity = (final_mult_done) ? sim_fp16 : 16'h0000;

endmodule


// `ifdef SIMULATION
// `timescale 1ns / 1ps
// `endif

// module cosine_similarity_unit (
//     input  logic         clk,
//     input  logic         rst_n,
//     input  logic         start,
//     output logic         done,
//     input  logic [15:0]  vec1[0:31],  // fp16 input vector 1
//     input  logic [15:0]  vec2[0:31],  // fp16 input vector 2
//     input  logic [15:0]  vec1_mag_inv, // fp16 1/magnitude of vec1
//     input  logic [15:0]  vec2_mag_inv, // fp16 1/magnitude of vec2
//     output logic [15:0]  similarity   // fp16 cosine similarity
// );
//     logic [15:0] products[0:31];
//     logic [31:0] products_fp32[0:31];
//     logic products_done;
//     logic [31:0] dot_product_fp32;
//     logic dot_product_done;

//     // Generate products
//     genvar i;
//     generate
//         for (i = 0; i < 32; i = i + 1) begin : PRODUCT_GEN
//             fp16_mult mult_inst (.a(vec1[i]), .b(vec2[i]), .result(products[i]));
//             fp16_to_fp32 conv_inst (.in_fp16(products[i]), .out_fp32(products_fp32[i]));
//         end
//     endgenerate
//     // assign products_done = start; // For simplicity, assume done immediately after start
//     always_ff @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             products_done <= 1'b0;
//         end else if (start) begin
//             products_done <= 1'b1; // In real design, this would be after some cycles
//             // `ifdef SIMULATION
//             // $display("INFO: products[%0d] = %h", 0, products[0]);
//             // `endif
//         end else begin
//             products_done <= 1'b0;
//         end
//     end

//     // Sum products to get dot product
//     sum_fp32 #(.DATA_WIDTH(32), .VECTOR_SIZE(32)) sum_inst (
//         .clk(clk),
//         .rst_n(rst_n),
//         .start(products_done),
//         .done(dot_product_done),
//         .vec(products_fp32),
//         .sum_out(dot_product_fp32)
//     );

//     // always_ff @(posedge clk) begin
//     //     if (dot_product_done) begin
//     //         `ifdef SIMULATION
//     //         $display("INFO: dot_product_fp32 = %h", dot_product_fp32);
//     //         `endif
//     //     end
//     // end

//     // Final multiplication to get cosine similarity
//     logic final_mult_done;
//     // fp16_mult final_mult_inst (.a(fp32_to_fp16_f(dot_product_fp32)), .b(vec1_mag_inv), .result(similarity), .start(dot_product_done), .done(final_mult_done));
//     logic [15:0] dot_product_fp16;
//     fp32_to_fp16 dot_product_conv_inst (.in_fp32(dot_product_fp32), .out_fp16(dot_product_fp16));
//     logic [15:0] similarity_tmp1;
//     logic [15:0] similarity_fp16;
//     fp16_mult final_mult_inst_vec1_mag (.a(dot_product_fp16), .b(vec1_mag_inv), .result(similarity_tmp1));
//     fp16_mult final_mult_inst_vec2_mag (.a(similarity_tmp1), .b(vec2_mag_inv), .result(similarity_fp16));
//     // assign final_mult_done = dot_product_done; // For simplicity, assume done immediately after dot product is ready
//     always_ff @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             final_mult_done <= 1'b0;
//         end else if (dot_product_done) begin
//             final_mult_done <= 1'b1; // In real design, this would be after some cycles
//         end else begin
//             final_mult_done <= 1'b0;
//         end
//     end

//     assign done = final_mult_done;
//     assign similarity = (final_mult_done) ? similarity_fp16 : 16'd0;

//     // function automatic logic [15:0] fp32_to_fp16_f(input logic [31:0] in);
//     //     return {in[31], in[30:23] - 8'd112, in[22:10]};
//     // endfunction

//     // always_ff @(posedge clk or negedge rst_n) begin
//     //     if (!rst_n) begin
//     //         similarity <= 16'd0;
//     //     end else if (start) begin
//     //         similarity <= 16'd0;
//     //     end else if (final_mult_done) begin
//     //         similarity <= similarity_fp16;
//     //     end else if (done) begin
//     //         // similarity is updated by final_mult_inst
//     //     end
//     // end
// endmodule
