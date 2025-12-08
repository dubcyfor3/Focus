module cmc_codec_pe (
    input wire clk,
    input wire rst_n,
    input wire [15:0] cur_pixels [0:63],
    input wire [15:0] ref_pixels [0:63],
    output wire [15:0] distance,
    output wire done
);
    // 1. Compute element wise absolute difference
    wire [15:0] abs_diff [0:63];
    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : abs_diff_gen
            assign abs_diff[i] = (cur_pixels[i] > ref_pixels[i]) ? 
                                 (cur_pixels[i] - ref_pixels[i]) :
                                 (ref_pixels[i] - cur_pixels[i]);
        end
    endgenerate

    // 2. Sum the absolute differences
    reg [15:0] sum;
    logic [20:0] sum_temp; // Temporary sum to handle overflow
    // Use a pipelined adder tree to merge 64 differences.
    adder_tree_64 adder_tree (
        .in(abs_diff),
        .sum(sum_temp)
    );
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum <= 16'b0;
        end else begin
            sum <= sum_temp[15:0]; // Take lower 16 bits of the sum
        end
    end
    
endmodule

// Ignore overflow for simplicity.
module adder_tree_64 (
    input  logic [15:0] in [0:63],   // 64 inputs, each 16-bit
    output logic [20:0]      sum    // Output sum (38-bit wide)
);

    // Stage 1: 64 -> 32
    logic [15:0] stage1 [31:0];
    genvar i;
    generate
        for (i = 0; i < 32; i++) begin
            assign stage1[i] = in[2*i] + in[2*i+1];
        end
    endgenerate

    // Stage 2: 32 -> 16
    logic [16:0] stage2 [15:0];
    generate
        for (i = 0; i < 16; i++) begin
            assign stage2[i] = stage1[2*i] + stage1[2*i+1];
        end
    endgenerate

    // Stage 3: 16 -> 8
    logic [17:0] stage3 [7:0];
    generate
        for (i = 0; i < 8; i++) begin
            assign stage3[i] = stage2[2*i] + stage2[2*i+1];
        end
    endgenerate

    // Stage 4: 8 -> 4
    logic [18:0] stage4 [3:0];
    generate
        for (i = 0; i < 4; i++) begin
            assign stage4[i] = stage3[2*i] + stage3[2*i+1];
        end
    endgenerate

    // Stage 5: 4 -> 2
    logic [19:0] stage5 [1:0];
    generate
        for (i = 0; i < 2; i++) begin
            assign stage5[i] = stage4[2*i] + stage4[2*i+1];
        end
    endgenerate

    // Stage 6: 2 -> 1
    assign sum = stage5[0] + stage5[1];

endmodule
