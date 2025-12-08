module inv_magnitude_unit (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [511:0] vec_flat,  // 32 * 16-bit FP16 numbers
    output reg [15:0] mag_inv,
    output reg done
);

    reg [4:0] sum_idx;
    reg [15:0] square_accum;
    reg [15:0] square_result;
    reg [15:0] sum_squares;

    reg [1:0] state, next_state;
    parameter IDLE = 2'd0, SQUARE_SUM = 2'd1, INVERSE_SQRT = 2'd2, DONE = 2'd3;

    wire [15:0] square_tmp;
    wire [15:0] inv_out;
    wire inv_done;
    wire inv_start = (state == INVERSE_SQRT);

    // unpack 512-bit vec_flat to 32 16-bit wires
    wire [15:0] vec [0:31];
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : vec_unpack
            assign vec[i] = vec_flat[i*16 +: 16];
        end
    endgenerate

    // fp16 square and accumulate
    fp16_mult mult_square (
        .a(vec[sum_idx]),
        .b(vec[sum_idx]),
        .result(square_tmp)
    );

    fp16_add add_square (
        .a(square_tmp),
        .b(square_accum),
        .result(square_result)
    );

    // FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state;
        case (state)
            IDLE: if (start) next_state = SQUARE_SUM;
            SQUARE_SUM: if (sum_idx == 5'd31) next_state = INVERSE_SQRT;
            INVERSE_SQRT: if (inv_done) next_state = DONE;
            DONE: next_state = IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_idx <= 0;
            square_accum <= 16'd0;
            sum_squares <= 16'd0;
            mag_inv <= 16'd0;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    sum_idx <= 0;
                    square_accum <= 16'd0;
                    done <= 0;
                end
                SQUARE_SUM: begin
                    square_accum <= square_result;
                    sum_idx <= sum_idx + 1;
                    if (sum_idx == 5'd31)
                        sum_squares <= square_result;
                end
                INVERSE_SQRT: begin
                    if (inv_done)
                        mag_inv <= (sum_squares == 16'd0) ? 16'd0 : inv_out;
                end
                DONE: begin
                    done <= 1;
                end
            endcase
        end
    end

    // instantiate inverse sqrt unit
    fast_inv_sqrt inv_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(inv_start),
        .in(sum_squares),
        .out(inv_out),
        .done(inv_done)
    );

endmodule


module fast_inv_sqrt (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [15:0] in,
    output reg [15:0] out,
    output reg done
);
    reg [2:0] state;
    parameter IDLE = 3'd0, MUL1 = 3'd1, MUL2 = 3'd2, MUL3 = 3'd3,
              SUB = 3'd4, MUL4 = 3'd5, DONE = 3'd6;

    reg [15:0] x2, y, y_squared, tmp1, tmp2, sub_res, final_out;
    wire [15:0] y_squared_w, tmp1_w, tmp2_w, sub_res_w, final_out_w;

    wire [15:0] half = 16'h3800;       // 0.5 in fp16
    wire [15:0] threehalves = 16'h3E00; // 1.5 in fp16

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 0;
            out <= 16'd0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        x2 <= in >> 1;
                        y <= 16'h5f37 - (in >> 1);
                        state <= MUL1;
                    end
                end
                MUL1: begin
                    y_squared <= y_squared_w;
                    state <= MUL2;
                end
                MUL2: begin
                    tmp1 <= tmp1_w;
                    state <= MUL3;
                end
                MUL3: begin
                    tmp2 <= tmp2_w;
                    state <= SUB;
                end
                SUB: begin
                    sub_res <= sub_res_w;
                    state <= MUL4;
                end
                MUL4: begin
                    final_out <= final_out_w;
                    state <= DONE;
                end
                DONE: begin
                    out <= final_out;
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // combinational datapath (results connected to registers above)
    fp16_mult m1 (.a(y), .b(y), .result(y_squared_w));
    fp16_mult m2 (.a(in), .b(y_squared), .result(tmp1_w));
    fp16_mult m3 (.a(half), .b(tmp1), .result(tmp2_w));
    fp16_add  sub (.a(threehalves), .b({~tmp2[15], tmp2[14:0]}), .result(sub_res_w)); // 1.5 - tmp2
    fp16_mult m4 (.a(y), .b(sub_res), .result(final_out_w));

endmodule

// `ifdef SIMULATION
// `timescale 1ns / 1ps
// `endif

// module inv_magnitude_unit (
//     input  logic         clk,
//     input  logic         rst_n,
//     input  logic         start,
//     input  logic [15:0]  vec[0:31],  // fp16 input vector
//     output logic [15:0]  mag_inv,    // fp16 1/sqrt(sum of squares)
//     output logic         done
// );

//     // State machine states
//     typedef enum logic [1:0] {
//         IDLE,
//         SQUARE_SUM,
//         INVERSE_SQRT,
//         DONE
//     } state_t;

//     state_t current_state, next_state;
//     logic [15:0] sum_squares; // Accumulator for sum of squares
//     logic sum_squares_done;   // Indicates sum of squares is done
//     logic fastinvsqrt_done;

//     // State transition
//     always_ff @(posedge clk or negedge rst_n) begin
//         if (!rst_n)
//             current_state <= IDLE;
//         else
//             current_state <= next_state;
//     end
//     // Next state logic
//     always_comb begin
//         next_state = current_state;
//         case (current_state)
//             IDLE: if (start) next_state = SQUARE_SUM;
//             SQUARE_SUM: if (sum_squares_done) next_state = INVERSE_SQRT;
//             INVERSE_SQRT: if (fastinvsqrt_done) next_state = DONE;
//             DONE: next_state = IDLE;
//         endcase
//     end

//     // Generate Square with for loop
//     logic [15:0] squares[0:31];
//     logic [31:0] squares_fp32[0:31];
//     genvar i;
//     generate
//         for (i = 0; i < 32; i = i + 1) begin : SQUARE_GEN
//             fp16_mult mult_inst (.a(vec[i]), .b(vec[i]), .result(squares[i]));
//             fp16_to_fp32 conv_inst (.in_fp16(squares[i]), .out_fp32(squares_fp32[i])); // Convert to fp32 for accumulation
//         end
//     endgenerate

//     // // Generate Sum for 32 squares
//     // SumSquares sum_squares_inst (.clk(clk), .rst_n(rst_n), 
//     //                             .start(current_state == SQUARE_SUM), 
//     //                             .squares(squares_fp32), 
//     //                             .sum_out(sum_squares), 
//     //                             .done(sum_squares_done));
//     // Generate Sum for 32 squares using 32 fp32_add. Sequentially. Use this for throughput equals 1. 
//     logic [31:0] temp_sum[0:31];
//     assign temp_sum[0] = squares_fp32[0];
//     generate
//         for (i = 1; i < 32; i = i + 1) begin : SUM_GEN
//             fp32_add add_inst (.clk(clk), .rst_n(rst_n), .a(temp_sum[i-1]), .b(squares_fp32[i]), .result(temp_sum[i]));
//         end
//     endgenerate
//     assign sum_squares = temp_sum[31];
//     assign sum_squares_done = (current_state == SQUARE_SUM) ? 1'b1 : 1'b0;

//     // Generate inverse sqrt with fast inverse sqrt unit
//     logic [31:0] mag_inv_fp32;
//     logic fastinvsqrt_start;
//     assign fastinvsqrt_start = (current_state == INVERSE_SQRT) ? 1'b1 : 1'b0;
    
//     fastinvsqrt_fp32 sqrt_inst (.clk(clk), .rst_n(rst_n), 
//         .start(fastinvsqrt_start), 
//         .done(fastinvsqrt_done),
//         .input_fp(sum_squares), 
//         .inv_sqrt(mag_inv_fp32));
//     logic [15:0] mag_inv_fp16;
//     fp32_to_fp16 conv_back_inst (.in_fp32(mag_inv_fp32), .out_fp16(mag_inv_fp16));

//     // Control signals
//     always_ff @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             done <= 1'b0;
//             // sum_squares <= 32'd0;
//             mag_inv <= 16'd0;
//         end else begin
//             case (current_state)
//                 IDLE: begin
//                     // sum_squares <= 32'd0;
//                     done <= 1'b0;
//                 end
//                 SQUARE_SUM: begin
//                     // sum_squares <= 32'd0;
//                     // for (int i = 0; i < 32; i++) begin
//                     //     fp32_add add_inst (.clk(clk), .rst_n(rst_n), .a(sum_squares), .b(squares_fp32[i]), .result(sum_squares));
//                     // end
//                 end
//                 INVERSE_SQRT: begin
//                     if (fastinvsqrt_done) begin
//                         if (sum_squares == 32'd0) begin
//                             mag_inv <= 16'd0;
//                         end else begin
//                             mag_inv <= mag_inv_fp16;
//                         end
//                     end
//                 end
//                 DONE: begin
//                     done <= 1'b1;
//                 end
//             endcase
//         end
//     end

// endmodule

