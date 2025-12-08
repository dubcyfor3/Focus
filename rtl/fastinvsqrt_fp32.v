// https://github.com/411568/FastInverseSqrtFPGA/blob/main/VerilogCode/FastInvSqrt.v

// module fastinvsqrt_fp32 (
//     input  wire        clk,       // Clock input
//     input  wire        rst_n,     // Active-low reset for new modules
//     input  wire        start,     // Start signal (optional, can be tied high)
//     output wire        done,      // Done signal (optional, can be used to indicate completion)
//     input  wire [31:0] input_fp,  // IEEE-754 floating point input
//     output wire [31:0] inv_sqrt    // Approximation of 1 / sqrt(input_fp)
// );

//     // Step 1: Reinterpret the input bits as an integer.
//     wire [31:0] int_input;
//     assign int_input = input_fp;

//     // Step 2: Apply the magic bit manipulation to compute an initial guess.
//     wire [31:0] half_int = int_input >> 1;
//     wire [31:0] magic_const = 32'h5f3759df;
//     wire [31:0] init_guess_int = magic_const - half_int;

//     // Step 3: Reinterpret the manipulated bits as a floating-point number.
//     wire [31:0] guess_fp;
//     assign guess_fp = init_guess_int;

//     // IEEE-754 constants (32-bit representations)
//     wire [31:0] fp_half   = 32'h3f000000;  // Represents 0.5
//     wire [31:0] fp_onept5 = 32'h3fc00000;  // Represents 1.5

//     // Intermediate signals for refining the initial guess:
//     wire [31:0] guess_squared;           // guess_fp * guess_fp
//     wire [31:0] input_times_guess_sq;    // input_fp * (guess_fp * guess_fp)
//     wire [31:0] half_input_guess_sq;     // 0.5 * input_fp * guess_fp^2
//     wire [31:0] neg_half_input_guess_sq; // Negated half_input_guess_sq
//     wire [31:0] refine_factor;           // 1.5 - 0.5 * input_fp * guess_fp^2
//     wire [31:0] inv_sqrt_result;         // Final result: guess_fp * refine_factor

//     // Multiply guess_fp * guess_fp
//     fp32_mult mult_guess_sq (
//         .clk(clk),
//         .rst_n(rst_n),
//         .a(guess_fp),
//         .b(guess_fp),
//         .result(guess_squared)
//     );

//     // Multiply input_fp * guess_squared
//     fp32_mult mult_input_guess (
//         .clk(clk),
//         .rst_n(rst_n),
//         .a(input_fp),
//         .b(guess_squared),
//         .result(input_times_guess_sq)
//     );

//     // Multiply 0.5 * input_times_guess_sq
//     fp32_mult mult_half (
//         .clk(clk),
//         .rst_n(rst_n),
//         .a(fp_half),
//         .b(input_times_guess_sq),
//         .result(half_input_guess_sq)
//     );

//     // Negate half_input_guess_sq by flipping the sign bit
//     assign neg_half_input_guess_sq = {~half_input_guess_sq[31], half_input_guess_sq[30:0]};

//     // Add 1.5 - 0.5 * input_fp * guess_fp^2
//     fp32_add add_refine (
//         .clk(clk),
//         .rst_n(rst_n),
//         .a(fp_onept5),
//         .b(neg_half_input_guess_sq),
//         .result(refine_factor)
//     );

//     // Final multiply guess_fp * refine_factor
//     fp32_mult mult_final (
//         .clk(clk),
//         .rst_n(rst_n),
//         .a(guess_fp),
//         .b(refine_factor),
//         .result(inv_sqrt_result)
//     );

//     // Output result
//     assign inv_sqrt = inv_sqrt_result;

// endmodule

module fastinvsqrt_fp32 (
    input  wire        clk,        // Clock input
    input  wire        rst_n,      // Active-low reset
    input  wire        start,      // Start signal
    output reg         done,       // Done signal
    input  wire [31:0] input_fp,   // IEEE-754 floating point input
    output reg  [31:0] inv_sqrt    // Approximation of 1 / sqrt(input_fp)
);

    // Pipeline state tracking
    reg [3:0] state;
    localparam IDLE       = 4'd0,
               STAGE1     = 4'd1,
               STAGE2     = 4'd2,
               STAGE3     = 4'd3,
               STAGE4     = 4'd4,
               STAGE5     = 4'd5,
               DONE_STATE = 4'd6;

    // Step 1: Reinterpret the input bits as an integer.
    wire [31:0] int_input = input_fp;
    wire [31:0] half_int = int_input >> 1;
    wire [31:0] magic_const = 32'h5f3759df;
    wire [31:0] init_guess_int = magic_const - half_int;
    wire [31:0] guess_fp = init_guess_int;

    // Constants
    wire [31:0] fp_half   = 32'h3f000000;  // 0.5
    wire [31:0] fp_onept5 = 32'h3fc00000;  // 1.5

    // Internal signals
    reg  [31:0] guess_squared;
    reg  [31:0] input_times_guess_sq;
    reg  [31:0] half_input_guess_sq;
    wire [31:0] neg_half_input_guess_sq;
    reg  [31:0] refine_factor;
    reg  [31:0] final_result;

    // Floating-point operation outputs
    wire [31:0] stage1_result;
    wire [31:0] stage2_result;
    wire [31:0] stage3_result;
    wire [31:0] stage4_result;
    wire [31:0] stage5_result;

    // Stage 1: guess_fp * guess_fp
    fp32_mult mult_guess_sq (
        .clk(clk),
        .rst_n(rst_n),
        .a(guess_fp),
        .b(guess_fp),
        .result(stage1_result)
    );

    // Stage 2: input_fp * guess_squared
    fp32_mult mult_input_guess (
        .clk(clk),
        .rst_n(rst_n),
        .a(input_fp),
        .b(guess_squared),
        .result(stage2_result)
    );

    // Stage 3: 0.5 * (input * guess^2)
    fp32_mult mult_half (
        .clk(clk),
        .rst_n(rst_n),
        .a(fp_half),
        .b(input_times_guess_sq),
        .result(stage3_result)
    );

    assign neg_half_input_guess_sq = {~half_input_guess_sq[31], half_input_guess_sq[30:0]};

    // Stage 4: 1.5 - 0.5 * input * guess^2
    fp32_add add_refine (
        .clk(clk),
        .rst_n(rst_n),
        .a(fp_onept5),
        .b(neg_half_input_guess_sq),
        .result(stage4_result)
    );

    // Stage 5: guess_fp * refine_factor
    fp32_mult mult_final (
        .clk(clk),
        .rst_n(rst_n),
        .a(guess_fp),
        .b(refine_factor),
        .result(stage5_result)
    );

    // FSM and pipeline progression
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 0;
            guess_squared <= 0;
            input_times_guess_sq <= 0;
            half_input_guess_sq <= 0;
            refine_factor <= 0;
            final_result <= 0;
            inv_sqrt <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= STAGE1;
                    end
                end

                STAGE1: begin
                    guess_squared <= stage1_result;
                    state <= STAGE2;
                end

                STAGE2: begin
                    input_times_guess_sq <= stage2_result;
                    state <= STAGE3;
                end

                STAGE3: begin
                    half_input_guess_sq <= stage3_result;
                    state <= STAGE4;
                end

                STAGE4: begin
                    refine_factor <= stage4_result;
                    state <= STAGE5;
                end

                STAGE5: begin
                    final_result <= stage5_result;
                    state <= DONE_STATE;
                end

                DONE_STATE: begin
                    inv_sqrt <= final_result;
                    done <= 1;
                    state <= IDLE; // or stay in DONE if you want one-shot behavior
                end
            endcase
        end
    end

endmodule
