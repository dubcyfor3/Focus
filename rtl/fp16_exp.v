module fp16_exp (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] x,        // FP16 input
    output reg  [15:0] exp_x,    // FP16 output
    output reg         done
);

    // FSM states
    reg [1:0] state;
    parameter IDLE = 2'd0, CALC = 2'd1, DONE = 2'd2;

    // Constants
    wire [15:0] one  = 16'h3C00;  // 1.0
    wire [15:0] half = 16'h3800;  // 0.5

    // Intermediate signals
    wire [15:0] x_sq;
    wire [15:0] term2;
    wire [15:0] tmp_sum1;
    wire [15:0] tmp_sum2;

    // FP16 units
    fp16_mult mul_xx    (.a(x),       .b(x),     .result(x_sq));
    fp16_mult mul_half  (.a(x_sq),    .b(half),  .result(term2));
    fp16_add  add1      (.a(one),     .b(x),     .result(tmp_sum1));
    fp16_add  add2      (.a(tmp_sum1),.b(term2), .result(tmp_sum2));

    // FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= IDLE;
            exp_x  <= 16'd0;
            done   <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start)
                        state <= CALC;
                end
                CALC: begin
                    exp_x <= tmp_sum2;
                    state <= DONE;
                end
                DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule