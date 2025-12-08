module fp16_recip (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] x,               // FP16 input
    input  wire        start_from_last_recip,
    input  wire [15:0] last_recip,      // previous reciprocal
    output reg  [15:0] recip_x,         // output = 1/x
    output reg         done
);

    // Internal wires
    reg [15:0] y;         // current approximation
    reg [15:0] xy;        // x * y
    reg [15:0] two_minus_xy;
    reg [15:0] y_next;

    reg [2:0] state;
    wire [2:0] next_state;

    localparam IDLE  = 3'd0;
    localparam CALC = 3'd1;
    localparam MUL1  = 3'd1;
    localparam SUB1  = 3'd2;
    localparam MUL2  = 3'd3;
    localparam ITER2 = 3'd4;
    localparam DONE  = 3'd5;

    // Constants
    wire [15:0] FP16_TWO = 16'h4000;  // 2.0 in FP16
    wire [15:0] FP16_ONE = 16'h3C00;  // 1.0 in FP16


    // Invert exponent around bias 15
    wire [4:0] exp_in  = x[14:10];
    wire [4:0] exp_out = 5'd30 - exp_in;  // - (exp_in-15) + 15
    // wire [9:0] frac_in = x[9:0];
    wire [9:0] frac_in = '0;

    // Keep same fraction as a rough start
    // wire [15:0] y0 = {x[15], exp_out, frac_in};
    wire [15:0] y0 = {1'b0, exp_out, frac_in};
    reg x_sign;
    wire [15:0] x_unsign = {1'b0, x[14:0]};


    // Instantiate FP16 units
    wire [15:0] mul_out_1, add_out, mul_out_2;
    wire [15:0] tmp_y;
    assign tmp_y = start_from_last_recip ? {1'b0, last_recip[14:0]} : y0;
    fp16_mult u_mult1 (.clk(clk), .rst_n(rst_n), .a(x_unsign), .b(tmp_y), .result(mul_out_1));
    fp16_add  u_add  (.clk(clk), .rst_n(rst_n), .a(FP16_TWO), .b({~mul_out_1[15], mul_out_1[14:0]}), .result(add_out));
    fp16_mult u_mult2 (.clk(clk), .rst_n(rst_n), .a(y), .b(add_out), .result(mul_out_2));
    // fp16_add  u_add  (.clk(clk), .rst_n(rst_n), .a(FP16_TWO), .b({~xy[15], xy[14:0]}), .result(add_out));
    // fp16_mult u_mult2 (.clk(clk), .rst_n(rst_n), .a(y), .b(two_minus_xy), .result(mul_out_2));

    reg [3:0] counter;

    // Control
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            recip_x <= '0;
            done <= '0;
            counter <= '0;
            x_sign <= '0;
        end else begin
            // `ifdef SIMULATION
            //     // $display("state=%0d, start=%0d, start_from_last_recip=%0d, x=%0h, last_recip=%0h, y0=%h, y=%0h, xy=%0h, mul_out_1=%0h, two_minus_xy=%0h, recip_x=%0h", 
            //     //     state, start, start_from_last_recip, x, last_recip, y0, y, xy, mul_out_1, two_minus_xy, recip_x);
            //     // $display("mul_out_1=%0h, add_1=%0h, add_2=%0h, add_out=%0h", 
            //     //     mul_out_1, FP16_TWO, {~mul_out_1[15], mul_out_1[14:0]}, add_out);
            //     $display("x=%0h, last_recip=%0h, y0=%h, y=%0h, mul_1=%0h, add_1=%0h, add_2=%0h, add_out=%0h, mul_out_2=%0h, recip_x=%0h, counter=%0d", 
            //         x, last_recip, y0, y, mul_out_1, FP16_TWO, {~mul_out_1[15], mul_out_1[14:0]}, add_out, mul_out_2, recip_x, counter);
            // `endif
            done <= '0;
            case (state)
                IDLE: begin
                    if (start) begin
                        y <= start_from_last_recip ? last_recip : y0;  // initial guess
                        // state <= MUL1;
                        state <= CALC;
                        counter <= '0;
                        x_sign <= x[15];
                    end
                end
                CALC: begin
                    counter <= counter + 1;
                    if (counter == 4'd3) begin
                        state <= DONE;
                        recip_x <= {x_sign, mul_out_2[14:0]};
                    end
                end



                // MUL1: begin
                //     // xy = x * y
                //     xy <= mul_out_1;
                //     state <= SUB1;
                // end

                // SUB1: begin
                //     // two_minus_xy = 2 - xy
                //     two_minus_xy <= add_out;
                //     state <= MUL2;
                // end

                // MUL2: begin
                //     // y_next = y * (2 - xy)
                //     recip_x <= mul_out_2;
                //     state <= DONE;
                // end

                DONE: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
