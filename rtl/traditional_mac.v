// https://github.com/scalesim-project/scale-sim-v2/blob/main/code-examples/systolic-array-rtl/systolic_array_8bit_MAC_PE/traditional_mac.v

module traditional_mac 
#(
    parameter WORD_SIZE = 16
)(
    clk,
    rst,
    
    //Control Signals
    fsm_op2_select_in,
    fsm_out_select_in,
    stat_bit_in,        // Drives selects for WS and IS modes

    // Data ports
    left_in,
    top_in, 
    right_out,
    bottom_out
);


input clk;
input rst;

input fsm_op2_select_in;
input fsm_out_select_in;
input stat_bit_in;

input [WORD_SIZE - 1: 0] left_in;
input [WORD_SIZE - 1: 0] top_in;

output [WORD_SIZE - 1: 0] right_out;
output [WORD_SIZE - 1: 0] bottom_out;

wire [255:0] tie_low;

reg [WORD_SIZE - 1: 0] stationary_operand_reg;
reg [WORD_SIZE - 1: 0] top_in_reg;
reg [WORD_SIZE - 1: 0] left_in_reg;
reg [WORD_SIZE - 1: 0] accumulator_reg;

wire [WORD_SIZE - 1: 0] multiplier_out;
wire [WORD_SIZE - 1: 0] adder_out; 
wire [WORD_SIZE - 1: 0] mult_op2_mux_out;
wire [WORD_SIZE - 1: 0] add_op2_mux_out;
wire [WORD_SIZE*2 - 1: 0] multiplier_out_fp32;
wire [WORD_SIZE*2 - 1: 0] add_op2_mux_out_fp32;
wire [WORD_SIZE*2 - 1: 0] adder_out_fp32;

assign right_out = left_in_reg;
assign bottom_out = (fsm_out_select_in == 1'b0) ? {tie_low[WORD_SIZE - 1: 0] | top_in_reg} : accumulator_reg;

// assign multiplier_out = left_in_reg * mult_op2_mux_out;
// assign adder_out = multiplier_out + add_op2_mux_out;

// use fp16 mult and fp32 add
fp16_mult fp16_mult_inst (
    .a(left_in_reg),
    .b(mult_op2_mux_out),
    .result(multiplier_out)
);
// fp16_add fp16_add_inst (
//     .a(multiplier_out),
//     .b(add_op2_mux_out),
//     .result(adder_out)
// );
fp16_to_fp32 fp16_to_fp32_a (
    .in_fp16(multiplier_out),
    .out_fp32(multiplier_out_fp32)
);
fp16_to_fp32 fp16_to_fp32_b (
    .in_fp16(add_op2_mux_out),
    .out_fp32(add_op2_mux_out_fp32)
);
fp32_add fp32_add_inst (
    .a(multiplier_out_fp32),
    .b(add_op2_mux_out_fp32),
    .result(adder_out_fp32)
);
fp32_to_fp16 fp32_to_fp16_inst (
    .in_fp32(adder_out_fp32),
    .out_fp16(adder_out)
);

assign mult_op2_mux_out = (stat_bit_in == 1'b1) ? stationary_operand_reg : top_in_reg;
assign add_op2_mux_out = (stat_bit_in == 1'b1) ? top_in_reg : accumulator_reg;

always @(posedge clk, posedge rst)
begin
     if(rst == 1'b1)
     begin
         top_in_reg <= tie_low[WORD_SIZE - 1: 0]; 
         left_in_reg <= tie_low[WORD_SIZE - 1: 0]; 
     end
     else
     begin 
        left_in_reg <= left_in;
        top_in_reg <= top_in;
     end
end

always @(posedge clk, posedge rst)
begin
    if(rst == 1'b1)
    begin
        accumulator_reg <= tie_low [WORD_SIZE - 1: 0]; 
        stationary_operand_reg <= tie_low [WORD_SIZE - 1: 0]; 
    end
    else
    begin
        if (fsm_op2_select_in == 1'b1)
        begin
            stationary_operand_reg <= top_in;
        end

        accumulator_reg <= adder_out;
    end
end

endmodule