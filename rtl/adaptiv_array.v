// https://github.com/scalesim-project/scale-sim-v2/blob/main/code-examples/systolic-array-rtl/systolic_array_8bit_MAC_PE/traditional_systolic.v

//`include "./traditional_mac.v"

module adaptiv_array
#(
    parameter ROWS = 64,
    parameter COLS = 16,
    parameter WORD_SIZE = 16

) (
    clk,
    rst,

    ctl_stat_bit_in, 
    ctl_dummy_fsm_op2_select_in,
    ctl_dummy_fsm_out_select_in,

    left_in_bus,
    top_in_bus,
    bottom_out_bus,
    right_out_bus
);

input clk;
input rst;

input [ROWS * WORD_SIZE - 1: 0] left_in_bus;
input [COLS * WORD_SIZE - 1: 0] top_in_bus;
output [ROWS * COLS * WORD_SIZE - 1: 0] right_out_bus;
output [ROWS * COLS * WORD_SIZE - 1: 0] bottom_out_bus;

input ctl_stat_bit_in; 
input ctl_dummy_fsm_op2_select_in;
input ctl_dummy_fsm_out_select_in;

genvar r, c;

generate
  for (r = 0; r < ROWS; r = r + 1) begin
    for (c = 0; c < COLS; c = c + 1) begin
      wire [WORD_SIZE-1:0] left_in  = left_in_bus[(r+1)*WORD_SIZE - 1 -: WORD_SIZE];
      wire [WORD_SIZE-1:0] top_in   = top_in_bus[(c+1)*WORD_SIZE - 1 -: WORD_SIZE];
      wire [WORD_SIZE-1:0] right_out;
      wire [WORD_SIZE-1:0] bottom_out;

      traditional_mac #(
        .WORD_SIZE(WORD_SIZE)
      ) pe (
        .clk(clk),
        .rst(rst),
        .fsm_op2_select_in(ctl_dummy_fsm_op2_select_in),
        .fsm_out_select_in(ctl_dummy_fsm_out_select_in),
        .stat_bit_in(ctl_stat_bit_in),
        .left_in(left_in),
        .top_in(top_in),
        .right_out(right_out),
        .bottom_out(bottom_out)
      );

      // Flattened bus assignment:
      assign right_out_bus[((r * COLS + c + 1) * WORD_SIZE - 1) -: WORD_SIZE] = right_out;
      assign bottom_out_bus[((r * COLS + c + 1) * WORD_SIZE - 1) -: WORD_SIZE] = bottom_out;
    end
  end
endgenerate

endmodule
