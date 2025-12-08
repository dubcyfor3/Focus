`ifdef SIMULATION
`timescale 1ns / 1ps
`endif

module max_unit_fp16 #(
    parameter INDEX_WIDTH = 16,
    parameter DATA_WIDTH  = 16
)(
    input                      clk,
    input                      rst_n,

    // Pulse `update` high when in_value/in_index are valid
    input                      update,
    input  [DATA_WIDTH-1:0]    in_value,   // fp16 value
    input  [INDEX_WIDTH-1:0]   in_index,   // index associated with value

    // Current maximum (registered)
    output [DATA_WIDTH-1:0]    max_value,
    output [INDEX_WIDTH-1:0]   max_index,
    output                     max_valid   // 0 until first value is taken
);

    // Internal registers
    reg [DATA_WIDTH-1:0]    max_value_r;
    reg [INDEX_WIDTH-1:0]   max_index_r;
    reg                     max_valid_r;

    assign max_value = max_value_r;
    assign max_index = max_index_r;
    assign max_valid = max_valid_r;

    // Sequential max update
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_value_r <= {DATA_WIDTH{1'b0}};
            max_index_r <= {INDEX_WIDTH{1'b0}};
            max_valid_r <= 1'b0;
        end else if (update) begin
            // First valid sample or strictly larger than current max
            if (!max_valid_r || fp16_lt_f(max_value_r, in_value)) begin
                max_value_r <= in_value;
                max_index_r <= in_index;
                max_valid_r <= 1'b1;
            end
        end
    end

    function fp16_lt_f;
        input [15:0] a;
        input [15:0] b;

        reg a_sign, b_sign;
        reg [4:0] a_exp, b_exp;
        reg [9:0] a_frac, b_frac;

        begin
            a_sign = a[15];
            a_exp  = a[14:10];
            a_frac = a[9:0];

            b_sign = b[15];
            b_exp  = b[14:10];
            b_frac = b[9:0];

            if (a == b) begin
                fp16_lt_f = 1'b0;
            end else if (a_sign == 1'b1 && b_sign == 1'b0) begin
                fp16_lt_f = 1'b1;
            end else if (a_sign == 1'b0 && b_sign == 1'b1) begin
                fp16_lt_f = 1'b0;
            end else if (a_sign == 1'b0 && b_sign == 1'b0) begin
                // Both positive
                if (a_exp < b_exp || (a_exp == b_exp && a_frac < b_frac))
                    fp16_lt_f = 1'b1;
                else
                    fp16_lt_f = 1'b0;
            end else begin
                // Both negative (reverse compare)
                if (a_exp > b_exp || (a_exp == b_exp && a_frac > b_frac))
                    fp16_lt_f = 1'b1;
                else
                    fp16_lt_f = 1'b0;
            end
        end
    endfunction

endmodule
