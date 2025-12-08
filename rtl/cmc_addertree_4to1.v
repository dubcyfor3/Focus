module cmc_addertree_4to1 (
    input  logic signed [15:0] in0,
    input  logic signed [15:0] in1,
    input  logic signed [15:0] in2,
    input  logic signed [15:0] in3,
    output logic signed [17:0] sum
);

    // First level of addition
    logic signed [16:0] sum0, sum1;

    assign sum0 = in0 + in1;
    assign sum1 = in2 + in3;

    // Second level of addition
    assign sum = sum0 + sum1;

endmodule
