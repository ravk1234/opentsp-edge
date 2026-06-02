// OpenTSP Milestone 7: tiny 2x2 signed INT8 systolic-style tile.
//
// This block consumes one 2x2 A tile and one 2x2 B tile per valid cycle and
// accumulates a 2x2 INT32 output tile:
//
//   C += A x B
//
// It is intentionally small so it can be simulated locally with Verilator and
// later mapped to very small FPGA boards for demonstration.

module systolic_tile_2x2 #(
    parameter IN_WIDTH = 8,
    parameter ACC_WIDTH = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_i,
    input  logic                         clear_i,

    input  logic signed [IN_WIDTH-1:0]   a00_i,
    input  logic signed [IN_WIDTH-1:0]   a01_i,
    input  logic signed [IN_WIDTH-1:0]   a10_i,
    input  logic signed [IN_WIDTH-1:0]   a11_i,

    input  logic signed [IN_WIDTH-1:0]   b00_i,
    input  logic signed [IN_WIDTH-1:0]   b01_i,
    input  logic signed [IN_WIDTH-1:0]   b10_i,
    input  logic signed [IN_WIDTH-1:0]   b11_i,

    output logic                         valid_o,
    output logic signed [ACC_WIDTH-1:0]  c00_o,
    output logic signed [ACC_WIDTH-1:0]  c01_o,
    output logic signed [ACC_WIDTH-1:0]  c10_o,
    output logic signed [ACC_WIDTH-1:0]  c11_o
);

    function automatic logic signed [ACC_WIDTH-1:0] mul_ext(
        input logic signed [IN_WIDTH-1:0] lhs,
        input logic signed [IN_WIDTH-1:0] rhs
    );
        logic signed [(2*IN_WIDTH)-1:0] product;
        begin
            product = lhs * rhs;
            mul_ext = {{(ACC_WIDTH-(2*IN_WIDTH)){product[(2*IN_WIDTH)-1]}}, product};
        end
    endfunction

    logic signed [ACC_WIDTH-1:0] c00_q;
    logic signed [ACC_WIDTH-1:0] c01_q;
    logic signed [ACC_WIDTH-1:0] c10_q;
    logic signed [ACC_WIDTH-1:0] c11_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c00_q   <= '0;
            c01_q   <= '0;
            c10_q   <= '0;
            c11_q   <= '0;
            valid_o <= 1'b0;
        end else begin
            valid_o <= valid_i;

            if (clear_i) begin
                c00_q <= '0;
                c01_q <= '0;
                c10_q <= '0;
                c11_q <= '0;
            end else if (valid_i) begin
                c00_q <= c00_q + mul_ext(a00_i, b00_i) + mul_ext(a01_i, b10_i);
                c01_q <= c01_q + mul_ext(a00_i, b01_i) + mul_ext(a01_i, b11_i);
                c10_q <= c10_q + mul_ext(a10_i, b00_i) + mul_ext(a11_i, b10_i);
                c11_q <= c11_q + mul_ext(a10_i, b01_i) + mul_ext(a11_i, b11_i);
            end
        end
    end

    assign c00_o = c00_q;
    assign c01_o = c01_q;
    assign c10_o = c10_q;
    assign c11_o = c11_q;

endmodule
