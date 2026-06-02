// OpenTSP local MVP: tiny signed 8-bit MAC unit.
// This is a starter RTL block for future Verilator/cocotb testing.

module mac_unit #(
    parameter IN_WIDTH = 8,
    parameter ACC_WIDTH = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_i,
    input  logic signed [IN_WIDTH-1:0]   a_i,
    input  logic signed [IN_WIDTH-1:0]   b_i,
    input  logic                         clear_i,
    output logic                         valid_o,
    output logic signed [ACC_WIDTH-1:0]  acc_o
);

    logic signed [ACC_WIDTH-1:0] acc_q;
    logic signed [(2*IN_WIDTH)-1:0] product;

    assign product = a_i * b_i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_q   <= '0;
            valid_o <= 1'b0;
        end else begin
            valid_o <= valid_i;

            if (clear_i) begin
                acc_q <= '0;
            end else if (valid_i) begin
                acc_q <= acc_q + {{(ACC_WIDTH-(2*IN_WIDTH)){product[(2*IN_WIDTH)-1]}}, product};
            end
        end
    end

    assign acc_o = acc_q;

endmodule
