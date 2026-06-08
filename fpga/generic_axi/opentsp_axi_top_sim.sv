// SPDX-License-Identifier: Apache-2.0
// OpenTSP Edge: generic FPGA-facing AXI-lite top simulation wrapper.
//
// This top-level module is intentionally simulation-friendly. It exposes the
// same AXI-lite-style register interface that a future FPGA/cloud-FPGA wrapper
// will use, and internally connects it to the OpenTSP 2x2 tile engine path.

module opentsp_axi_top_sim #(
    parameter int ADDR_WIDTH = 8,
    parameter int DATA_WIDTH = 32
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,

    // AXI-lite write address channel
    input  logic [ADDR_WIDTH-1:0]  awaddr_i,
    input  logic                   awvalid_i,
    output logic                   awready_o,

    // AXI-lite write data channel
    input  logic [DATA_WIDTH-1:0]  wdata_i,
    input  logic                   wvalid_i,
    output logic                   wready_o,

    // AXI-lite write response channel
    output logic [1:0]             bresp_o,
    output logic                   bvalid_o,
    input  logic                   bready_i,

    // AXI-lite read address channel
    input  logic [ADDR_WIDTH-1:0]  araddr_i,
    input  logic                   arvalid_i,
    output logic                   arready_o,

    // AXI-lite read data channel
    output logic [DATA_WIDTH-1:0]  rdata_o,
    output logic [1:0]             rresp_o,
    output logic                   rvalid_o,
    input  logic                   rready_i
);

    // These are internal status/debug outputs from the tile engine.
    // The FPGA-facing top does not expose them yet, but Verilator requires
    // us to connect all output ports explicitly.
    logic engine_busy_unused;
    logic engine_done_unused;
    logic stored_valid_unused;

    axi_lite_tile_engine_2x2 #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_axi_lite_tile_engine_2x2 (
        .clk_i(clk_i),
        .rst_ni(rst_ni),

        .awaddr_i(awaddr_i),
        .awvalid_i(awvalid_i),
        .awready_o(awready_o),

        .wdata_i(wdata_i),
        .wvalid_i(wvalid_i),
        .wready_o(wready_o),

        .bresp_o(bresp_o),
        .bvalid_o(bvalid_o),
        .bready_i(bready_i),

        .araddr_i(araddr_i),
        .arvalid_i(arvalid_i),
        .arready_o(arready_o),

        .rdata_o(rdata_o),
        .rresp_o(rresp_o),
        .rvalid_o(rvalid_o),
        .rready_i(rready_i),

        .engine_busy_o(engine_busy_unused),
        .engine_done_o(engine_done_unused),
        .stored_valid_o(stored_valid_unused)
    );

endmodule