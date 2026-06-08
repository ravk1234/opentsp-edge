// SPDX-License-Identifier: Apache-2.0
// OpenTSP Edge: generic AXI-lite top-level wrapper.
//
// This module provides a target-neutral top-level boundary around the
// AXI-lite-controlled 2x2 tile engine.

module opentsp_axi_top #(
    parameter int ADDR_WIDTH = 8,
    parameter int DATA_WIDTH = 32
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,

    // AXI-lite-style write address channel
    input  logic [ADDR_WIDTH-1:0]  awaddr_i,
    input  logic                   awvalid_i,
    output logic                   awready_o,

    // AXI-lite-style write data channel
    input  logic [DATA_WIDTH-1:0]  wdata_i,
    input  logic                   wvalid_i,
    output logic                   wready_o,

    // AXI-lite-style write response channel
    output logic [1:0]             bresp_o,
    output logic                   bvalid_o,
    input  logic                   bready_i,

    // AXI-lite-style read address channel
    input  logic [ADDR_WIDTH-1:0]  araddr_i,
    input  logic                   arvalid_i,
    output logic                   arready_o,

    // AXI-lite-style read data channel
    output logic [DATA_WIDTH-1:0]  rdata_o,
    output logic [1:0]             rresp_o,
    output logic                   rvalid_o,
    input  logic                   rready_i
);

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
        .rready_i(rready_i)
    );

endmodule
