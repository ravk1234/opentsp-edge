// SPDX-License-Identifier: Apache-2.0
// OpenTSP Edge: placeholder AWS F2 custom-logic top.
//
// This is intentionally not a complete AWS F2 custom logic implementation.
// It documents the intended top-level boundary for future AWS F2 integration.
//
// A future version should map AWS shell AXI/PCIe-facing signals into the
// generic OpenTSP AXI-lite tile-engine top.

module cl_opentsp_top #(
    parameter int ADDR_WIDTH = 8,
    parameter int DATA_WIDTH = 32
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,

    // Placeholder AXI-lite-style write address channel
    input  logic [ADDR_WIDTH-1:0]  awaddr_i,
    input  logic                   awvalid_i,
    output logic                   awready_o,

    // Placeholder AXI-lite-style write data channel
    input  logic [DATA_WIDTH-1:0]  wdata_i,
    input  logic                   wvalid_i,
    output logic                   wready_o,

    // Placeholder AXI-lite-style write response channel
    output logic [1:0]             bresp_o,
    output logic                   bvalid_o,
    input  logic                   bready_i,

    // Placeholder AXI-lite-style read address channel
    input  logic [ADDR_WIDTH-1:0]  araddr_i,
    input  logic                   arvalid_i,
    output logic                   arready_o,

    // Placeholder AXI-lite-style read data channel
    output logic [DATA_WIDTH-1:0]  rdata_o,
    output logic [1:0]             rresp_o,
    output logic                   rvalid_o,
    input  logic                   rready_i
);

    // For now this AWS-facing placeholder simply instantiates the generic
    // OpenTSP AXI-lite top. A real AWS F2 shell integration will replace these
    // placeholder ports with the AWS shell interface.
    opentsp_axi_top #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_opentsp_axi_top (
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
