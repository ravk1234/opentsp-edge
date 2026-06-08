// SPDX-License-Identifier: Apache-2.0
// OpenTSP Edge: AXI-lite-style wrapper connected to the 2x2 tile engine.
//
// This top-level prototype connects the AXI-lite-style register decoder to the
// existing host-programmable tile engine. It is still intentionally small and
// simulation-friendly, but now AXI-lite-style writes can load instructions and
// A/B data, start execution, and read back C outputs through registers.

module axi_lite_tile_engine_2x2 #(
    parameter int ADDR_WIDTH   = 8,
    parameter int DATA_WIDTH   = 32,
    parameter int IN_WIDTH     = 8,
    parameter int ACC_WIDTH    = 32,
    parameter int PROGRAM_LEN  = 64,
    parameter int PC_WIDTH     = 8,
    parameter int OPCODE_WIDTH = 4,
    parameter int CYCLES_WIDTH = 16
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
    input  logic                   rready_i,

    // Debug/status outputs for testbenches and future integration.
    output logic                   engine_busy_o,
    output logic                   engine_done_o,
    output logic                   stored_valid_o
);

    logic                  start;
    logic                  clear;
    logic [7:0]            program_len;

    logic [7:0]            instr_addr;
    logic [DATA_WIDTH-1:0] instr_word;
    logic                  instr_write;

    logic [7:0]            data_addr;
    logic [1:0]            data_bank;
    logic [DATA_WIDTH-1:0] data_word;
    logic                  data_write;

    logic                  engine_done_pulse;
    logic                  done_sticky_q;

    logic                  instr_valid;
    logic [PC_WIDTH-1:0]   pc;
    logic [OPCODE_WIDTH-1:0] opcode;
    logic [CYCLES_WIDTH-1:0] cycles_left;
    logic [31:0]           global_cycle;
    logic                  load_a;
    logic                  load_b;
    logic                  mac_tile;
    logic                  store_c;
    logic                  attention;
    logic                  baseline;

    logic signed [IN_WIDTH-1:0]  a00;
    logic signed [IN_WIDTH-1:0]  a01;
    logic signed [IN_WIDTH-1:0]  a10;
    logic signed [IN_WIDTH-1:0]  a11;
    logic signed [IN_WIDTH-1:0]  b00;
    logic signed [IN_WIDTH-1:0]  b01;
    logic signed [IN_WIDTH-1:0]  b10;
    logic signed [IN_WIDTH-1:0]  b11;
    logic                        tile_valid;
    logic signed [ACC_WIDTH-1:0] c00;
    logic signed [ACC_WIDTH-1:0] c01;
    logic signed [ACC_WIDTH-1:0] c10;
    logic signed [ACC_WIDTH-1:0] c11;
    logic signed [ACC_WIDTH-1:0] stored_c00;
    logic signed [ACC_WIDTH-1:0] stored_c01;
    logic signed [ACC_WIDTH-1:0] stored_c10;
    logic signed [ACC_WIDTH-1:0] stored_c11;

    assign engine_done_o = done_sticky_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            done_sticky_q <= 1'b0;
        end else begin
            if (clear || start) begin
                done_sticky_q <= 1'b0;
            end else if (engine_done_pulse) begin
                done_sticky_q <= 1'b1;
            end
        end
    end

    axi_lite_host_regs #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_regs (
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
        .engine_busy_i(engine_busy_o),
        .engine_done_i(done_sticky_q),
        .c00_i(stored_c00),
        .c01_i(stored_c01),
        .c10_i(stored_c10),
        .c11_i(stored_c11),
        .start_o(start),
        .clear_o(clear),
        .program_len_o(program_len),
        .instr_addr_o(instr_addr),
        .instr_word_o(instr_word),
        .instr_write_o(instr_write),
        .data_addr_o(data_addr),
        .data_bank_o(data_bank),
        .data_word_o(data_word),
        .data_write_o(data_write)
    );

    tile_engine_mem_2x2 #(
        .IN_WIDTH(IN_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PROGRAM_LEN(PROGRAM_LEN),
        .PC_WIDTH(PC_WIDTH),
        .OPCODE_WIDTH(OPCODE_WIDTH),
        .CYCLES_WIDTH(CYCLES_WIDTH)
    ) u_engine (
        .clk(clk_i),
        .rst_n(rst_ni),
        .host_we_i(instr_write),
        .host_addr_i(instr_addr[PC_WIDTH-1:0]),
        .host_opcode_i(instr_word[OPCODE_WIDTH-1:0]),
        .host_cycles_i(instr_word[23:8]),
        .data_we_i(data_write),
        .data_bank_i(data_bank[0]),
        .data_addr_i(data_addr[PC_WIDTH-1:0]),
        .data00_i(data_word[7:0]),
        .data01_i(data_word[15:8]),
        .data10_i(data_word[23:16]),
        .data11_i(data_word[31:24]),
        .start_i(start),
        .program_len_i(program_len[PC_WIDTH-1:0]),
        .clear_tile_i(clear),
        .busy_o(engine_busy_o),
        .done_o(engine_done_pulse),
        .instr_valid_o(instr_valid),
        .pc_o(pc),
        .opcode_o(opcode),
        .cycles_left_o(cycles_left),
        .global_cycle_o(global_cycle),
        .load_a_o(load_a),
        .load_b_o(load_b),
        .mac_tile_o(mac_tile),
        .store_c_o(store_c),
        .attention_o(attention),
        .baseline_o(baseline),
        .a00_o(a00),
        .a01_o(a01),
        .a10_o(a10),
        .a11_o(a11),
        .b00_o(b00),
        .b01_o(b01),
        .b10_o(b10),
        .b11_o(b11),
        .tile_valid_o(tile_valid),
        .c00_o(c00),
        .c01_o(c01),
        .c10_o(c10),
        .c11_o(c11),
        .stored_valid_o(stored_valid_o),
        .stored_c00_o(stored_c00),
        .stored_c01_o(stored_c01),
        .stored_c10_o(stored_c10),
        .stored_c11_o(stored_c11)
    );

endmodule
