// OpenTSP Milestone 14: instruction-controller-driven 2x2 tile engine.
//
// This module connects the RTL instruction controller to the 2x2 systolic tile.
// It is intentionally small: the controller decodes abstract OpenTSP opcodes and
// the MAC_TILE decoded signal drives the tile's valid input. LOAD/STORE/ATTENTION
// do not yet move real SRAM data; they are exposed so the next milestone can add
// instruction-driven memory and data routing.

module tile_engine_2x2 #(
    parameter IN_WIDTH     = 8,
    parameter ACC_WIDTH    = 32,
    parameter PROGRAM_LEN  = 64,
    parameter PC_WIDTH     = 8,
    parameter OPCODE_WIDTH = 4,
    parameter CYCLES_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Host/programming interface passed into instruction_controller.
    input  logic                         host_we_i,
    input  logic [PC_WIDTH-1:0]          host_addr_i,
    input  logic [OPCODE_WIDTH-1:0]      host_opcode_i,
    input  logic [CYCLES_WIDTH-1:0]      host_cycles_i,

    // Execution control.
    input  logic                         start_i,
    input  logic [PC_WIDTH-1:0]          program_len_i,
    input  logic                         clear_tile_i,

    // Current A/B tile values. A later memory milestone will replace these with
    // SRAM-bank outputs driven by LOAD_A and LOAD_B instructions.
    input  logic signed [IN_WIDTH-1:0]   a00_i,
    input  logic signed [IN_WIDTH-1:0]   a01_i,
    input  logic signed [IN_WIDTH-1:0]   a10_i,
    input  logic signed [IN_WIDTH-1:0]   a11_i,

    input  logic signed [IN_WIDTH-1:0]   b00_i,
    input  logic signed [IN_WIDTH-1:0]   b01_i,
    input  logic signed [IN_WIDTH-1:0]   b10_i,
    input  logic signed [IN_WIDTH-1:0]   b11_i,

    // Controller state.
    output logic                         busy_o,
    output logic                         done_o,
    output logic                         instr_valid_o,
    output logic [PC_WIDTH-1:0]          pc_o,
    output logic [OPCODE_WIDTH-1:0]      opcode_o,
    output logic [CYCLES_WIDTH-1:0]      cycles_left_o,
    output logic [31:0]                  global_cycle_o,

    // Decoded controller outputs.
    output logic                         load_a_o,
    output logic                         load_b_o,
    output logic                         mac_tile_o,
    output logic                         store_c_o,
    output logic                         attention_o,
    output logic                         baseline_o,

    // Tile outputs.
    output logic                         tile_valid_o,
    output logic signed [ACC_WIDTH-1:0]  c00_o,
    output logic signed [ACC_WIDTH-1:0]  c01_o,
    output logic signed [ACC_WIDTH-1:0]  c10_o,
    output logic signed [ACC_WIDTH-1:0]  c11_o
);

    logic controller_mac_tile;

    instruction_controller #(
        .PROGRAM_LEN(PROGRAM_LEN),
        .PC_WIDTH(PC_WIDTH),
        .OPCODE_WIDTH(OPCODE_WIDTH),
        .CYCLES_WIDTH(CYCLES_WIDTH)
    ) u_controller (
        .clk(clk),
        .rst_n(rst_n),
        .host_we_i(host_we_i),
        .host_addr_i(host_addr_i),
        .host_opcode_i(host_opcode_i),
        .host_cycles_i(host_cycles_i),
        .start_i(start_i),
        .program_len_i(program_len_i),
        .busy_o(busy_o),
        .done_o(done_o),
        .valid_o(instr_valid_o),
        .pc_o(pc_o),
        .opcode_o(opcode_o),
        .cycles_left_o(cycles_left_o),
        .global_cycle_o(global_cycle_o),
        .load_a_o(load_a_o),
        .load_b_o(load_b_o),
        .mac_tile_o(controller_mac_tile),
        .store_c_o(store_c_o),
        .attention_o(attention_o),
        .baseline_o(baseline_o)
    );

    assign mac_tile_o = controller_mac_tile;

    systolic_tile_2x2 #(
        .IN_WIDTH(IN_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_tile (
        .clk(clk),
        .rst_n(rst_n),
        .valid_i(controller_mac_tile),
        .clear_i(clear_tile_i || start_i),
        .a00_i(a00_i),
        .a01_i(a01_i),
        .a10_i(a10_i),
        .a11_i(a11_i),
        .b00_i(b00_i),
        .b01_i(b01_i),
        .b10_i(b10_i),
        .b11_i(b11_i),
        .valid_o(tile_valid_o),
        .c00_o(c00_o),
        .c01_o(c01_o),
        .c10_o(c10_o),
        .c11_o(c11_o)
    );

endmodule
