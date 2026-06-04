// OpenTSP Milestone 15: instruction-driven 2x2 tile engine with tiny A/B tile memories.
//
// This module extends tile_engine_2x2 by adding simple register-file style
// storage for A and B tiles. LOAD_A and LOAD_B instructions select tiles from
// those memories using the current controller PC. MAC_TILE then feeds the latched
// A/B tiles into systolic_tile_2x2. STORE_C captures the accumulated C tile.

module tile_engine_mem_2x2 #(
    parameter IN_WIDTH     = 8,
    parameter ACC_WIDTH    = 32,
    parameter PROGRAM_LEN  = 64,
    parameter PC_WIDTH     = 8,
    parameter OPCODE_WIDTH = 4,
    parameter CYCLES_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         host_we_i,
    input  logic [PC_WIDTH-1:0]          host_addr_i,
    input  logic [OPCODE_WIDTH-1:0]      host_opcode_i,
    input  logic [CYCLES_WIDTH-1:0]      host_cycles_i,

    input  logic                         data_we_i,
    input  logic                         data_bank_i, // 0 = A memory, 1 = B memory
    input  logic [PC_WIDTH-1:0]          data_addr_i,
    input  logic signed [IN_WIDTH-1:0]   data00_i,
    input  logic signed [IN_WIDTH-1:0]   data01_i,
    input  logic signed [IN_WIDTH-1:0]   data10_i,
    input  logic signed [IN_WIDTH-1:0]   data11_i,

    input  logic                         start_i,
    input  logic [PC_WIDTH-1:0]          program_len_i,
    input  logic                         clear_tile_i,

    output logic                         busy_o,
    output logic                         done_o,
    output logic                         instr_valid_o,
    output logic [PC_WIDTH-1:0]          pc_o,
    output logic [OPCODE_WIDTH-1:0]      opcode_o,
    output logic [CYCLES_WIDTH-1:0]      cycles_left_o,
    output logic [31:0]                  global_cycle_o,

    output logic                         load_a_o,
    output logic                         load_b_o,
    output logic                         mac_tile_o,
    output logic                         store_c_o,
    output logic                         attention_o,
    output logic                         baseline_o,

    output logic signed [IN_WIDTH-1:0]   a00_o,
    output logic signed [IN_WIDTH-1:0]   a01_o,
    output logic signed [IN_WIDTH-1:0]   a10_o,
    output logic signed [IN_WIDTH-1:0]   a11_o,
    output logic signed [IN_WIDTH-1:0]   b00_o,
    output logic signed [IN_WIDTH-1:0]   b01_o,
    output logic signed [IN_WIDTH-1:0]   b10_o,
    output logic signed [IN_WIDTH-1:0]   b11_o,

    output logic                         tile_valid_o,
    output logic signed [ACC_WIDTH-1:0]  c00_o,
    output logic signed [ACC_WIDTH-1:0]  c01_o,
    output logic signed [ACC_WIDTH-1:0]  c10_o,
    output logic signed [ACC_WIDTH-1:0]  c11_o,

    output logic                         stored_valid_o,
    output logic signed [ACC_WIDTH-1:0]  stored_c00_o,
    output logic signed [ACC_WIDTH-1:0]  stored_c01_o,
    output logic signed [ACC_WIDTH-1:0]  stored_c10_o,
    output logic signed [ACC_WIDTH-1:0]  stored_c11_o
);

    localparam int ADDR_WIDTH = $clog2(PROGRAM_LEN);

    wire [ADDR_WIDTH-1:0] pc_idx;
    wire [ADDR_WIDTH-1:0] data_addr_idx;

    assign pc_idx        = pc_o[ADDR_WIDTH-1:0];
    assign data_addr_idx = data_addr_i[ADDR_WIDTH-1:0];

    logic signed [IN_WIDTH-1:0] a00_mem [0:PROGRAM_LEN-1];
    logic signed [IN_WIDTH-1:0] a01_mem [0:PROGRAM_LEN-1];
    logic signed [IN_WIDTH-1:0] a10_mem [0:PROGRAM_LEN-1];
    logic signed [IN_WIDTH-1:0] a11_mem [0:PROGRAM_LEN-1];

    logic signed [IN_WIDTH-1:0] b00_mem [0:PROGRAM_LEN-1];
    logic signed [IN_WIDTH-1:0] b01_mem [0:PROGRAM_LEN-1];
    logic signed [IN_WIDTH-1:0] b10_mem [0:PROGRAM_LEN-1];
    logic signed [IN_WIDTH-1:0] b11_mem [0:PROGRAM_LEN-1];

    logic signed [IN_WIDTH-1:0] a00_q, a01_q, a10_q, a11_q;
    logic signed [IN_WIDTH-1:0] b00_q, b01_q, b10_q, b11_q;

    integer i;

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
        .mac_tile_o(mac_tile_o),
        .store_c_o(store_c_o),
        .attention_o(attention_o),
        .baseline_o(baseline_o)
    );

    systolic_tile_2x2 #(
        .IN_WIDTH(IN_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_tile (
        .clk(clk),
        .rst_n(rst_n),
        .valid_i(mac_tile_o),
        .clear_i(clear_tile_i || start_i),
        .a00_i(a00_q),
        .a01_i(a01_q),
        .a10_i(a10_q),
        .a11_i(a11_q),
        .b00_i(b00_q),
        .b01_i(b01_q),
        .b10_i(b10_q),
        .b11_i(b11_q),
        .valid_o(tile_valid_o),
        .c00_o(c00_o),
        .c01_o(c01_o),
        .c10_o(c10_o),
        .c11_o(c11_o)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a00_q <= '0; a01_q <= '0; a10_q <= '0; a11_q <= '0;
            b00_q <= '0; b01_q <= '0; b10_q <= '0; b11_q <= '0;
            stored_valid_o <= 1'b0;
            stored_c00_o <= '0; stored_c01_o <= '0; stored_c10_o <= '0; stored_c11_o <= '0;
            for (i = 0; i < PROGRAM_LEN; i = i + 1) begin
                a00_mem[i] <= '0; a01_mem[i] <= '0; a10_mem[i] <= '0; a11_mem[i] <= '0;
                b00_mem[i] <= '0; b01_mem[i] <= '0; b10_mem[i] <= '0; b11_mem[i] <= '0;
            end
        end else begin
            if (data_we_i && !busy_o && (data_addr_i < PROGRAM_LEN[PC_WIDTH-1:0])) begin
                if (data_bank_i == 1'b0) begin
                    a00_mem[data_addr_idx] <= data00_i;
                    a01_mem[data_addr_idx] <= data01_i;
                    a10_mem[data_addr_idx] <= data10_i;
                    a11_mem[data_addr_idx] <= data11_i;
                end else begin
                    b00_mem[data_addr_idx] <= data00_i;
                    b01_mem[data_addr_idx] <= data01_i;
                    b10_mem[data_addr_idx] <= data10_i;
                    b11_mem[data_addr_idx] <= data11_i;
                end
            end

            if (clear_tile_i || start_i) begin
                stored_valid_o <= 1'b0;
                stored_c00_o <= '0; stored_c01_o <= '0; stored_c10_o <= '0; stored_c11_o <= '0;
            end

            if (load_a_o) begin
                a00_q <= a00_mem[pc_idx];
                a01_q <= a01_mem[pc_idx];
                a10_q <= a10_mem[pc_idx];
                a11_q <= a11_mem[pc_idx];
            end

            if (load_b_o) begin
                b00_q <= b00_mem[pc_idx];
                b01_q <= b01_mem[pc_idx];
                b10_q <= b10_mem[pc_idx];
                b11_q <= b11_mem[pc_idx];
            end

            if (store_c_o) begin
                stored_valid_o <= 1'b1;
                stored_c00_o <= c00_o;
                stored_c01_o <= c01_o;
                stored_c10_o <= c10_o;
                stored_c11_o <= c11_o;
            end
        end
    end

    assign a00_o = a00_q;
    assign a01_o = a01_q;
    assign a10_o = a10_q;
    assign a11_o = a11_q;
    assign b00_o = b00_q;
    assign b01_o = b01_q;
    assign b10_o = b10_q;
    assign b11_o = b11_q;

endmodule
