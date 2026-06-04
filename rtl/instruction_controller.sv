// OpenTSP Milestone 13: tiny RTL instruction-memory + controller FSM.
//
// This block is intentionally small. It stores a short instruction program via
// a simple host write port, then steps through the program deterministically.
// Each instruction is held valid for its programmed cycle count. The decoded
// one-hot outputs are the bridge toward a future controller that drives SRAM
// banks and the 2x2 systolic tile.

module instruction_controller #(
    parameter PROGRAM_LEN  = 64,
    parameter PC_WIDTH     = 8,
    parameter OPCODE_WIDTH = 4,
    parameter CYCLES_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Host/programming interface. Writes are accepted only while idle.
    input  logic                         host_we_i,
    input  logic [PC_WIDTH-1:0]          host_addr_i,
    input  logic [OPCODE_WIDTH-1:0]      host_opcode_i,
    input  logic [CYCLES_WIDTH-1:0]      host_cycles_i,

    // Execution control.
    input  logic                         start_i,
    input  logic [PC_WIDTH-1:0]          program_len_i,

    // Current instruction state.
    output logic                         busy_o,
    output logic                         done_o,
    output logic                         valid_o,
    output logic [PC_WIDTH-1:0]          pc_o,
    output logic [OPCODE_WIDTH-1:0]      opcode_o,
    output logic [CYCLES_WIDTH-1:0]      cycles_left_o,
    output logic [31:0]                  global_cycle_o,

    // Decoded instruction-class strobes, held high while valid_o is high.
    output logic                         load_a_o,
    output logic                         load_b_o,
    output logic                         mac_tile_o,
    output logic                         store_c_o,
    output logic                         attention_o,
    output logic                         baseline_o
);

    localparam logic [OPCODE_WIDTH-1:0] OP_NOP       = 'd0;
    localparam logic [OPCODE_WIDTH-1:0] OP_LOAD_A    = 'd1;
    localparam logic [OPCODE_WIDTH-1:0] OP_LOAD_B    = 'd2;
    localparam logic [OPCODE_WIDTH-1:0] OP_MAC_TILE  = 'd3;
    localparam logic [OPCODE_WIDTH-1:0] OP_STORE_C   = 'd4;
    localparam logic [OPCODE_WIDTH-1:0] OP_ATTENTION = 'd5;
    localparam logic [OPCODE_WIDTH-1:0] OP_BASELINE  = 'd6;

    localparam int ADDR_WIDTH = $clog2(PROGRAM_LEN);

    wire [ADDR_WIDTH-1:0] host_addr_idx;
    wire [ADDR_WIDTH-1:0] pc_idx;
    wire [ADDR_WIDTH-1:0] next_pc_idx;

    wire [PC_WIDTH-1:0] next_pc;

    assign host_addr_idx = host_addr_i[ADDR_WIDTH-1:0];
    assign pc_idx        = pc_o[ADDR_WIDTH-1:0];
    assign next_pc       = pc_o + {{(PC_WIDTH-1){1'b0}}, 1'b1};
    assign next_pc_idx   = next_pc[ADDR_WIDTH-1:0];

    logic [OPCODE_WIDTH-1:0] opcode_mem [0:PROGRAM_LEN-1];
    logic [CYCLES_WIDTH-1:0] cycles_mem [0:PROGRAM_LEN-1];

    function automatic logic [CYCLES_WIDTH-1:0] normalize_cycles(
        input logic [CYCLES_WIDTH-1:0] raw_cycles
    );
        begin
            normalize_cycles = (raw_cycles == '0) ? {{(CYCLES_WIDTH-1){1'b0}}, 1'b1} : raw_cycles;
        end
    endfunction

    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_o         <= 1'b0;
            done_o         <= 1'b0;
            valid_o        <= 1'b0;
            pc_o           <= '0;
            opcode_o       <= OP_NOP;
            cycles_left_o  <= '0;
            global_cycle_o <= 32'd0;
            for (i = 0; i < PROGRAM_LEN; i = i + 1) begin
                opcode_mem[i] <= OP_NOP;
                cycles_mem[i] <= {{(CYCLES_WIDTH-1){1'b0}}, 1'b1};
            end
        end else begin
            done_o <= 1'b0;

            if (host_we_i && !busy_o && (host_addr_i < PROGRAM_LEN[PC_WIDTH-1:0])) begin
                opcode_mem[host_addr_idx] <= host_opcode_i;
                cycles_mem[host_addr_idx] <= normalize_cycles(host_cycles_i);
            end

            if (start_i && !busy_o) begin
                pc_o           <= '0;
                opcode_o       <= OP_NOP;
                cycles_left_o  <= '0;
                global_cycle_o <= 32'd0;
                valid_o        <= 1'b0;
                busy_o         <= 1'b0;

                if (program_len_i == '0) begin
                    done_o <= 1'b1;
                end else begin
                    busy_o        <= 1'b1;
                    valid_o       <= 1'b1;
                    opcode_o      <= opcode_mem[0];
                    cycles_left_o <= normalize_cycles(cycles_mem[0]);
                end
            end else if (busy_o) begin
                global_cycle_o <= global_cycle_o + 32'd1;

                if (cycles_left_o > {{(CYCLES_WIDTH-1){1'b0}}, 1'b1}) begin
                    cycles_left_o <= cycles_left_o - {{(CYCLES_WIDTH-1){1'b0}}, 1'b1};
                end else begin
                    if ((pc_o + {{(PC_WIDTH-1){1'b0}}, 1'b1}) < program_len_i) begin
                        pc_o          <= pc_o + {{(PC_WIDTH-1){1'b0}}, 1'b1};
                        opcode_o      <= opcode_mem[next_pc_idx];
                        cycles_left_o <= normalize_cycles(cycles_mem[next_pc_idx]);
                        valid_o       <= 1'b1;
                    end else begin
                        busy_o        <= 1'b0;
                        valid_o       <= 1'b0;
                        cycles_left_o <= '0;
                        done_o        <= 1'b1;
                    end
                end
            end
        end
    end

    always_comb begin
        load_a_o    = valid_o && (opcode_o == OP_LOAD_A);
        load_b_o    = valid_o && (opcode_o == OP_LOAD_B);
        mac_tile_o  = valid_o && (opcode_o == OP_MAC_TILE);
        store_c_o   = valid_o && (opcode_o == OP_STORE_C);
        attention_o = valid_o && (opcode_o == OP_ATTENTION);
        baseline_o  = valid_o && (opcode_o == OP_BASELINE);
    end

endmodule
