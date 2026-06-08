// OpenTSP Milestone 20: host-style register interface for the 2x2 tile engine.
//
// This wrapper exposes a tiny memory-mapped register surface around
// tile_engine_mem_2x2. It is intentionally simple and meant for local RTL
// simulation before adding AXI/PCIe or board-specific host interfaces.

module host_tile_engine_2x2 #(
    parameter IN_WIDTH     = 8,
    parameter ACC_WIDTH    = 32,
    parameter PROGRAM_LEN  = 64,
    parameter PC_WIDTH     = 8,
    parameter OPCODE_WIDTH = 4,
    parameter CYCLES_WIDTH = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    reg_we_i,
    input  logic                    reg_re_i,
    input  logic [7:0]              reg_addr_i,
    input  logic [31:0]             reg_wdata_i,
    output logic [31:0]             reg_rdata_o,
    output logic                    reg_ready_o,

    output logic                    busy_o,
    output logic                    done_o,
    output logic                    stored_valid_o
);

    localparam logic [7:0] ADDR_CONTROL     = 8'h00;
    localparam logic [7:0] ADDR_STATUS      = 8'h04;
    localparam logic [7:0] ADDR_PROGRAM_LEN = 8'h08;
    localparam logic [7:0] ADDR_PC          = 8'h0c;
    localparam logic [7:0] ADDR_INSTR_ADDR  = 8'h10;
    localparam logic [7:0] ADDR_INSTR_WORD  = 8'h14;
    localparam logic [7:0] ADDR_DATA_ADDR   = 8'h18;
    localparam logic [7:0] ADDR_DATA_BANK   = 8'h1c;
    localparam logic [7:0] ADDR_DATA_WORD   = 8'h20;
    localparam logic [7:0] ADDR_C00         = 8'h30;
    localparam logic [7:0] ADDR_C01         = 8'h34;
    localparam logic [7:0] ADDR_C10         = 8'h38;
    localparam logic [7:0] ADDR_C11         = 8'h3c;

    logic [PC_WIDTH-1:0]          program_len_q;
    logic [PC_WIDTH-1:0]          instr_addr_q;
    logic [PC_WIDTH-1:0]          data_addr_q;
    logic                         data_bank_q;

    logic                         start_q;
    logic                         clear_tile_q;
    logic                         done_sticky_q;

    logic                         instr_we_q;
    logic [PC_WIDTH-1:0]          instr_write_addr_q;
    logic [OPCODE_WIDTH-1:0]      instr_write_opcode_q;
    logic [CYCLES_WIDTH-1:0]      instr_write_cycles_q;

    logic                         data_we_q;
    logic                         data_write_bank_q;
    logic [PC_WIDTH-1:0]          data_write_addr_q;
    logic signed [IN_WIDTH-1:0]   data00_q;
    logic signed [IN_WIDTH-1:0]   data01_q;
    logic signed [IN_WIDTH-1:0]   data10_q;
    logic signed [IN_WIDTH-1:0]   data11_q;

    logic                         engine_done;
    logic                         instr_valid;
    logic [PC_WIDTH-1:0]          pc;
    logic [OPCODE_WIDTH-1:0]      opcode;
    logic [CYCLES_WIDTH-1:0]      cycles_left;
    logic [31:0]                  global_cycle;
    logic                         load_a;
    logic                         load_b;
    logic                         mac_tile;
    logic                         store_c;
    logic                         attention;
    logic                         baseline;
    logic signed [IN_WIDTH-1:0]   a00;
    logic signed [IN_WIDTH-1:0]   a01;
    logic signed [IN_WIDTH-1:0]   a10;
    logic signed [IN_WIDTH-1:0]   a11;
    logic signed [IN_WIDTH-1:0]   b00;
    logic signed [IN_WIDTH-1:0]   b01;
    logic signed [IN_WIDTH-1:0]   b10;
    logic signed [IN_WIDTH-1:0]   b11;
    logic                         tile_valid;
    logic signed [ACC_WIDTH-1:0]  c00;
    logic signed [ACC_WIDTH-1:0]  c01;
    logic signed [ACC_WIDTH-1:0]  c10;
    logic signed [ACC_WIDTH-1:0]  c11;
    logic signed [ACC_WIDTH-1:0]  stored_c00;
    logic signed [ACC_WIDTH-1:0]  stored_c01;
    logic signed [ACC_WIDTH-1:0]  stored_c10;
    logic signed [ACC_WIDTH-1:0]  stored_c11;

    tile_engine_mem_2x2 #(
        .IN_WIDTH(IN_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PROGRAM_LEN(PROGRAM_LEN),
        .PC_WIDTH(PC_WIDTH),
        .OPCODE_WIDTH(OPCODE_WIDTH),
        .CYCLES_WIDTH(CYCLES_WIDTH)
    ) u_engine (
        .clk(clk),
        .rst_n(rst_n),
        .host_we_i(instr_we_q),
        .host_addr_i(instr_write_addr_q),
        .host_opcode_i(instr_write_opcode_q),
        .host_cycles_i(instr_write_cycles_q),
        .data_we_i(data_we_q),
        .data_bank_i(data_write_bank_q),
        .data_addr_i(data_write_addr_q),
        .data00_i(data00_q),
        .data01_i(data01_q),
        .data10_i(data10_q),
        .data11_i(data11_q),
        .start_i(start_q),
        .program_len_i(program_len_q),
        .clear_tile_i(clear_tile_q),
        .busy_o(busy_o),
        .done_o(engine_done),
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

    assign done_o = done_sticky_q;
    assign reg_ready_o = 1'b1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            program_len_q <= '0;
            instr_addr_q <= '0;
            data_addr_q <= '0;
            data_bank_q <= 1'b0;
            start_q <= 1'b0;
            clear_tile_q <= 1'b0;
            done_sticky_q <= 1'b0;
            instr_we_q <= 1'b0;
            instr_write_addr_q <= '0;
            instr_write_opcode_q <= '0;
            instr_write_cycles_q <= '0;
            data_we_q <= 1'b0;
            data_write_bank_q <= 1'b0;
            data_write_addr_q <= '0;
            data00_q <= '0;
            data01_q <= '0;
            data10_q <= '0;
            data11_q <= '0;
        end else begin
            start_q <= 1'b0;
            clear_tile_q <= 1'b0;
            instr_we_q <= 1'b0;
            data_we_q <= 1'b0;

            if (engine_done) begin
                done_sticky_q <= 1'b1;
            end

            if (reg_we_i) begin
                unique case (reg_addr_i)
                    ADDR_CONTROL: begin
                        if (reg_wdata_i[0]) begin
                            start_q <= 1'b1;
                            done_sticky_q <= 1'b0;
                        end
                        if (reg_wdata_i[1]) begin
                            clear_tile_q <= 1'b1;
                            done_sticky_q <= 1'b0;
                        end
                    end
                    ADDR_PROGRAM_LEN: begin
                        program_len_q <= reg_wdata_i[PC_WIDTH-1:0];
                    end
                    ADDR_INSTR_ADDR: begin
                        instr_addr_q <= reg_wdata_i[PC_WIDTH-1:0];
                    end
                    ADDR_INSTR_WORD: begin
                        instr_write_addr_q <= instr_addr_q;
                        instr_write_opcode_q <= reg_wdata_i[OPCODE_WIDTH-1:0];
                        instr_write_cycles_q <= reg_wdata_i[23:8];
                        instr_we_q <= 1'b1;
                    end
                    ADDR_DATA_ADDR: begin
                        data_addr_q <= reg_wdata_i[PC_WIDTH-1:0];
                    end
                    ADDR_DATA_BANK: begin
                        data_bank_q <= reg_wdata_i[0];
                    end
                    ADDR_DATA_WORD: begin
                        data_write_addr_q <= data_addr_q;
                        data_write_bank_q <= data_bank_q;
                        data00_q <= reg_wdata_i[7:0];
                        data01_q <= reg_wdata_i[15:8];
                        data10_q <= reg_wdata_i[23:16];
                        data11_q <= reg_wdata_i[31:24];
                        data_we_q <= 1'b1;
                    end
                    default: begin
                        // Unknown writes are ignored by design.
                    end
                endcase
            end
        end
    end

    always_comb begin
        reg_rdata_o = 32'h00000000;
        if (reg_re_i) begin
            unique case (reg_addr_i)
                ADDR_STATUS: begin
                    reg_rdata_o = {
                        24'h000000,
                        baseline,
                        attention,
                        store_c,
                        mac_tile,
                        instr_valid,
                        stored_valid_o,
                        done_sticky_q,
                        busy_o
                    };
                end
                ADDR_PROGRAM_LEN: reg_rdata_o = {{(32-PC_WIDTH){1'b0}}, program_len_q};
                ADDR_PC:          reg_rdata_o = {{(32-PC_WIDTH){1'b0}}, pc};
                ADDR_INSTR_ADDR:  reg_rdata_o = {{(32-PC_WIDTH){1'b0}}, instr_addr_q};
                ADDR_DATA_ADDR:   reg_rdata_o = {{(32-PC_WIDTH){1'b0}}, data_addr_q};
                ADDR_DATA_BANK:   reg_rdata_o = {31'h00000000, data_bank_q};
                ADDR_C00:         reg_rdata_o = stored_c00;
                ADDR_C01:         reg_rdata_o = stored_c01;
                ADDR_C10:         reg_rdata_o = stored_c10;
                ADDR_C11:         reg_rdata_o = stored_c11;
                default:          reg_rdata_o = 32'h00000000;
            endcase
        end
    end

endmodule
