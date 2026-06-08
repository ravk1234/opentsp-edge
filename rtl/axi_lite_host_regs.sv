// SPDX-License-Identifier: Apache-2.0
// OpenTSP Edge: minimal AXI-lite-style host register wrapper prototype.
//
// This module is intentionally small and simulation-friendly. It models the
// register protocol shape we will later use to connect an FPGA/SoC host to the
// OpenTSP tile engine. It does not instantiate the tile engine yet; instead it
// decodes host-visible writes into clean sideband signals.

module axi_lite_host_regs #(
    parameter int ADDR_WIDTH = 8,
    parameter int DATA_WIDTH = 32
) (
    input  logic                   clk_i,
    input  logic                   rst_ni,

    // Write address channel
    input  logic [ADDR_WIDTH-1:0]  awaddr_i,
    input  logic                   awvalid_i,
    output logic                   awready_o,

    // Write data channel
    input  logic [DATA_WIDTH-1:0]  wdata_i,
    input  logic                   wvalid_i,
    output logic                   wready_o,

    // Write response channel
    output logic [1:0]             bresp_o,
    output logic                   bvalid_o,
    input  logic                   bready_i,

    // Read address channel
    input  logic [ADDR_WIDTH-1:0]  araddr_i,
    input  logic                   arvalid_i,
    output logic                   arready_o,

    // Read data channel
    output logic [DATA_WIDTH-1:0]  rdata_o,
    output logic [1:0]             rresp_o,
    output logic                   rvalid_o,
    input  logic                   rready_i,

    // Engine status inputs
    input  logic                   engine_busy_i,
    input  logic                   engine_done_i,
    input  logic [DATA_WIDTH-1:0]  c00_i,
    input  logic [DATA_WIDTH-1:0]  c01_i,
    input  logic [DATA_WIDTH-1:0]  c10_i,
    input  logic [DATA_WIDTH-1:0]  c11_i,

    // Decoded host control outputs
    output logic                   start_o,
    output logic                   clear_o,
    output logic [7:0]             program_len_o,

    output logic [7:0]             instr_addr_o,
    output logic [DATA_WIDTH-1:0]  instr_word_o,
    output logic                   instr_write_o,

    output logic [7:0]             data_addr_o,
    output logic [1:0]             data_bank_o,
    output logic [DATA_WIDTH-1:0]  data_word_o,
    output logic                   data_write_o
);

    localparam logic [ADDR_WIDTH-1:0] REG_CONTROL     = 8'h00;
    localparam logic [ADDR_WIDTH-1:0] REG_STATUS      = 8'h04;
    localparam logic [ADDR_WIDTH-1:0] REG_PROGRAM_LEN = 8'h08;
    localparam logic [ADDR_WIDTH-1:0] REG_INSTR_ADDR  = 8'h10;
    localparam logic [ADDR_WIDTH-1:0] REG_INSTR_WORD  = 8'h14;
    localparam logic [ADDR_WIDTH-1:0] REG_DATA_ADDR   = 8'h18;
    localparam logic [ADDR_WIDTH-1:0] REG_DATA_BANK   = 8'h1C;
    localparam logic [ADDR_WIDTH-1:0] REG_DATA_WORD   = 8'h20;
    localparam logic [ADDR_WIDTH-1:0] REG_C00         = 8'h30;
    localparam logic [ADDR_WIDTH-1:0] REG_C01         = 8'h34;
    localparam logic [ADDR_WIDTH-1:0] REG_C10         = 8'h38;
    localparam logic [ADDR_WIDTH-1:0] REG_C11         = 8'h3C;

    logic [7:0] instr_addr_q;
    logic [7:0] data_addr_q;
    logic [1:0] data_bank_q;
    logic [7:0] program_len_q;

    wire write_fire;
    wire read_fire;

    assign awready_o = 1'b1;
    assign wready_o  = 1'b1;
    assign arready_o = 1'b1;

    assign write_fire = awvalid_i && wvalid_i;
    assign read_fire  = arvalid_i;

    assign bresp_o = 2'b00;
    assign rresp_o = 2'b00;

    assign program_len_o = program_len_q;
    assign instr_addr_o  = instr_addr_q;
    assign data_addr_o   = data_addr_q;
    assign data_bank_o   = data_bank_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            bvalid_o      <= 1'b0;
            rvalid_o      <= 1'b0;
            rdata_o       <= '0;

            start_o       <= 1'b0;
            clear_o       <= 1'b0;
            instr_write_o <= 1'b0;
            data_write_o  <= 1'b0;
            instr_word_o  <= '0;
            data_word_o   <= '0;

            instr_addr_q  <= '0;
            data_addr_q   <= '0;
            data_bank_q   <= '0;
            program_len_q <= '0;
        end else begin
            start_o       <= 1'b0;
            clear_o       <= 1'b0;
            instr_write_o <= 1'b0;
            data_write_o  <= 1'b0;

            if (write_fire) begin
                unique case (awaddr_i)
                    REG_CONTROL: begin
                        start_o <= wdata_i[0];
                        clear_o <= wdata_i[1];
                    end

                    REG_PROGRAM_LEN: begin
                        program_len_q <= wdata_i[7:0];
                    end

                    REG_INSTR_ADDR: begin
                        instr_addr_q <= wdata_i[7:0];
                    end

                    REG_INSTR_WORD: begin
                        instr_word_o  <= wdata_i;
                        instr_write_o <= 1'b1;
                    end

                    REG_DATA_ADDR: begin
                        data_addr_q <= wdata_i[7:0];
                    end

                    REG_DATA_BANK: begin
                        data_bank_q <= wdata_i[1:0];
                    end

                    REG_DATA_WORD: begin
                        data_word_o  <= wdata_i;
                        data_write_o <= 1'b1;
                    end

                    default: begin
                        // Unknown writes are acknowledged but ignored.
                    end
                endcase

                bvalid_o <= 1'b1;
            end else if (bvalid_o && bready_i) begin
                bvalid_o <= 1'b0;
            end

            if (read_fire) begin
                unique case (araddr_i)
                    REG_STATUS: begin
                        rdata_o <= {{(DATA_WIDTH-2){1'b0}}, engine_done_i, engine_busy_i};
                    end

                    REG_PROGRAM_LEN: begin
                        rdata_o <= {{(DATA_WIDTH-8){1'b0}}, program_len_q};
                    end

                    REG_INSTR_ADDR: begin
                        rdata_o <= {{(DATA_WIDTH-8){1'b0}}, instr_addr_q};
                    end

                    REG_DATA_ADDR: begin
                        rdata_o <= {{(DATA_WIDTH-8){1'b0}}, data_addr_q};
                    end

                    REG_DATA_BANK: begin
                        rdata_o <= {{(DATA_WIDTH-2){1'b0}}, data_bank_q};
                    end

                    REG_C00: begin
                        rdata_o <= c00_i;
                    end

                    REG_C01: begin
                        rdata_o <= c01_i;
                    end

                    REG_C10: begin
                        rdata_o <= c10_i;
                    end

                    REG_C11: begin
                        rdata_o <= c11_i;
                    end

                    default: begin
                        rdata_o <= '0;
                    end
                endcase

                rvalid_o <= 1'b1;
            end else if (rvalid_o && rready_i) begin
                rvalid_o <= 1'b0;
            end
        end
    end

endmodule