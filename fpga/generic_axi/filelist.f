# OpenTSP Edge generic AXI synthesis/simulation RTL filelist.
#
# Paths are relative to the repository root.
#
# This file is intended as the first board-agnostic source list for synthesis
# preparation. It mirrors the verified FPGA-facing AXI simulation path.

rtl/instruction_controller.sv
rtl/systolic_tile_2x2.sv
rtl/tile_engine_mem_2x2.sv
rtl/host_tile_engine_2x2.sv
rtl/axi_lite_host_regs.sv
rtl/axi_lite_tile_engine_2x2.sv
fpga/generic_axi/opentsp_axi_top_sim.sv
