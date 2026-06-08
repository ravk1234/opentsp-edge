# OpenTSP Edge generic Xilinx XDC constraint skeleton.
#
# This file is a template only. It is not tied to a specific FPGA board.
# Replace PACKAGE_PIN and IOSTANDARD values before using on real hardware.

## Clock
# set_property PACKAGE_PIN <CLK_PIN> [get_ports clk_i]
# set_property IOSTANDARD LVCMOS33 [get_ports clk_i]
create_clock -period 10.000 -name opentsp_clk [get_ports clk_i]

## Reset
# set_property PACKAGE_PIN <RESET_PIN> [get_ports rst_ni]
# set_property IOSTANDARD LVCMOS33 [get_ports rst_ni]

## AXI-lite-style interface
#
# The current top exposes a simple AXI-lite-style simulation interface.
# For real hardware, most targets will connect this through a platform wrapper,
# soft CPU bus, or vendor shell rather than direct pins.
#
# Add platform-specific constraints here after selecting a board.

## Early target frequency
#
# Start with 50-100 MHz for the first board bring-up. The current goal is
# functional validation of the host-to-tile-engine path.
