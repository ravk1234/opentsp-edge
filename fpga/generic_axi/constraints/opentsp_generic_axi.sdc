# OpenTSP Edge generic SDC constraint skeleton.
#
# This is a board-agnostic template. Edit it for the target FPGA toolchain and
# board before use.

# Example 100 MHz clock.
# Change clk_i to match the real top-level clock port if needed.
create_clock -name opentsp_clk -period 10.000 [get_ports clk_i]

# Reset is asynchronous in the current RTL style.
# Platform-specific flows may need false-path constraints for reset.
set_false_path -from [get_ports rst_ni]

# Placeholder input/output delay constraints.
# Replace these with real board/platform timing constraints.
# set_input_delay  2.0 -clock opentsp_clk [all_inputs]
# set_output_delay 2.0 -clock opentsp_clk [all_outputs]

# The first bring-up target is correctness, not maximum Fmax.
# Tighten this after the top-level wrapper is mapped to a real board/platform.
