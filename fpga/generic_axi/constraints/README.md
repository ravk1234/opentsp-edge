# Generic AXI Constraint Templates

This directory contains placeholder timing/constraint templates for future FPGA
bring-up of the OpenTSP generic AXI wrapper.

These files are intentionally conservative and board-agnostic. They are not
ready to use directly on a specific board without editing pin names, clock
ports, and timing targets.

## Files

- `opentsp_generic_axi.xdc`
  - Xilinx-style constraint skeleton.

- `opentsp_generic_axi.sdc`
  - Generic SDC-style timing skeleton.

## Expected top-level signals

The simulation wrapper currently uses:

```text
clk_i
rst_ni
AXI-lite-style write/read channels
```

For a real board, `clk_i` and `rst_ni` must be mapped to actual board pins or
shell-provided signals.

## First timing target

Start with a modest timing target:

```text
50 MHz to 100 MHz
```

The early design is focused on correctness and host integration, not peak
frequency.

## Bring-up checklist

1. Select board/platform.
2. Edit clock pin/period.
3. Edit reset polarity and pin.
4. Connect host bus or shell wrapper.
5. Run timing analysis.
