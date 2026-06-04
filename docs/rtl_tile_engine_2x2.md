# RTL tile engine 2x2

Milestone 14 connects two previously separate RTL blocks:

- `rtl/instruction_controller.sv`
- `rtl/systolic_tile_2x2.sv`

The new wrapper, `rtl/tile_engine_2x2.sv`, lets the instruction controller drive
when the systolic tile performs work. Specifically, the decoded `MAC_TILE`
instruction becomes the tile's `valid_i` signal.

## What this proves

Earlier milestones verified:

1. a 2x2 systolic tile can compute signed INT8 tile products, and
2. an instruction controller can step through abstract OpenTSP opcodes.

This milestone proves the first combined datapath:

```text
instruction program
        ↓
instruction_controller
        ↓
MAC_TILE decoded signal
        ↓
systolic_tile_2x2 valid_i
        ↓
INT32 C tile output
```

LOAD/STORE/ATTENTION are still control-only signals in this milestone. Real SRAM
movement is intentionally left for the next memory/data-routing milestone.

## Run the RTL test

From WSL Ubuntu with the project venv active:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.tile_engine_2x2
```

Expected result:

```text
TESTS=6 PASS=6 FAIL=0
```

Clean generated files:

```bash
make -C tests_rtl -f Makefile.tile_engine_2x2 clean
```

## Why this matters

This is the first OpenTSP RTL block where a controller instruction affects a
compute primitive. It is still tiny, but it is the bridge toward a full
instruction-driven tiled matmul engine.
