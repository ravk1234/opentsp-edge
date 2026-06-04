# RTL instruction-memory controller

Milestone 13 adds a tiny RTL instruction-memory and controller FSM in
`rtl/instruction_controller.sv`.

This is the first hardware-side bridge from OpenTSP's emitted instruction stream
toward a future accelerator controller.

## What it does

The controller has a simple host/programming port:

```text
host_we_i
host_addr_i
host_opcode_i
host_cycles_i
```

It stores a small instruction program internally. When `start_i` is asserted, it
walks through the program in order. Each instruction is held valid for its
programmed cycle count.

The decoded outputs are:

```text
load_a_o
load_b_o
mac_tile_o
store_c_o
attention_o
baseline_o
```

These correspond to the current abstract OpenTSP instruction classes:

```text
LOAD_A
LOAD_B
MAC_TILE
STORE_C
ATTENTION
BASELINE
```

This controller does not yet drive SRAM banks or the systolic tile. It only
proves the instruction-memory and deterministic program-counter behavior.

## Run the RTL test

Use WSL Ubuntu with Verilator and cocotb installed:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.instruction_controller
```

Expected result:

```text
TESTS=5 PASS=5 FAIL=0
```

Clean generated simulator files:

```bash
make -C tests_rtl -f Makefile.instruction_controller clean
```

## Why this matters

Earlier milestones generated controller-style instructions in Python. This
milestone adds the first RTL block that can store and step through those classes
of instructions with deterministic timing.

Next steps are:

1. connect emitted JSON instructions to this controller testbench,
2. add data-path signals for the 2x2 systolic tile, and
3. simulate a small instruction-driven tiled matmul end to end.
