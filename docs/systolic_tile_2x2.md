# 2x2 systolic tile RTL

Milestone 7 adds a tiny 2x2 signed INT8 systolic-style tile in
`rtl/systolic_tile_2x2.sv`.

The tile accepts one 2x2 A tile and one 2x2 B tile per valid cycle and
accumulates a 2x2 INT32 output tile:

```text
C += A x B
```

This is still a small educational block, not a production accelerator. Its
purpose is to bridge the Python tiled-matmul simulator with a real RTL matrix
primitive that can be verified locally.

## Run the RTL test

Use WSL Ubuntu with Verilator and cocotb installed. From the repo root:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.systolic_tile_2x2
```

Expected result:

```text
TESTS=5 PASS=5 FAIL=0
```

Clean generated simulation files:

```bash
make -C tests_rtl -f Makefile.systolic_tile_2x2 clean
```

## What is tested

The cocotb testbench verifies:

1. reset behavior,
2. one signed 2x2 matrix product,
3. accumulation across multiple K tiles,
4. clear/reset between independent products, and
5. deterministic random tile sequences against a Python reference.

## Why this matters

Earlier milestones verified the deterministic compiler, tiled INT8 matmul
schedule, KV-cache attention schedule, unified timeline, and scalar MAC RTL.
This milestone is the first RTL block that directly represents a small matrix
operation.
