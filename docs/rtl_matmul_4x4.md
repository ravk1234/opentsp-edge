# RTL 4x4 matmul through the 2x2 tile engine

Milestone 16 runs a small full matrix multiplication through the memory-backed
2x2 RTL tile engine.

The workload is:

```text
A: signed INT8 4x4
B: signed INT8 4x4
C: signed INT32 4x4
```

The matrix multiplication is decomposed into four independent 2x2 output-tile
programs. Each output-tile program runs:

```text
LOAD_A
LOAD_B
MAC_TILE
LOAD_A
LOAD_B
MAC_TILE
STORE_C
```

The two `MAC_TILE` instructions correspond to the two K tiles needed for a
4-wide inner dimension.

## Generate the program JSON

```bash
PYTHONPATH=. python examples/generate_rtl_matmul_program.py
```

This writes:

```text
artifacts/rtl_programs/matmul_4x4_program.json
```

The JSON is useful for inspection and future controller/program loading work,
but the RTL cocotb test generates the same deterministic program in memory.

## Run Python tests

```bash
PYTHONPATH=. python -m pytest tests/test_rtl_matmul_program.py -q
```

## Run the RTL test

Use WSL Ubuntu with the project virtual environment active:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.tile_engine_matmul_4x4
```

Expected result:

```text
TESTS=1 PASS=1 FAIL=0
```

Clean generated simulator files:

```bash
make -C tests_rtl -f Makefile.tile_engine_matmul_4x4 clean
```

## Why this matters

Earlier milestones verified the scalar MAC, the 2x2 tile, the instruction
controller, and the memory-backed tile engine. This milestone verifies the first
small tiled matmul end-to-end in RTL simulation:

```text
Python 4x4 matmul reference
        ↓
2x2 output-tile programs
        ↓
tile_engine_mem_2x2.sv
        ↓
reconstructed RTL 4x4 C matrix
```
