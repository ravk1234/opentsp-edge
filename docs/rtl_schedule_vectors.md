# Schedule-derived RTL vectors

Milestone 11 connects the Python tiled-matmul scheduler to the RTL 2x2 systolic
tile testbench.

Earlier milestones generated standalone RTL vectors. This milestone extracts
vectors from real `mac_tile` events emitted by the Python tiled matmul simulator.
That means the RTL tile is now checked against the same schedule structure used
by the OpenTSP runtime.

## What is verified

For every accelerated matmul in the Tiny GPT adapter:

1. The matmul is quantized to signed INT8.
2. The Python tiled-matmul simulator emits `load_a_tile`, `load_b_tile`,
   `mac_tile`, and `store_c_tile` events.
3. The `mac_tile` regions are decomposed into 2x2 systolic-tile vectors.
4. Odd or partial edge regions are zero padded.
5. Verilator + cocotb runs those vectors against `rtl/systolic_tile_2x2.sv`.
6. The RTL INT32 outputs must match the Python reference exactly.

## Python demo

```bash
PYTHONPATH=. python examples/generate_schedule_rtl_vectors.py
```

Expected result:

```text
Schedule-derived RTL vector generation check: PASSED
```

## Python tests

```bash
PYTHONPATH=. python -m pytest tests/test_rtl_schedule_vectors.py -q
```

## RTL test

Run from WSL/Linux:

```bash
make -C tests_rtl -f Makefile.systolic_tile_schedule_vectors
```

Expected result:

```text
TESTS=1 PASS=1 FAIL=0
```

## Cleanup

```bash
make -C tests_rtl -f Makefile.systolic_tile_schedule_vectors clean
```

Generated JSON vectors under `artifacts/` should not be committed.
