# RTL test vectors

Milestone 9 adds deterministic Python-generated test vectors for the 2x2
systolic tile RTL block.

The vector format is JSON and is generated from the same Python contract used in
Milestone 8:

```text
signed INT8 A tile sequence
signed INT8 B tile sequence
expected signed INT32 C tile
```

The generator lives in:

```text
opentsp/rtl_test_vectors.py
```

## Generate JSON vectors

```bash
PYTHONPATH=. python examples/generate_rtl_test_vectors.py
```

This writes:

```text
artifacts/rtl_vectors/systolic_tile_2x2_vectors.json
```

The `artifacts/` directory is intentionally not required for normal test runs.
The cocotb test can generate the same deterministic vectors in memory.

## Run Python vector tests

```bash
PYTHONPATH=. python -m pytest tests/test_rtl_test_vectors.py -q
```

## Run RTL vector test

Use WSL Ubuntu with Verilator and cocotb:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.systolic_tile_vectors
```

Expected result:

```text
TESTS=1 PASS=1 FAIL=0
```

Clean generated simulation files:

```bash
make -C tests_rtl -f Makefile.systolic_tile_vectors clean
```

## Why this matters

This milestone makes the hardware verification flow more realistic. Instead of
hand-writing every cocotb stimulus case, we generate vectors from the Python
reference and run those same vectors through the RTL block.

This is the start of a scalable verification loop:

```text
Python tiled matmul model -> deterministic vector JSON -> RTL simulation -> exact compare
```
