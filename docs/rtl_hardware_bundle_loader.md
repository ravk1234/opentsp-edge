# RTL hardware bundle loader

Milestone 19 verifies that the hardware-friendly export bundle can be consumed by the RTL simulation flow.

Milestone 18 created this bundle:

```text
artifacts/hardware_bundle/matmul_4x4/
  manifest.json
  expected_c.json
  c00/
    instructions.hex
    a_memory.hex
    b_memory.hex
    expected_c.json
  c01/
  c10/
  c11/
```

This milestone adds a Python loader and a cocotb RTL test that reads those files, programs `tile_engine_mem_2x2`, runs each 2x2 output-tile program, and reconstructs the full 4x4 matrix output.

## Flow

```text
hardware bundle files
        ↓
Python bundle loader
        ↓
cocotb writes instruction/data memories
        ↓
tile_engine_mem_2x2 executes LOAD_A / LOAD_B / MAC_TILE / STORE_C
        ↓
stored C tile is compared with expected_c.json
        ↓
full 4x4 C matrix is reconstructed
```

## Run the Python loader demo

```bash
PYTHONPATH=. python examples/load_hardware_bundle.py
```

Expected:

```text
Hardware bundle loader check: PASSED
```

## Run Python tests

```bash
PYTHONPATH=. python -m pytest tests/test_hardware_loader.py -q
```

## Run RTL test

From WSL Ubuntu with the project virtual environment active:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.hardware_bundle_loader
```

Expected:

```text
TESTS=1 PASS=1 FAIL=0
```

Clean generated simulator files:

```bash
make -C tests_rtl -f Makefile.hardware_bundle_loader clean
```

## Notes

The cocotb test generates the bundle if it is not already present. Generated `artifacts/` files are intentionally not committed.

This is not an FPGA host interface yet. It is the local simulation version of the same concept: exported instructions/data are treated as the external hardware input contract.
