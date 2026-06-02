# RTL/Python contract tests

Milestone 8 connects the Python tiled-matmul simulator assumptions to the
`rtl/systolic_tile_2x2.sv` hardware behavior.

The contract is intentionally small and explicit:

```text
A tile: signed INT8 [2, 2]
B tile: signed INT8 [2, 2]
C accumulator: signed INT32 [2, 2]

C += A @ B
```

The Python reference lives in:

```text
opentsp/rtl_reference.py
```

It models the same signed INT8 to INT32 accumulation behavior as the RTL tile.
It also provides a helper for computing a `2xK` by `Kx2` matmul through repeated
2x2 K-tiles, including zero padding for odd K.

## Run Python contract tests

```bash
PYTHONPATH=. pytest tests/test_rtl_reference.py -q
```

## Run RTL/Python contract tests

Use WSL Ubuntu with Verilator and cocotb installed:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.systolic_tile_contract
```

Expected result:

```text
TESTS=4 PASS=4 FAIL=0
```

Clean generated files:

```bash
make -C tests_rtl -f Makefile.systolic_tile_contract clean
```

## Why this matters

Earlier milestones separately verified:

1. Python INT8 tiled matmul simulation, and
2. RTL 2x2 systolic tile behavior.

This milestone bridges them by verifying that both sides follow the same core
math contract. That makes future larger RTL tiles easier to validate against the
existing Python simulator.
