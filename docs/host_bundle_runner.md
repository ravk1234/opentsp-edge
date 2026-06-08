# Host-driven hardware bundle runner

Milestone 21 connects the exported hardware bundle to the host-style register interface added in Milestone 20.

Earlier milestones verified two separate flows:

```text
hardware bundle files -> cocotb loader -> tile_engine_mem_2x2
host register writes -> host_tile_engine_2x2
```

This milestone combines them:

```text
hardware bundle files
        ↓
Python host-driver transaction builder
        ↓
cocotb writes host registers
        ↓
host_tile_engine_2x2
        ↓
tile_engine_mem_2x2 + systolic_tile_2x2
        ↓
host reads C output registers
        ↓
full 4x4 C matrix is reconstructed
```

## Register transaction model

`opentsp/host_driver.py` converts a loaded tile program into a deterministic sequence of host register writes:

```text
CONTROL.clear
INSTR_ADDR / INSTR_WORD ...
DATA_BANK / DATA_ADDR / DATA_WORD ...
PROGRAM_LEN
CONTROL.start
```

This is intentionally still a tiny local simulation interface, not AXI/PCIe. The point is to prove that the same exported bundle can be loaded through a host-like control surface.

## Run the host transaction demo

```bash
PYTHONPATH=. python examples/build_host_transactions.py
```

Expected:

```text
Host transaction build check: PASSED
```

## Run Python tests

```bash
PYTHONPATH=. python -m pytest tests/test_host_driver.py -q
```

## Run RTL test

From WSL Ubuntu with the project virtual environment active:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.host_bundle_runner
```

Expected:

```text
TESTS=1 PASS=1 FAIL=0
```

Clean generated simulator files:

```bash
make -C tests_rtl -f Makefile.host_bundle_runner clean
```

Generated `artifacts/` outputs are not committed.
