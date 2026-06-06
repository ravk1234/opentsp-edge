# Hardware bundle export

Milestone 18 exports the tiny 4x4 RTL matmul program into hardware-friendly files.

Earlier milestones generated Python objects and JSON payloads for RTL tests. This milestone writes a bundle that looks more like something a future testbench, FPGA loader, or RTL memory initializer can consume.

## Bundle contents

Running the demo writes:

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
    ...
  c10/
    ...
  c11/
    ...
```

Each `cXX` directory contains one 2x2 output-tile program.

## Encodings

### Instruction word

Instructions are exported as 32-bit words:

```text
bits  7:0   opcode
bits 23:8   cycles
bits 31:24  reserved
```

Example:

```text
00000101
```

means opcode `1` with `1` cycle.

### Tile word

A signed INT8 2x2 tile is packed into one 32-bit word:

```text
bits  7:0   tile[0][0]
bits 15:8   tile[0][1]
bits 23:16  tile[1][0]
bits 31:24  tile[1][1]
```

Values are stored as two's-complement signed INT8 bytes.

## Run demo

```bash
PYTHONPATH=. python examples/export_hardware_bundle.py
```

Expected result:

```text
Hardware bundle export check: PASSED
```

## Run tests

```bash
PYTHONPATH=. python -m pytest tests/test_hardware_export.py -q
```

## Why this matters

This milestone creates the handoff between Python program generation and future RTL/FPGA loading:

```text
Python matmul program
  -> instructions.hex
  -> A/B memory hex
  -> expected C outputs
  -> future RTL loader / FPGA host
```

Generated `artifacts/` outputs should not be committed.
