# FPGA Deployment Skeleton

This document describes the first FPGA/cloud-FPGA deployment boundary for
OpenTSP Edge.

The current project already has:

- deterministic instruction generation
- hardware export bundles
- host transaction generation
- a C/C++ host-runner shape
- AXI-lite-style host registers
- a 2x2 tile engine with RTL simulation tests

This milestone adds a repository skeleton for moving those pieces toward an
FPGA or cloud-FPGA target.

## Deployment boundary

The intended hardware boundary is:

```text
host software
    ↓ memory-mapped register writes
AXI-lite control/status interface
    ↓ decoded writes
OpenTSP tile engine
    ↓
2x2 systolic tile
    ↓
C output registers
    ↓ memory-mapped register reads
host software
```

## Register interface

The first deployment interface is intentionally simple and mirrors the existing
host register flow:

| Register | Purpose |
| --- | --- |
| `CONTROL` | start / clear |
| `STATUS` | busy / done |
| `PROGRAM_LEN` | number of instructions |
| `INSTR_ADDR` | selected instruction address |
| `INSTR_WORD` | instruction word write |
| `DATA_ADDR` | selected data memory address |
| `DATA_BANK` | A/B memory bank select |
| `DATA_WORD` | data memory word write |
| `C00..C11` | output tile registers |

## Files added by this milestone

```text
fpga/
  README.md
  generic_axi/
    README.md
    opentsp_axi_top.sv
  aws_f2/
    README.md
    cl_opentsp_top.sv
    host_runner_notes.md
```

## What this milestone is

This is a deployment skeleton. It defines the top-level shape, naming, and
integration notes for FPGA/cloud-FPGA work.

It is useful for:

- documenting the intended hardware boundary
- showing how the AXI-lite tile engine will be wrapped
- making the repo look like a real accelerator project
- preparing for board/cloud-specific wrappers

## What this milestone is not yet

This milestone does not yet provide:

- a complete FPGA build project
- AWS F2 AFI build scripts
- PCIe/DMA integration
- timing constraints
- synthesis reports
- real hardware benchmarks

Those come after the AXI-lite top-level wrapper has a Verilator integration test
and after a specific FPGA target is selected.
