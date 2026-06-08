# AWS F2 Skeleton

This directory is a placeholder for an eventual AWS F2 integration.

AWS F2 integration is not active yet. The current purpose is to document the
expected shape of the custom logic boundary.

## Intended boundary

OpenTSP should eventually be exposed to the AWS shell through a small
memory-mapped register interface:

```text
AWS host application
    ↓
AWS shell / PCIe / AXI infrastructure
    ↓
AXI-lite-style control/status registers
    ↓
OpenTSP tile engine
```

The first OpenTSP accelerator payload is intentionally small:

```text
host writes instruction/data registers
host starts engine
engine computes 2x2 / 4x4 tiled INT8 matmul
host polls done
host reads C output registers
```

## Files

```text
cl_opentsp_top.sv
  Placeholder AWS-F2-style custom-logic top.

host_runner_notes.md
  Notes for adapting the generated C/C++ host runner to AWS F2.
```

## What still needs to be added

- real AWS shell interface mapping
- AXI-lite bridge wiring
- AWS FPGA build configuration
- simulation scripts compatible with the AWS FPGA flow
- AFI build/register steps
- host application that uses AWS FPGA runtime APIs

## Recommended development order

Before attempting AWS F2, finish:

1. Generic AXI top-level Verilator test.
2. Host-runner integration against a simulated register model.
3. A clean memory map document.
4. A small FPGA-board or pure Verilator top-level run.

Only then should the project attempt AWS F2 packaging.
