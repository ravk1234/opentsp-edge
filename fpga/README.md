# OpenTSP Edge FPGA Skeleton

This directory contains the first deployment skeleton for taking OpenTSP Edge
from local RTL simulation toward FPGA or cloud-FPGA execution.

OpenTSP currently uses a small AXI-lite-style control path:

```text
host program
    ↓
memory-mapped register writes
    ↓
AXI-lite-style register wrapper
    ↓
OpenTSP tile engine
    ↓
2x2 systolic tile
    ↓
memory-mapped C output reads
```

## Directories

```text
fpga/
  generic_axi/
    Generic AXI-lite top-level wrapper.

  aws_f2/
    AWS F2-facing notes and placeholder custom-logic top.
```

## Current status

This is not a full FPGA build system yet. It is a clean starting point for
future FPGA work.

The current verified path is still local:

```text
Python hardware bundle
    ↓
host transaction generation
    ↓
cocotb / Verilator simulation
    ↓
AXI-lite tile engine RTL
```

## Near-term FPGA tasks

1. Add a Verilator top-level test for `fpga/generic_axi/opentsp_axi_top.sv`.
2. Decide the first real target:
   - low-cost FPGA board
   - AWS F2
   - another cloud FPGA provider
3. Map the AXI-lite register interface to the selected platform.
4. Add board/cloud-specific build scripts.
5. Connect the generated C/C++ host runner to the platform-specific register
   access API.

## Design philosophy

The first hardware target should be small and easy to understand. The goal is
not immediate performance. The goal is a reproducible end-to-end path:

```text
model schedule
→ instruction/data bundle
→ host writes
→ RTL tile engine
→ correct output
```
