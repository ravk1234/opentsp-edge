# FPGA and Cloud-FPGA Build Plan

Milestone 28 defines the first practical build-planning layer for moving
OpenTSP Edge from Verilator simulation toward FPGA and cloud-FPGA deployment.

This milestone does not claim that OpenTSP builds on a physical FPGA yet.
Instead, it documents the interface boundary, constraints strategy, and bring-up
sequence needed for a future board or AWS F2-style flow.

## Current verified path

The project currently has a simulated hardware path:

```text
hardware bundle
  -> host transactions
  -> AXI-lite-style writes
  -> FPGA-facing top wrapper
  -> AXI-lite tile engine
  -> tile engine memory
  -> 2x2 systolic tile
  -> C output registers
```

The next step is to package that path into a buildable FPGA project structure.

## Deployment target classes

OpenTSP should support three target classes over time:

1. **Generic local FPGA board**
   - simple clock/reset
   - AXI-lite-style control
   - small internal memories
   - host program writes instructions and tile data

2. **Cloud FPGA / AWS F2-style target**
   - cloud shell wrapper
   - host runtime writes registers over the platform interface
   - OpenTSP appears as a custom logic block

3. **Simulation-only bring-up**
   - Verilator/cocotb remains the first correctness gate
   - every FPGA-facing wrapper should have a local test before hardware bring-up

## Minimum FPGA wrapper responsibilities

A real FPGA wrapper should provide:

- a stable clock input
- a reset input
- AXI-lite or AXI-lite-like register access
- register writes for instructions and A/B memory
- start/clear control
- status readback
- C output readback

## Register map

The current host register map is:

| Address | Name | Direction | Purpose |
|---:|---|---|---|
| `0x00` | CONTROL | write | bit 0 = start, bit 1 = clear |
| `0x04` | STATUS | read | bit 0 = busy, bit 1 = done |
| `0x08` | PROGRAM_LEN | write/read | number of instructions |
| `0x10` | INSTR_ADDR | write/read | selected instruction slot |
| `0x14` | INSTR_WORD | write | instruction word |
| `0x18` | DATA_ADDR | write/read | selected data memory slot |
| `0x1C` | DATA_BANK | write/read | 0 = A memory, 1 = B memory |
| `0x20` | DATA_WORD | write | tile data word |
| `0x30` | C00 | read | output C[0][0] |
| `0x34` | C01 | read | output C[0][1] |
| `0x38` | C10 | read | output C[1][0] |
| `0x3C` | C11 | read | output C[1][1] |

## Bring-up order

Recommended bring-up order:

1. Run Verilator/cocotb tests.
2. Build generic AXI wrapper in FPGA tooling.
3. Verify clock/reset constraints.
4. Expose the register map through the board or platform host interface.
5. Port the generated C host runner to the platform runtime.
6. Write a known 4x4 matmul bundle.
7. Start execution and read C output.
8. Compare against `expected_c.json`.

## Success criteria

Milestone 28 is complete when the repo contains:

- generic constraint templates
- cloud-FPGA/AWS F2 planning notes
- a clear top-level build boundary
- a documented bring-up checklist
