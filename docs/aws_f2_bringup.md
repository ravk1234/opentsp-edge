# AWS F2 Bring-Up Plan

## Goal

Run the OpenTSP AXI tile engine on an AWS F2 FPGA instance with a minimal host runtime.

Success criteria:

1. Host writes instruction memory.
2. Host writes A/B matrix data.
3. Host starts execution.
4. FPGA computes one 4x4 INT8 matmul tile.
5. Host reads C outputs.
6. Demo recorded.

---

## Current Status

Completed:

- Tile engine RTL
- Host register interface
- AXI-lite wrapper
- FPGA-facing top
- Cocotb verification
- Standalone Verilator build

Remaining:

- AWS F2 integration
- Host runtime
- FPGA image build
- Cloud deployment

---

## Minimum Demo

Matrix size: 4x4

Inputs:
- A matrix
- B matrix

Output:
- C matrix

Validation:
- FPGA result matches Python golden model