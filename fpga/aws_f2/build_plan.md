# AWS F2 Build Plan

This directory is a planning skeleton for a future AWS F2 or cloud-FPGA
deployment path.

It is not a complete AWS F2 build flow yet.

## Goal

The goal is to wrap the OpenTSP AXI-lite tile engine behind the cloud FPGA
shell interface so that a host program can:

```text
write instructions
write A/B tile memory
start execution
poll status
read C outputs
compare expected result
```

## Current OpenTSP hardware boundary

The current simulation-verified boundary is:

```text
opentsp_axi_top_sim
  -> axi_lite_tile_engine_2x2
  -> axi_lite_host_regs
  -> host_tile_engine_2x2
  -> tile_engine_mem_2x2
  -> systolic_tile_2x2
```

For AWS F2, the simulation top will need to be adapted into an AWS custom-logic
top wrapper.

## Expected future AWS F2 tasks

1. Map the OpenTSP register interface to the AWS shell-facing control path.
2. Add clock/reset adaptation.
3. Add a platform-specific host runtime.
4. Build and simulate with the AWS FPGA development flow.
5. Package and register an FPGA image.
6. Run the generated host runner against the deployed image.

## Host runner mapping

The generated C/C++ host runner already emits logical register operations:

```c
write_reg(addr, value);
read_reg(addr);
```

On AWS F2, these callbacks should be mapped to the platform register access API.

## First cloud-FPGA demo target

The first realistic AWS/cloud-FPGA demo should remain intentionally small:

- 4x4 INT8 matmul
- 2x2 tile engine
- fixed register map
- single clock domain
- host-driven program loading
- C output readback

The point of the first demo is correctness and flow validation, not speed.

## Non-goals for this milestone

This milestone does not:

- generate an AFI
- provide AWS shell integration
- provide PCIe DMA
- run a large model
- claim real accelerator performance

Those come after local FPGA-facing wrapper simulation is stable.
