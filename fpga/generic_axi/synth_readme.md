# Generic AXI Synthesis Preparation

Milestone 29 prepares the verified FPGA-facing AXI simulation path for future
synthesis and board/cloud-FPGA integration.

This is still not a board-specific FPGA build. The goal is to define the RTL
source boundary clearly enough that the design can be handed to a synthesis
tool or adapted into a target-specific project.

## Top module

For the current simulation-ready top:

```text
opentsp_axi_top_sim
```

The top module wraps:

```text
axi_lite_tile_engine_2x2
  -> axi_lite_host_regs
  -> host_tile_engine_2x2
  -> tile_engine_mem_2x2
  -> systolic_tile_2x2
```

## RTL filelist

The board-agnostic filelist is:

```text
fpga/generic_axi/filelist.f
```

It contains all SystemVerilog sources required for the current AXI-lite tile
engine path.

## First synthesis target

Start conservatively:

```text
50 MHz to 100 MHz
```

The first goal is to prove that the design can elaborate and synthesize, not to
optimize frequency or area.

## Expected bring-up flow

1. Run existing Verilator/cocotb tests.
2. Check that every file in `filelist.f` exists.
3. Feed the filelist into synthesis tooling.
4. Select or create a board/cloud wrapper.
5. Map the AXI-lite-style register interface to the platform bus.
6. Reuse the generated C/C++ host runner for register writes and output checks.

## Non-goals

This milestone does not:

- run a vendor synthesis build
- generate a bitstream
- build an AWS F2 AFI
- claim timing closure
- deploy to a physical board
