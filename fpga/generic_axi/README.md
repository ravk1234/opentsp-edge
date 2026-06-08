# Generic AXI-lite OpenTSP Top

This directory contains a generic AXI-lite-style top-level wrapper for the
OpenTSP 2x2 tile-engine path.

The wrapper is meant to be target-neutral. It can be adapted to:

- small FPGA boards
- SoC FPGA systems
- cloud FPGA shells
- Verilator top-level integration tests

## Top-level module

```text
opentsp_axi_top.sv
```

The module exposes a compact AXI-lite-style register interface and internally
instantiates:

```text
axi_lite_tile_engine_2x2
```

## Current limitations

This is a skeleton top. It assumes the AXI-lite tile engine module has already
been verified in local RTL simulation.

It does not yet include:

- vendor-specific constraints
- clock/reset conditioning
- CDC handling
- AXI interconnect integration
- synthesis scripts
- board pin mappings

## Test plan

The next milestone should add a Verilator test for this top-level wrapper:

```bash
make -C tests_rtl -f Makefile.opentsp_axi_top
```

That test should:

1. write instructions and data through the AXI-lite interface
2. start the engine
3. poll status
4. read C output registers
5. compare against the expected 4x4 matmul result
