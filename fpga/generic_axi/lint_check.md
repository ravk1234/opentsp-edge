# RTL Filelist and Lint Check Notes

This document describes the lightweight source checks added before a real
synthesis flow.

## What is checked

The current Python checker validates:

- the RTL filelist exists
- all listed source files exist
- no duplicate source entries exist
- the expected top-level wrapper source is present
- source extensions are `.sv`, `.v`, or `.vh`

## Why this matters

The project already has cocotb/Verilator tests. The next deployment step is to
make the RTL source boundary explicit, so it can be passed to FPGA tools,
cloud-FPGA shells, or external lint/synthesis tooling.

## Future checks

Later milestones can add:

- Verilator lint-only mode
- Yosys/Surelog parsing
- vendor synthesis dry-run
- dependency ordering checks
- top-module elaboration checks
