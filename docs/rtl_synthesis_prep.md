# RTL Synthesis Preparation

Milestone 29 adds a board-agnostic synthesis preparation layer for OpenTSP Edge.

## What it adds

- a generic AXI RTL filelist
- filelist validation utilities
- a source check script
- tests for missing/duplicate RTL files
- synthesis-prep documentation

## Why it exists

The project already has local RTL simulation coverage. The next step toward
real FPGA/cloud-FPGA deployment is to make the RTL source boundary explicit.

A synthesis or vendor project needs a clean list of sources and a known top
module. This milestone provides that layer without committing to a specific
board or cloud platform yet.

## Commands

Run the source checker:

```bash
PYTHONPATH=. python scripts/check_rtl_sources.py
```

Run tests:

```bash
PYTHONPATH=. python -m pytest tests/test_rtl_filelist.py -q
```

Expected result:

```text
RTL source check: PASSED
5 passed
```

## Current top

```text
opentsp_axi_top_sim
```

## Current filelist

```text
fpga/generic_axi/filelist.f
```
