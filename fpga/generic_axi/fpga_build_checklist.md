# Generic FPGA Build Checklist

This checklist tracks the steps needed to move the generic AXI-lite simulation
top toward a board-specific FPGA build.

## 1. Select target

Choose one:

- low-cost local FPGA board
- vendor evaluation board
- cloud-FPGA shell
- simulation-only wrapper

Record:

- FPGA family
- toolchain
- clock source
- reset source
- host communication path

## 2. Confirm top-level module

The current simulation top is:

```text
opentsp_axi_top_sim
```

For real hardware, create a board-specific top that instantiates:

```text
axi_lite_tile_engine_2x2
```

or wraps `opentsp_axi_top_sim` behind a platform bus adapter.

## 3. Map registers

Expose the OpenTSP register map to the host:

- CONTROL
- STATUS
- PROGRAM_LEN
- INSTR_ADDR
- INSTR_WORD
- DATA_ADDR
- DATA_BANK
- DATA_WORD
- C00/C01/C10/C11

## 4. Load a known hardware bundle

Use the generated hardware bundle and C/C++ runner path:

```text
instructions.hex
a_memory.hex
b_memory.hex
expected_c.json
```

## 5. Validate

Success means:

```text
host writes bundle -> hardware runs -> host reads C -> expected C matches
```
