# AXI-lite Tile Engine Prototype

Milestone 25 connects the AXI-lite-style host register wrapper to the existing
OpenTSP 2x2 tile engine.

The flow is now:

```text
AXI-lite-style writes
  -> CONTROL / PROGRAM_LEN / INSTR / DATA registers
  -> tile_engine_mem_2x2
  -> systolic_tile_2x2
  -> stored C output
  -> AXI-lite-style reads
```

This is not a complete production AXI peripheral yet. It is a small,
simulation-friendly bridge that proves the intended host-control shape.

## Registers

The wrapper uses the same register map as `axi_lite_host_regs.sv`:

| Address | Name | Purpose |
|---:|---|---|
| `0x00` | CONTROL | bit 0 = start, bit 1 = clear |
| `0x04` | STATUS | bit 0 = busy, bit 1 = done |
| `0x08` | PROGRAM_LEN | number of instructions to execute |
| `0x10` | INSTR_ADDR | selected instruction memory address |
| `0x14` | INSTR_WORD | instruction word write |
| `0x18` | DATA_ADDR | selected data memory address |
| `0x1C` | DATA_BANK | 0 = A memory, 1 = B memory |
| `0x20` | DATA_WORD | packed signed INT8 2x2 tile data |
| `0x30..0x3C` | C output | stored INT32 C tile outputs |

## Test

Run from WSL:

```bash
make -C tests_rtl -f Makefile.axi_lite_tile_engine
```

Expected:

```text
TESTS=1 PASS=1 FAIL=0 SKIP=0
```

The test loads one exported 2x2 tile program through AXI-lite-style writes,
starts execution, polls `STATUS.done`, and reads back the C tile through AXI-lite
reads.
