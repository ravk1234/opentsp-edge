# AXI-lite-style host register wrapper

Milestone 24 introduces a minimal AXI-lite-style register wrapper for the OpenTSP tile-engine host interface.

The goal is not to provide a board-specific AXI integration yet. Instead, this module defines and verifies the protocol shape needed by FPGA boards or cloud-FPGA shells later:

```text
AXI-lite-style writes
    -> CONTROL / PROGRAM_LEN / INSTR / DATA registers
    -> decoded host sideband signals
    -> future tile engine wrapper

AXI-lite-style reads
    -> STATUS / C output registers
    -> host-visible result path
```

## Register map

| Address | Register | Purpose |
|---:|---|---|
| `0x00` | `CONTROL` | bit 0 = start pulse, bit 1 = clear pulse |
| `0x04` | `STATUS` | bit 0 = busy, bit 1 = done |
| `0x08` | `PROGRAM_LEN` | number of instructions to execute |
| `0x10` | `INSTR_ADDR` | instruction memory index |
| `0x14` | `INSTR_WORD` | instruction word write port |
| `0x18` | `DATA_ADDR` | data memory index |
| `0x1C` | `DATA_BANK` | data bank select: A/B/etc. |
| `0x20` | `DATA_WORD` | data word write port |
| `0x30` | `C00` | C tile output word 0 |
| `0x34` | `C01` | C tile output word 1 |
| `0x38` | `C10` | C tile output word 2 |
| `0x3C` | `C11` | C tile output word 3 |

## What is verified

The cocotb tests verify:

- reset state is clear
- register writes decode to one-cycle sideband pulses
- instruction and data write ports work
- status reads expose `busy` and `done`
- C output registers are readable
- unknown register accesses are safe

## Run

```bash
make -C tests_rtl -f Makefile.axi_lite_host_regs
```

Expected result:

```text
TESTS=4 PASS=4 FAIL=0
```

## Next step

The next milestone can connect this register wrapper directly to `host_tile_engine_2x2.sv`, or create a board/cloud-specific wrapper around the same register map.
