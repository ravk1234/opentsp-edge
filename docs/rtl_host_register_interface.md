# RTL host register interface

Milestone 20 adds `rtl/host_tile_engine_2x2.sv`, a small memory-mapped style host wrapper around the memory-backed 2x2 tile engine.

The previous milestone verified this local simulation flow:

```text
hardware bundle files
        ↓
cocotb loader
        ↓
tile_engine_mem_2x2
        ↓
4x4 C matrix reconstruction
```

This milestone adds a more hardware-like control surface:

```text
host writes registers
        ↓
instruction/data memories are loaded
        ↓
host writes START
        ↓
engine runs
        ↓
host polls STATUS.done
        ↓
host reads C output registers
```

## Register map

All registers are 32-bit.

| Address | Name | Direction | Description |
|---:|---|---|---|
| `0x00` | `CONTROL` | write | bit 0 = start pulse, bit 1 = clear tile/done |
| `0x04` | `STATUS` | read | bit 0 = busy, bit 1 = done sticky, bit 2 = stored valid, bit 3 = instruction valid |
| `0x08` | `PROGRAM_LEN` | read/write | number of instructions to execute |
| `0x0c` | `PC` | read | current controller program counter |
| `0x10` | `INSTR_ADDR` | read/write | instruction write address |
| `0x14` | `INSTR_WORD` | write | packed instruction: bits `[7:0]` opcode, bits `[23:8]` cycles |
| `0x18` | `DATA_ADDR` | read/write | A/B tile memory write address |
| `0x1c` | `DATA_BANK` | read/write | `0` = A memory, `1` = B memory |
| `0x20` | `DATA_WORD` | write | packed signed INT8 2x2 tile: `[7:0]=00`, `[15:8]=01`, `[23:16]=10`, `[31:24]=11` |
| `0x30` | `C00` | read | stored C tile element 00 |
| `0x34` | `C01` | read | stored C tile element 01 |
| `0x38` | `C10` | read | stored C tile element 10 |
| `0x3c` | `C11` | read | stored C tile element 11 |

## Run the RTL test

From WSL Ubuntu with the project virtual environment active:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.host_register_interface
```

Expected result:

```text
TESTS=4 PASS=4 FAIL=0
```

Clean generated simulator files:

```bash
make -C tests_rtl -f Makefile.host_register_interface clean
```

## Why this matters

This is the first host-style interface in OpenTSP. It is still a local simulation interface, not AXI/PCIe yet, but the model is closer to what an FPGA/cloud-FPGA host wrapper will need:

```text
write instructions → write data → start → poll done → read output
```
