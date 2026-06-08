# AWS F2 Host Runner Notes

OpenTSP already generates C/C++ host-runner artifacts for the 4x4 matmul demo.
Those artifacts use a simple callback-style register interface:

```c
write_reg(addr, value);
value = read_reg(addr);
```

For AWS F2, these calls will eventually need to map to AWS FPGA runtime or
memory-mapped BAR access.

## Current generated flow

```text
hardware bundle
    ↓
host transactions
    ↓
C/C++ host runner
    ↓
write_reg / read_reg abstraction
```

## Future AWS F2 mapping

The generated host runner should eventually be adapted like this:

```text
write_reg(addr, value)
    → platform_mmio_write(base + addr, value)

read_reg(addr)
    → platform_mmio_read(base + addr)
```

The exact API depends on the AWS F2 shell/runtime chosen for the integration.

## Register map

The OpenTSP demo uses this compact register map:

| Address | Name | Direction | Purpose |
| --- | --- | --- | --- |
| `0x00` | CONTROL | write | bit 0 start, bit 1 clear |
| `0x04` | STATUS | read | bit 0 busy, bit 1 done |
| `0x08` | PROGRAM_LEN | write/read | instruction count |
| `0x10` | INSTR_ADDR | write/read | selected instruction address |
| `0x14` | INSTR_WORD | write | packed instruction word |
| `0x18` | DATA_ADDR | write/read | selected data address |
| `0x1C` | DATA_BANK | write/read | A/B memory select |
| `0x20` | DATA_WORD | write | packed tile data word |
| `0x30` | C00 | read | output C tile lane |
| `0x34` | C01 | read | output C tile lane |
| `0x38` | C10 | read | output C tile lane |
| `0x3C` | C11 | read | output C tile lane |

## First AWS-facing success criterion

The first useful AWS F2 milestone should be:

```text
host writes exported 4x4 matmul bundle
host starts OpenTSP tile engine
host polls done
host reads C output registers
host verifies expected C matrix
```

No performance claims should be made until the design is actually built and
timed on hardware.
