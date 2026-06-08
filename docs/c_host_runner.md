# C-style host runner export

Milestone 22 exports the host transaction list into portable C-style source and header files.

Earlier milestones proved this flow in Python/cocotb:

```text
hardware bundle -> host writes -> RTL host register interface -> 4x4 C output
```

This milestone adds a C-style host runner that mirrors the same flow:

```text
OPENTSP_MATMUL_4X4_WRITES[]
        ↓
write_reg(addr, value)
        ↓
poll OPENTSP_REG_STATUS until DONE
        ↓
read OPENTSP_REG_C00/C01/C10/C11
        ↓
reconstruct 4x4 C output
```

The generated C does not assume AXI, PCIe, AWS F2, or a specific board. It only assumes the user provides two functions:

```c
void write_reg(uint32_t addr, uint32_t value);
uint32_t read_reg(uint32_t addr);
```

That makes the generated runner useful for later:

- Verilator host wrappers
- memory-mapped FPGA drivers
- AWS/cloud-FPGA host applications
- small board demos

## Generate the C host runner

```bash
PYTHONPATH=. python examples/export_c_host_runner.py
```

Output:

```text
artifacts/host_runner/matmul_4x4/opentsp_matmul_4x4_runner.c
artifacts/host_runner/matmul_4x4/opentsp_matmul_4x4_runner.h
```

## Run tests

```bash
PYTHONPATH=. python -m pytest tests/test_c_host_runner.py -q
```

Generated `artifacts/` outputs are not committed.
