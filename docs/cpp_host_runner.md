# C++ host executable runner

Milestone 23 exports a standalone C++ host simulation executable around the generated C-style host runner.

Earlier milestones proved this flow:

```text
hardware bundle -> generated C runner -> host register writes
```

This milestone adds a small C++ host-side register model so the generated runner can be compiled and executed as a normal program:

```text
opentsp_matmul_4x4_runner.c/.h
        ↓
C++ host register model
        ↓
write_reg(addr, value)
poll STATUS.done
read C output registers
        ↓
compare against expected 4x4 C matrix
```

This is not an AXI/PCIe runtime yet. It is a host executable scaffold that mirrors the host API needed later for Verilator wrappers, FPGA boards, or cloud FPGA hosts.

## Generate the C++ host runner

```bash
PYTHONPATH=. python examples/export_cpp_host_runner.py
```

Output:

```text
artifacts/cpp_host_runner/matmul_4x4/opentsp_matmul_4x4_runner.c
artifacts/cpp_host_runner/matmul_4x4/opentsp_matmul_4x4_runner.h
artifacts/cpp_host_runner/matmul_4x4/opentsp_matmul_4x4_host_sim.cpp
artifacts/cpp_host_runner/matmul_4x4/Makefile
```

## Compile and run

From WSL or any environment with `g++` and `make`:

```bash
make -C artifacts/cpp_host_runner/matmul_4x4
./artifacts/cpp_host_runner/matmul_4x4/opentsp_matmul_4x4_host_sim
```

Expected:

```text
OpenTSP C++ host simulation: PASS
```

## Run tests

```bash
PYTHONPATH=. python -m pytest tests/test_cpp_host_runner.py -q
```

Generated `artifacts/` outputs are not committed.
