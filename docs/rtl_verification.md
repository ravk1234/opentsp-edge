# RTL verification

Milestone 6 verifies the first real hardware primitive in OpenTSP: the signed
INT8 multiply-accumulate unit in `rtl/mac_unit.sv`.

The RTL test flow uses:

- **Verilator** to simulate SystemVerilog locally
- **cocotb** to write the testbench in Python
- **make** to run the simulation

No FPGA board, Shrike-lite board, or cloud FPGA instance is required for this
milestone.

## Setup on WSL Ubuntu

Install simulator/toolchain packages:

```bash
sudo apt update
sudo apt install -y verilator make g++ python3-dev
```

Activate your existing Python environment and install cocotb:

```bash
cd /mnt/d/ravi/code_project/groq_tts/opentsp_local_mvp
source .venv/bin/activate
pip install cocotb pytest
```

If you use a Windows venv for Python tests, still prefer WSL for RTL tests.
Verilator is much easier to install and run from Ubuntu.

## Run the RTL test

From the repo root:

```bash
make -C tests_rtl
```

Expected result:

```text
... TESTS=4 PASS=4 FAIL=0 ...
```

## What is tested

The cocotb testbench verifies that `mac_unit.sv`:

1. resets `acc_o` and `valid_o` correctly,
2. accumulates signed INT8 products correctly,
3. handles negative inputs using two's-complement encoding,
4. clears the accumulator with `clear_i`, and
5. matches a deterministic Python reference sequence.

This connects the Python-side INT8 tiled matmul simulator to the first verified
hardware primitive.
