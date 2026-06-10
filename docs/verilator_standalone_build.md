# Standalone Verilator Top Build

Milestone 30 adds a standalone Verilator build generator for the OpenTSP
FPGA-facing AXI top.

This is different from the cocotb tests. Cocotb remains the main RTL
verification path, while this milestone creates a small C++ executable shape
that is closer to a future host/runtime flow.

## What it generates

Running:

```bash
PYTHONPATH=. python examples/build_verilator_top.py
```

creates:

```text
artifacts/verilator_top/
  opentsp_axi_top_sim_main.cpp
  build.sh
  verilator_top_build.json
```

The generated C++ harness:

- instantiates `Vopentsp_axi_top_sim`
- toggles `clk_i`
- applies reset
- runs a short smoke test
- optionally emits a VCD trace
- prints a PASS line

## Optional build command

If Verilator is installed, run:

```bash
bash artifacts/verilator_top/build.sh
artifacts/verilator_top/obj_dir/opentsp_axi_top_sim
```

Expected output:

```text
OpenTSP standalone Verilator top smoke test: PASS
```

## Why this matters

The project already has:

```text
hardware bundle -> host transactions -> AXI-lite top -> RTL simulation
```

This milestone adds the first standalone executable shape that can evolve toward:

- Verilator C++ host simulation
- board host runners
- cloud-FPGA host runtimes
- CI smoke builds

## Non-goals

This milestone does not:

- replace cocotb verification
- build a bitstream
- connect to PCIe/AXI shell APIs
- deploy to AWS F2
