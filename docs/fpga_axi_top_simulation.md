# FPGA AXI Top Simulation

Milestone 27 adds a Verilator/cocotb simulation target for the FPGA-facing AXI top path.

The purpose is to prove that the deployment-facing top-level wrapper can drive the already-verified OpenTSP tile engine through AXI-lite-style register writes and reads.

## Tested path

```text
opentsp_axi_top_sim
  -> axi_lite_tile_engine_2x2
  -> axi_lite_host_regs
  -> host_tile_engine_2x2
  -> tile_engine_mem_2x2
  -> systolic_tile_2x2
```

The test writes one real exported `c00` tile program from the 4x4 matmul hardware bundle through the AXI-lite-style interface:

```text
CONTROL clear
instruction memory writes
A/B tile memory writes
PROGRAM_LEN write
CONTROL start
STATUS polling
C output register reads
```

Expected output tile:

```text
[[26, -16],
 [-11, 35]]
```

## Run

```bash
make -C tests_rtl -f Makefile.fpga_axi_top
```

Expected result:

```text
TESTS=1 PASS=1 FAIL=0 SKIP=0
```

## Notes

This is still a simulation-only milestone. It does not yet build a vendor FPGA bitstream or AWS F2 AFI. It proves that the FPGA-facing wrapper boundary can run an exported tile program through the RTL engine under cocotb/Verilator.
