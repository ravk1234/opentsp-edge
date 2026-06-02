# Roadmap

## Milestone 1: Local software MVP

Status: implemented.

- graph IR
- deterministic compiler
- cycle estimates
- SRAM bank allocation
- NumPy simulator
- toy voice decoder step

## Milestone 2: INT8 path

Add:

- quantize/dequantize utilities
- INT8 matmul reference kernel
- scale/zero-point metadata
- accuracy comparison vs float32

## Milestone 3: Tiled schedule

Lower high-level matmul into:

```text
load_tile_a
load_tile_b
matmul_tile
store_tile_c
```

Add bank conflict checks.

## Milestone 4: RTL verification

Add Verilator/cocotb tests for:

- mac_unit
- dot_product_unit
- tiny 2x2 systolic tile

## Milestone 5: Tiny real model

Target a small audio-token or text-token decoder block.

Possible first targets:

- TinyStories-style decoder
- small Mimi/SoundStream token predictor
- tiny Hindi phoneme/token predictor

## Milestone 6: FPGA/cloud path

Only after the local simulator and RTL blocks are stable:

- larger FPGA dev board
- AWS EC2 F2
- FireSim/Gemmini exploration
