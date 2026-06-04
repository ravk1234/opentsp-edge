# RTL tile engine with data routing

Milestone 15 adds `rtl/tile_engine_mem_2x2.sv`, a small instruction-driven tile
engine with register-file style A/B tile memories.

This extends the previous `tile_engine_2x2` milestone. The previous engine could
let `MAC_TILE` drive the 2x2 systolic tile, but A and B tile values were still
provided directly as top-level inputs. This milestone adds a minimal data path:

```text
host writes A/B tiles into tiny memories
        ↓
LOAD_A / LOAD_B choose tiles by program counter
        ↓
MAC_TILE drives systolic_tile_2x2
        ↓
STORE_C captures the accumulated C tile
```

## What is intentionally simple

This is not a full SRAM subsystem yet. The A and B memories are small RTL arrays
inside the wrapper. The current program counter is used as the memory index for
LOAD instructions. That keeps the design deterministic and easy to verify before
adding richer address fields to the instruction format.

## Run the RTL test

From WSL Ubuntu with the project virtual environment active:

```bash
source .venv/bin/activate
make -C tests_rtl -f Makefile.tile_engine_mem_2x2
```

Expected result:

```text
TESTS=5 PASS=5 FAIL=0
```

Clean generated simulator files:

```bash
make -C tests_rtl -f Makefile.tile_engine_mem_2x2 clean
```

## What the tests verify

The cocotb testbench verifies:

1. reset state clears loaded tiles and stored C,
2. `LOAD_A` and `LOAD_B` fetch memory tiles by program counter,
3. `MAC_TILE` computes using the loaded A/B tiles,
4. multiple K tiles accumulate before `STORE_C`, and
5. a deterministic random memory-backed program matches a Python reference.

## Why this matters

This milestone is the first small end-to-end RTL data path:

```text
instruction controller + data memories + systolic tile + stored output
```

It is still tiny, but it is the bridge toward running a full tiled matmul program
inside RTL instead of manually feeding tile inputs.
