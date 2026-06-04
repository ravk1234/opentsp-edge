# Accelerator instruction format

Milestone 12 introduces a small controller-style instruction format for OpenTSP.
The goal is to bridge the Python schedule and a future RTL controller FSM.

Earlier milestones produced a unified timeline with events such as:

```text
load_a_tile -> load_b_tile -> mac_tile -> store_c_tile
```

This milestone converts those events into explicit instructions:

```text
LOAD_A
LOAD_B
MAC_TILE
STORE_C
ATTENTION
BASELINE
```

## What this is

This is an abstract instruction stream. It is not a final hardware ISA yet.
It gives every timeline event a deterministic opcode, cycle range, bank, offset,
bytes moved, MAC count, and tile range metadata.

That makes it suitable for the next stage:

```text
Tiny GPT model
  -> OpenTSP tiled/attention timeline
  -> accelerator instruction stream
  -> future RTL instruction-memory/controller FSM
```

## Logical banks

The default logical bank assignment is:

| Bank | Purpose |
|---:|---|
| 0 | A/input tiles |
| 1 | B/weight tiles |
| 2 | C/output accumulator tiles |
| 3 | KV-cache attention data |
| 4 | coarse baseline operations |

These are not physical SRAM claims yet. They are deterministic metadata for the
instruction stream and future memory-bank checks.

## Instruction fields

Each instruction includes:

- `index`
- `opcode`
- `op_name`
- `micro_op`
- `start_cycle`
- `cycles`
- `end_cycle`
- `source`
- `bank`
- `offset`
- `bytes_moved`
- `macs`
- optional tile ranges: `m_range`, `n_range`, `k_range`
- optional attention ranges: `token_range`, `dim_range`

## Running the demo

```bash
PYTHONPATH=. python examples/emit_tiny_gpt_instructions.py
```

Expected result:

```text
Instruction emission check: PASSED
```

The demo writes:

```text
artifacts/instructions/tiny_gpt_instructions.json
```

Do not commit generated `artifacts/` outputs.

## Next step

The next milestone should add a small RTL instruction-memory/controller FSM that
can read a tiny instruction sequence and drive the 2x2 systolic tile.
