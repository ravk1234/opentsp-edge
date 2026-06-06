# Memory-bank conflict checker

Milestone 17 adds a small software checker for SRAM-bank access conflicts.

The checker is intentionally simple. It models a banked scratchpad with a fixed
number of read and write ports per bank. Given scheduled memory accesses, it can
report:

- read-port conflicts
- write-port conflicts
- read/write conflicts on the same bank in the same cycle
- invalid accesses
- out-of-bounds accesses

## Why this matters

Earlier milestones generated deterministic timelines, instruction streams, and
RTL tile-engine tests. Those flows assumed memory movement was valid. This
checker makes that assumption explicit by checking whether each scheduled
LOAD/STORE can be served by the configured bank model.

## Instruction mapping

The instruction checker maps:

- `LOAD_A` to a read access
- `LOAD_B` to a read access
- `STORE_C` to a write access

Other instruction kinds such as `MAC_TILE`, `ATTENTION`, and `BASELINE` are not
modeled as SRAM accesses in this milestone.

## Demo

```bash
PYTHONPATH=. python examples/check_memory_banks.py
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_bank_conflict_checker.py -q
```
