# Architecture Notes

## Goal

Build a small deterministic inference path for edge voice / Indic TTS experiments.

The first local MVP is intentionally simple:

```text
Model graph -> compiler -> static schedule -> simulator
```

## Deterministic execution

The compiler emits a fixed operation order with fixed estimated cycle ranges:

```text
op_0: start=0, end=16
op_1: start=16, end=32
...
```

No runtime graph decisions happen inside the simulator. The schedule is precomputed.

## SRAM-first idea

Every tensor receives a bank and offset before execution. This simulates the idea that tensors live in explicit scratchpad/SRAM, not in opaque cache.

Current limitation: allocation is greedy and does not reuse memory after tensor lifetimes end.

## Current high-level ops

The simulator currently supports:

- matmul
- add
- silu
- rmsnorm
- append_cache
- attention_decode
- softmax
- argmax

The most important next step is to lower these high-level ops into tiled primitive operations.
