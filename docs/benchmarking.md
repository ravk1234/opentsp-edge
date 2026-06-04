# Tiny GPT benchmarking

Milestone 10B adds a benchmark for the Tiny GPT OpenTSP adapter.

The benchmark reports two different classes of numbers.

## 1. Real CPU wall-clock timings

These are measured with Python's `perf_counter` on the local machine. They show
how long the FP32 reference path and OpenTSP Python simulator path take on CPU.

These are real software timings, but they are not accelerator timings. The
OpenTSP simulator is intentionally detailed and can be slower than optimized
NumPy/PyTorch code.

## 2. Deterministic accelerator estimates

The unified timeline reports cycle counts for one token step. Estimated latency
is computed as:

```text
latency_us = cycles / clock_mhz
```

Estimated throughput is computed as:

```text
tokens_per_second = clock_mhz * 1_000_000 / cycles
```

These are cycle-model estimates. They become real measurements only after a
controller, memory system, and compute tile are deployed to real hardware.

## Run

```bash
PYTHONPATH=. python examples/benchmark_tiny_gpt.py
```

Expected output includes:

- FP32 vs INT8/OpenTSP next-token comparison
- max absolute logits error
- CPU wall-clock timings
- timeline events and cycles
- estimated latency at 50 MHz, 100 MHz, and 200 MHz
- estimated tokens/sec

## Test

```bash
PYTHONPATH=. python -m pytest tests/test_tiny_gpt_benchmark.py -q
```
