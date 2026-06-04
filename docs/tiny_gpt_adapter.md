# Tiny GPT adapter

Milestone 10A adds a real tiny GPT-style model path without requiring GPU,
cloud, Hugging Face downloads, or FPGA hardware.

The adapter is intentionally small and deterministic. It includes:

```text
token embedding
position embedding
Q/K/V projections
KV-cache attention
output projection
MLP up/down
vocab logits
argmax next token
```

The embedding lookup and KV-cache prefill happen in Python. The resulting
single-token decode graph is then executed through the same OpenTSP path used by
the toy decoder:

```text
FP32 reference path
INT8 tiled matmul runtime
KV-cache attention scheduler
unified deterministic timeline
```

## Run the demo

```bash
PYTHONPATH=. python examples/run_tiny_gpt_demo.py
```

Expected result:

```text
Tiny GPT OpenTSP adapter check: PASSED
```

## Run tests

```bash
PYTHONPATH=. python -m pytest tests/test_tiny_gpt_adapter.py -q
```

## Why this matters

This milestone moves the project from a hand-built toy decoder step toward a
real model-shaped workflow. It still runs locally on CPU, but now the inputs come
from prompt token IDs, token embeddings, positional embeddings, and a prefilled
KV cache.

This is the right foundation for the next benchmark milestone:

```text
real tiny GPT-style model
  -> FP32 CPU path
  -> INT8 OpenTSP path
  -> unified timeline cycles
  -> estimated accelerator latency/tokens per second
```
