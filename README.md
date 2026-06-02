# OpenTSP Local MVP

A tiny local prototype for a **deterministic, SRAM-first inference accelerator** aimed at edge voice / Indic TTS research.

This is not a Groq clone. It is a small open-source-style research starter that lets you test the main ideas locally:

- static graph compilation
- deterministic operation scheduling
- simple SRAM bank allocation
- NumPy schedule simulator
- tiny transformer-style voice token decoder step
- starter RTL MAC block for future Verilator testing

## What runs locally

```text
Tiny voice-token decoder graph
        ↓
Deterministic compiler
        ↓
Static schedule with cycle estimates
        ↓
SRAM bank assignment
        ↓
NumPy simulator
        ↓
Correctness check vs eager NumPy reference
```

## Setup

```bash
cd opentsp_local_mvp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python examples/run_local_demo.py
pytest -q
```

On Windows, use WSL2 Ubuntu and run the same commands.

## Output example

```text
Compiled deterministic schedule
00 q_proj          matmul            start=0      end=16     cycles=16
01 k_proj          matmul            start=16     end=32     cycles=16
...
Total estimated cycles: 713
Deterministic check: PASSED
Next token: 22
```

## Folder layout

```text
opentsp/
  ir.py           graph IR
  hardware.py     accelerator config and SRAM banks
  compiler.py     deterministic scheduler + bank allocator
  simulator.py    schedule-following NumPy simulator
  models.py       tiny voice-token decoder graph + eager reference
examples/
  run_local_demo.py
tests/
  test_local_mvp.py
rtl/
  mac_unit.sv     small signed 8-bit MAC block for later RTL testing
  README.md
```

## Next milestones

1. Add proper tiled matmul schedule.
2. Add bank conflict detection.
3. Add quantized INT8 path.
4. Add Verilator/cocotb test for `rtl/mac_unit.sv`.
5. Replace high-level attention op with scheduled primitive ops.
6. Port a tiny audio-token decoder block.
