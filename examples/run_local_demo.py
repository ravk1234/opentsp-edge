from __future__ import annotations

import numpy as np

from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.models import build_tiny_voice_decoder_graph, eager_reference
from opentsp.simulator import run_schedule


def main() -> None:
    graph, values = build_tiny_voice_decoder_graph(
        d_model=16,
        d_ff=32,
        vocab_size=64,
        cache_len=4,
        seed=7,
    )

    hw = AcceleratorConfig(
        name="opentsp-local-v0",
        num_banks=4,
        bank_size_bytes=64 * 1024,
        mac_lanes=64,
        elem_lanes=16,
        clock_mhz=100,
    )

    program = compile_graph(graph, hw)
    scheduled_values = run_schedule(program, values)
    reference = eager_reference(values, cache_len=4)

    print("\nCompiled deterministic schedule")
    print("-" * 92)
    for op in program.schedule:
        print(
            f"{op.index:02d} {op.name:<18} {op.kind:<17} "
            f"start={op.start_cycle:<6} end={op.end_cycle:<6} cycles={op.cycles:<6}"
        )
    print("-" * 92)
    print(f"Total estimated cycles: {program.total_cycles}")
    print(f"Estimated single-token latency at {hw.clock_mhz} MHz: {program.total_cycles / hw.clock_mhz:.3f} microseconds")

    print("\nSRAM bank allocation")
    print("-" * 92)
    for name, alloc in program.allocations.items():
        print(f"{name:<18} bank={alloc.bank:<2} offset={alloc.offset:<6} bytes={alloc.size_bytes:<6}")

    np.testing.assert_allclose(scheduled_values["logits"], reference["logits"], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(scheduled_values["next_token"], reference["next_token"])

    print("\nDeterministic check: PASSED")
    print(f"Next token: {int(scheduled_values['next_token'][0])}")
    print("\nThis is still a software simulator. Next step: replace high-level ops with tiled INT8 kernels.")


if __name__ == "__main__":
    main()
