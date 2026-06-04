from __future__ import annotations

from pathlib import Path

from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.rtl_schedule_vectors import collect_schedule_vectors_from_program
from opentsp.rtl_test_vectors import save_systolic_tile_vectors
from opentsp.tiny_gpt import build_tiny_gpt_step


def main() -> None:
    step = build_tiny_gpt_step([4, 12, 9, 3, 21], seed=123)
    program = compile_graph(step.graph, AcceleratorConfig())
    extraction = collect_schedule_vectors_from_program(program, step.values)
    vectors = [vector for group in extraction.groups for vector in group.vectors]

    out = save_systolic_tile_vectors(Path("artifacts/rtl_vectors/tiny_gpt_schedule_vectors.json"), vectors)

    print("Scheduled matmul RTL vector generation demo")
    print("-" * 80)
    print(f"Graph: {step.graph.name}")
    print(f"Output: {out}")
    print(f"Accelerated matmul ops: {len(extraction.groups)}")
    print(f"Vector count: {extraction.total_vectors}")
    print(f"Total RTL K-tile cycles: {extraction.total_k_tiles}")
    print("\nPer-op vector summary")
    print("-" * 80)
    for group in extraction.groups:
        print(
            f"{group.op_name:<14} shape={group.input_shape} x {group.weight_shape} "
            f"vectors={group.vector_count:<4} k_tiles={group.k_tile_count:<4} "
            f"schedule_events={group.schedule_event_count:<4} cycles={group.schedule_cycles:<4}"
        )

    if len(extraction.groups) != 7:
        raise SystemExit("Expected 7 accelerated matmul ops")
    if extraction.total_vectors <= 0 or extraction.total_k_tiles <= 0:
        raise SystemExit("Expected non-empty schedule-derived vectors")
    print("\nSchedule-derived RTL vector generation check: PASSED")


if __name__ == "__main__":
    main()
