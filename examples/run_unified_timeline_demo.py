from __future__ import annotations

from opentsp.accelerator_runtime import Int8RuntimeConfig, run_schedule_int8_tiled
from opentsp.attention_schedule import AttentionScheduleConfig
from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.models import build_tiny_voice_decoder_graph
from opentsp.tiled_matmul import TiledMatmulConfig
from opentsp.timeline import build_unified_timeline


def main() -> None:
    graph, values = build_tiny_voice_decoder_graph()
    hw = AcceleratorConfig(clock_mhz=100, mac_lanes=32, elem_lanes=16)
    program = compile_graph(graph, hw)

    cfg = Int8RuntimeConfig(
        tiled_matmul=TiledMatmulConfig(tile_m=1, tile_n=8, tile_k=8, mac_lanes=32, load_lanes=16, store_lanes=16),
        attention=AttentionScheduleConfig(cache_tile=2, vector_lanes=16, mac_lanes=32, load_lanes=16, store_lanes=16),
    )
    result = run_schedule_int8_tiled(program, values, cfg)
    timeline = build_unified_timeline(program, result.matmul_stats, result.attention_stats)

    latency_us = timeline.total_cycles / hw.clock_mhz

    print("Unified deterministic decoder timeline demo")
    print("-" * 80)
    print(f"Graph: {graph.name}")
    print(f"Total timeline events: {timeline.event_count}")
    print(f"Total timeline cycles: {timeline.total_cycles}")
    print(f"Estimated full token-step latency at {hw.clock_mhz} MHz: {latency_us:.3f} microseconds")

    print("First 32 unified timeline events")
    print("-" * 80)
    for event in timeline.events[:32]:
        print(
            f"{event.index:03d} {event.op_name:<14} {event.source:<14} {event.micro_op:<20} "
            f"start={event.start_cycle:<5} end={event.end_cycle:<5} "
            f"cycles={event.cycles:<4} bytes={event.bytes_moved:<4} macs={event.macs:<4} {event.detail}"
        )

    print("Per-op timeline summary")
    print("-" * 80)
    for op_name, source, event_count, cycles in timeline.op_summary():
        print(f"{op_name:<14} {source:<14} events={event_count:<4} cycles={cycles:<5}")

    print("Unified deterministic timeline check: PASSED")


if __name__ == "__main__":
    main()
