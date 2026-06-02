from __future__ import annotations

import numpy as np

from opentsp.accelerator_runtime import Int8RuntimeConfig, run_schedule_int8_tiled
from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.models import build_tiny_voice_decoder_graph
from opentsp.simulator import run_schedule
from opentsp.attention_schedule import AttentionScheduleConfig
from opentsp.tiled_matmul import TiledMatmulConfig


def main() -> None:
    graph, values = build_tiny_voice_decoder_graph(
        d_model=16,
        d_ff=32,
        vocab_size=64,
        cache_len=4,
        seed=7,
    )
    hw = AcceleratorConfig(mac_lanes=64, elem_lanes=16, clock_mhz=100)
    program = compile_graph(graph, hw)

    fp32 = run_schedule(program, values)

    int8_cfg = Int8RuntimeConfig(
        tiled_matmul=TiledMatmulConfig(
            tile_m=1,
            tile_n=8,
            tile_k=8,
            mac_lanes=32,
            load_lanes=16,
            store_lanes=16,
        ),
        attention=AttentionScheduleConfig(
            cache_tile=2,
            vector_lanes=16,
            mac_lanes=32,
            load_lanes=16,
            store_lanes=16,
        ),
    )
    int8 = run_schedule_int8_tiled(program, values, int8_cfg)

    fp32_token = int(fp32["next_token"][0])
    int8_token = int(int8.values["next_token"][0])
    logits_error = float(np.max(np.abs(fp32["logits"] - int8.values["logits"])))
    latency_us = int8.total_tiled_cycles / hw.clock_mhz

    print("INT8 decoder integration demo")
    print("-" * 80)
    print(f"Graph: {graph.name}")
    attention_latency_us = int8.total_attention_cycles / hw.clock_mhz

    print(f"Accelerated matmul ops: {len(int8.matmul_stats)}")
    print(f"Scheduled attention ops: {len(int8.attention_stats)}")
    print(f"Total tiled matmul micro-ops: {int8.total_tiled_events}")
    print(f"Total attention micro-ops: {int8.total_attention_events}")
    print(f"Total tiled matmul cycles: {int8.total_tiled_cycles}")
    print(f"Total attention cycles: {int8.total_attention_cycles}")
    print(f"Estimated tiled-matmul latency at {hw.clock_mhz} MHz: {latency_us:.3f} microseconds")
    print(f"Estimated attention latency at {hw.clock_mhz} MHz: {attention_latency_us:.3f} microseconds")
    print(f"FP32 next token: {fp32_token}")
    print(f"INT8 next token: {int8_token}")
    print(f"Token match: {'PASSED' if fp32_token == int8_token else 'DIFFERENT'}")
    print(f"Max abs logits error vs FP32 path: {logits_error:.6f}")

    print("\nAccelerated ops")
    print("-" * 80)
    for stat in int8.matmul_stats:
        print(
            f"{stat.op_name:<14} "
            f"{stat.input_shape} x {stat.weight_shape} -> {stat.output_shape}  "
            f"events={stat.event_count:<4} cycles={stat.cycles:<5} "
            f"scales=({stat.input_scale:.6g}, {stat.weight_scale:.6g})"
        )

    print("\nScheduled attention ops")
    print("-" * 80)
    for stat in int8.attention_stats:
        print(
            f"{stat.op_name:<14} "
            f"q={stat.q_shape} k={stat.k_cache_shape} v={stat.v_cache_shape} -> {stat.output_shape}  "
            f"events={stat.event_count:<4} cycles={stat.cycles:<5}"
        )

    print("\nFirst 12 tiled micro-ops from first accelerated matmul")
    print("-" * 80)
    first = int8.matmul_stats[0]
    for event in first.events[:12]:
        print(
            f"{event.index:03d} {event.kind:<14} "
            f"M{event.m_range} N{event.n_range} K{event.k_range} "
            f"start={event.start_cycle:<5} end={event.end_cycle:<5} "
            f"cycles={event.cycles:<4} bytes={event.bytes_moved:<4} macs={event.macs:<4}"
        )

    print("\nAttention micro-ops")
    print("-" * 80)
    if int8.attention_stats:
        for event in int8.attention_stats[0].events:
            print(
                f"{event.index:03d} {event.kind:<20} "
                f"T{event.token_range} D{event.dim_range} "
                f"start={event.start_cycle:<5} end={event.end_cycle:<5} "
                f"cycles={event.cycles:<4} bytes={event.bytes_moved:<4} macs={event.macs:<4}"
            )

    print("\nDeterministic INT8 decoder + KV-cache attention schedule check: PASSED")


if __name__ == "__main__":
    main()
