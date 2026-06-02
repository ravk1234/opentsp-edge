from __future__ import annotations

from opentsp.accelerator_runtime import Int8RuntimeConfig, run_schedule_int8_tiled
from opentsp.attention_schedule import AttentionScheduleConfig
from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.models import build_tiny_voice_decoder_graph
from opentsp.tiled_matmul import TiledMatmulConfig
from opentsp.timeline import build_unified_timeline


def _build_timeline():
    graph, values = build_tiny_voice_decoder_graph()
    program = compile_graph(graph, AcceleratorConfig(clock_mhz=100, mac_lanes=32, elem_lanes=16))
    cfg = Int8RuntimeConfig(
        tiled_matmul=TiledMatmulConfig(tile_m=1, tile_n=8, tile_k=8, mac_lanes=32, load_lanes=16, store_lanes=16),
        attention=AttentionScheduleConfig(cache_tile=2, vector_lanes=16, mac_lanes=32, load_lanes=16, store_lanes=16),
    )
    result = run_schedule_int8_tiled(program, values, cfg)
    return program, result, build_unified_timeline(program, result.matmul_stats, result.attention_stats)


def test_unified_timeline_is_monotonic():
    _, _, timeline = _build_timeline()
    assert timeline.events
    previous_end = 0
    for event in timeline.events:
        assert event.start_cycle >= previous_end
        assert event.end_cycle >= event.start_cycle
        previous_end = event.end_cycle
    assert timeline.total_cycles == timeline.events[-1].end_cycle


def test_unified_timeline_contains_all_sources():
    _, _, timeline = _build_timeline()
    sources = {event.source for event in timeline.events}
    assert "tiled_matmul" in sources
    assert "kv_attention" in sources
    assert "baseline" in sources


def test_unified_timeline_has_expected_decoder_ops():
    _, _, timeline = _build_timeline()
    op_names = {event.op_name for event in timeline.events}
    assert "q_proj" in op_names
    assert "attn_decode" in op_names
    assert "vocab_logits" in op_names
    assert "next_token" in op_names


def test_unified_timeline_summary_matches_cycles():
    _, _, timeline = _build_timeline()
    summary_cycles = sum(cycles for _, _, _, cycles in timeline.op_summary())
    assert summary_cycles == timeline.total_cycles
