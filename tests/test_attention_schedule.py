from __future__ import annotations

import numpy as np
import pytest

from opentsp.attention_schedule import AttentionScheduleConfig, schedule_attention_decode
from opentsp.accelerator_runtime import Int8RuntimeConfig, run_schedule_int8_tiled
from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.models import build_tiny_voice_decoder_graph
from opentsp.simulator import run_schedule


def _reference_attention(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
    d = q.shape[-1]
    scores = (q @ k_cache.T) / np.sqrt(float(d))
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    return (probs @ v_cache).astype(np.float32)


def test_attention_schedule_matches_fp32_reference() -> None:
    rng = np.random.default_rng(123)
    q = rng.normal(0.0, 0.5, size=(1, 16)).astype(np.float32)
    k_cache = rng.normal(0.0, 0.3, size=(5, 16)).astype(np.float32)
    v_cache = rng.normal(0.0, 0.3, size=(5, 16)).astype(np.float32)

    result = schedule_attention_decode(q, k_cache, v_cache, AttentionScheduleConfig(cache_tile=2))
    reference = _reference_attention(q, k_cache, v_cache)

    np.testing.assert_allclose(result.context, reference, rtol=1e-6, atol=1e-6)
    assert result.event_count > 0
    assert result.total_cycles > 0
    assert result.events[0].kind == "load_q_vector"
    assert result.events[-1].kind == "store_context"


def test_attention_schedule_is_deterministic() -> None:
    rng = np.random.default_rng(456)
    q = rng.normal(0.0, 0.5, size=(1, 12)).astype(np.float32)
    k_cache = rng.normal(0.0, 0.3, size=(7, 12)).astype(np.float32)
    v_cache = rng.normal(0.0, 0.3, size=(7, 12)).astype(np.float32)
    cfg = AttentionScheduleConfig(cache_tile=3, vector_lanes=8, mac_lanes=8)

    first = schedule_attention_decode(q, k_cache, v_cache, cfg)
    second = schedule_attention_decode(q, k_cache, v_cache, cfg)

    assert first.total_cycles == second.total_cycles
    assert [e.kind for e in first.events] == [e.kind for e in second.events]
    assert [(e.start_cycle, e.end_cycle) for e in first.events] == [
        (e.start_cycle, e.end_cycle) for e in second.events
    ]
    np.testing.assert_allclose(first.context, second.context, rtol=0, atol=0)


def test_attention_schedule_rejects_bad_shapes() -> None:
    q = np.zeros((2, 16), dtype=np.float32)
    k_cache = np.zeros((4, 16), dtype=np.float32)
    v_cache = np.zeros((4, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="q must have shape"):
        schedule_attention_decode(q, k_cache, v_cache)


def test_decoder_runtime_schedules_attention_decode() -> None:
    graph, values = build_tiny_voice_decoder_graph(cache_len=4, seed=7)
    program = compile_graph(graph, AcceleratorConfig())
    fp32 = run_schedule(program, values)
    cfg = Int8RuntimeConfig(attention=AttentionScheduleConfig(cache_tile=2))

    result = run_schedule_int8_tiled(program, values, cfg)

    assert result.attention_op_names == ("attn_decode",)
    assert result.total_attention_events > 0
    assert result.total_attention_cycles > 0
    assert result.attention_stats[0].events[0].kind == "load_q_vector"
    np.testing.assert_allclose(result.values["attn_ctx"], fp32["attn_ctx"], rtol=0, atol=0.02)
