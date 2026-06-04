from __future__ import annotations

import numpy as np

from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.quant import quantize_symmetric_int8
from opentsp.rtl_schedule_vectors import (
    collect_schedule_vectors_from_program,
    generate_systolic_vectors_from_int8_matmul,
    systolic_vectors_from_tiled_matmul_schedule,
)
from opentsp.tiled_matmul import TiledMatmulConfig, int8_tiled_matmul
from opentsp.tiny_gpt import build_tiny_gpt_step


def _vector_valid_window(vector_name: str) -> tuple[slice, slice]:
    # Names are formatted as op_m0_m1_n0_n1. Op names can contain underscores,
    # so parse from the right.
    parts = vector_name.rsplit("_", 4)
    m0 = int(parts[1][1:])
    m1 = int(parts[2])
    n0 = int(parts[3][1:])
    n1 = int(parts[4])
    return slice(m0, m1), slice(n0, n1)


def test_vectors_from_one_tiled_matmul_match_accumulator_subtiles() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 6)).astype(np.float32)
    b = rng.normal(size=(6, 4)).astype(np.float32)
    cfg = TiledMatmulConfig(tile_m=2, tile_n=4, tile_k=4, mac_lanes=16)

    vectors, acc = generate_systolic_vectors_from_int8_matmul("demo_matmul", a, b, cfg)

    assert len(vectors) == 4
    for vector in vectors:
        m_slice, n_slice = _vector_valid_window(vector.name)
        expected = np.asarray(vector.expected_c, dtype=np.int32)
        np.testing.assert_array_equal(expected[: m_slice.stop - m_slice.start, : n_slice.stop - n_slice.start], acc[m_slice, n_slice])


def test_schedule_vector_extraction_handles_odd_edges_with_padding() -> None:
    rng = np.random.default_rng(11)
    a = rng.normal(size=(3, 5)).astype(np.float32)
    b = rng.normal(size=(5, 3)).astype(np.float32)
    cfg = TiledMatmulConfig(tile_m=2, tile_n=3, tile_k=3, mac_lanes=8)

    qa = quantize_symmetric_int8(a)
    qb = quantize_symmetric_int8(b)
    tiled = int8_tiled_matmul(qa, qb, cfg)
    vectors = systolic_vectors_from_tiled_matmul_schedule("odd_matmul", qa, qb, tiled.events)

    # Output shape 3x3 decomposes into four 2x2 RTL windows with edge padding.
    assert len(vectors) == 4
    for vector in vectors:
        m_slice, n_slice = _vector_valid_window(vector.name)
        expected = np.asarray(vector.expected_c, dtype=np.int32)
        valid_rows = m_slice.stop - m_slice.start
        valid_cols = n_slice.stop - n_slice.start
        np.testing.assert_array_equal(expected[:valid_rows, :valid_cols], tiled.acc_int32[m_slice, n_slice])


def test_collect_schedule_vectors_from_tiny_gpt_program() -> None:
    step = build_tiny_gpt_step([4, 12, 9, 3, 21], seed=123)
    program = compile_graph(step.graph, AcceleratorConfig())

    extraction = collect_schedule_vectors_from_program(program, step.values)

    assert extraction.op_names == ("q_proj", "k_proj", "v_proj", "o_proj", "ffn_up", "ffn_down", "vocab_logits")
    assert len(extraction.groups) == 7
    assert extraction.total_vectors == 88
    assert extraction.total_k_tiles > extraction.total_vectors
    assert int(np.asarray(extraction.values["next_token"]).reshape(-1)[0]) == 38


def test_schedule_vectors_are_deterministic() -> None:
    step = build_tiny_gpt_step([4, 12, 9, 3, 21], seed=123)
    program = compile_graph(step.graph, AcceleratorConfig())

    first = collect_schedule_vectors_from_program(program, step.values)
    second = collect_schedule_vectors_from_program(program, step.values)

    first_vectors = [v for group in first.groups for v in group.vectors]
    second_vectors = [v for group in second.groups for v in group.vectors]
    assert first_vectors == second_vectors
