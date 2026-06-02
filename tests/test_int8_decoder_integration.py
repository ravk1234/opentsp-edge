from __future__ import annotations

import numpy as np

from opentsp.accelerator_runtime import Int8RuntimeConfig, run_schedule_int8_tiled
from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.models import build_tiny_voice_decoder_graph
from opentsp.simulator import run_schedule
from opentsp.tiled_matmul import TiledMatmulConfig


EXPECTED_MATMUL_OPS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "ffn_up",
    "ffn_down",
    "vocab_logits",
)


def _program_and_values(seed: int = 7):
    graph, values = build_tiny_voice_decoder_graph(seed=seed)
    program = compile_graph(graph, AcceleratorConfig())
    return program, values


def test_int8_decoder_accelerates_all_matmul_ops() -> None:
    program, values = _program_and_values(seed=7)
    result = run_schedule_int8_tiled(program, values)

    assert result.accelerated_op_names == EXPECTED_MATMUL_OPS
    assert result.total_tiled_events > 0
    assert result.total_tiled_cycles > 0
    assert result.values["logits"].shape == (1, 64)
    assert result.values["next_token"].shape == (1,)


def test_int8_decoder_outputs_are_close_to_fp32_baseline() -> None:
    program, values = _program_and_values(seed=11)
    fp32 = run_schedule(program, values)
    int8 = run_schedule_int8_tiled(program, values)

    assert np.all(np.isfinite(int8.values["logits"]))
    assert np.max(np.abs(fp32["logits"] - int8.values["logits"])) < 0.05
    np.testing.assert_allclose(fp32["k_all"], int8.values["k_all"], rtol=0, atol=0.02)
    np.testing.assert_allclose(fp32["v_all"], int8.values["v_all"], rtol=0, atol=0.02)


def test_int8_decoder_schedule_is_deterministic() -> None:
    program, values = _program_and_values(seed=13)
    cfg = Int8RuntimeConfig(
        tiled_matmul=TiledMatmulConfig(tile_m=1, tile_n=8, tile_k=8, mac_lanes=16)
    )

    first = run_schedule_int8_tiled(program, values, cfg)
    second = run_schedule_int8_tiled(program, values, cfg)

    assert first.accelerated_op_names == second.accelerated_op_names
    assert first.total_tiled_cycles == second.total_tiled_cycles
    assert first.total_tiled_events == second.total_tiled_events
    np.testing.assert_allclose(first.values["logits"], second.values["logits"], rtol=0, atol=0)
    np.testing.assert_array_equal(first.values["next_token"], second.values["next_token"])


def test_int8_decoder_can_accelerate_selected_matmuls_only() -> None:
    program, values = _program_and_values(seed=17)
    cfg = Int8RuntimeConfig.only({"ffn_up", "ffn_down"})
    result = run_schedule_int8_tiled(program, values, cfg)

    assert result.accelerated_op_names == ("ffn_up", "ffn_down")
    assert result.total_tiled_cycles > 0
    assert result.values["logits"].shape == (1, 64)
