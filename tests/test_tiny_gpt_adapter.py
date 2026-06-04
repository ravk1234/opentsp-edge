from __future__ import annotations

import numpy as np
import pytest

from opentsp.tiny_gpt import TinyGPTConfig, build_tiny_gpt_step, tiny_gpt_fp32_reference
from opentsp.tiny_gpt_adapter import run_tiny_gpt_opentsp


def test_tiny_gpt_step_builds_realistic_decode_inputs() -> None:
    cfg = TinyGPTConfig(vocab_size=32, d_model=8, d_ff=16, max_seq_len=8, cache_len=4)
    step = build_tiny_gpt_step([1, 2, 3], config=cfg, seed=1)

    assert step.graph.name == "tiny_gpt_single_block_decode"
    assert step.values["x"].shape == (1, 8)
    assert step.values["k_cache"].shape == (4, 8)
    assert step.values["v_cache"].shape == (4, 8)
    assert step.token_embedding.shape == (32, 8)
    assert step.position_embedding.shape == (8, 8)


def test_tiny_gpt_step_rejects_invalid_tokens() -> None:
    cfg = TinyGPTConfig(vocab_size=8, d_model=8, d_ff=16, max_seq_len=4, cache_len=2)
    with pytest.raises(ValueError):
        build_tiny_gpt_step([], config=cfg)
    with pytest.raises(ValueError):
        build_tiny_gpt_step([0, 8], config=cfg)
    with pytest.raises(ValueError):
        build_tiny_gpt_step([1, 2, 3, 4, 5], config=cfg)


def test_fp32_reference_outputs_expected_shapes() -> None:
    cfg = TinyGPTConfig(vocab_size=32, d_model=8, d_ff=16, max_seq_len=8, cache_len=4)
    step = build_tiny_gpt_step([4, 2, 1], config=cfg, seed=2)
    out = tiny_gpt_fp32_reference(step)

    assert out["logits"].shape == (1, 32)
    assert out["k_all"].shape == (4, 8)
    assert out["v_all"].shape == (4, 8)
    assert out["next_token"].shape == (1,)


def test_tiny_gpt_opentsp_matches_next_token_and_builds_timeline() -> None:
    cfg = TinyGPTConfig(vocab_size=64, d_model=16, d_ff=32, max_seq_len=16, cache_len=4)
    result = run_tiny_gpt_opentsp([4, 12, 9, 3, 21], config=cfg, seed=123)

    assert result.token_match
    assert result.matmul_op_count == 7
    assert result.attention_op_count == 1
    assert result.timeline.event_count > 0
    assert result.timeline.total_cycles > 0
    assert result.max_abs_logits_error < 0.02


def test_tiny_gpt_opentsp_is_deterministic_for_same_seed() -> None:
    cfg = TinyGPTConfig(vocab_size=64, d_model=16, d_ff=32, max_seq_len=16, cache_len=4)
    a = run_tiny_gpt_opentsp([2, 5, 7, 11], config=cfg, seed=999)
    b = run_tiny_gpt_opentsp([2, 5, 7, 11], config=cfg, seed=999)

    np.testing.assert_allclose(a.fp32_values["logits"], b.fp32_values["logits"])
    np.testing.assert_allclose(a.int8_result.values["logits"], b.int8_result.values["logits"])
    assert a.timeline.total_cycles == b.timeline.total_cycles
    assert a.fp32_next_token == b.fp32_next_token
    assert a.int8_next_token == b.int8_next_token
