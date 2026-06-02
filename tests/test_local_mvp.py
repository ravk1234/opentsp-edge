from __future__ import annotations

import numpy as np

from opentsp.compiler import compile_graph
from opentsp.hardware import AcceleratorConfig
from opentsp.models import build_tiny_voice_decoder_graph, eager_reference
from opentsp.simulator import run_schedule


def test_schedule_is_deterministic() -> None:
    graph, _ = build_tiny_voice_decoder_graph(seed=123)
    hw = AcceleratorConfig()
    a = compile_graph(graph, hw)
    b = compile_graph(graph, hw)

    assert a.total_cycles == b.total_cycles
    assert [(x.name, x.start_cycle, x.end_cycle) for x in a.schedule] == [
        (x.name, x.start_cycle, x.end_cycle) for x in b.schedule
    ]
    assert a.allocations == b.allocations


def test_scheduled_simulator_matches_eager_reference() -> None:
    graph, values = build_tiny_voice_decoder_graph(seed=99)
    program = compile_graph(graph, AcceleratorConfig())
    out = run_schedule(program, values)
    ref = eager_reference(values, cache_len=4)

    np.testing.assert_allclose(out["logits"], ref["logits"], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out["next_token"], ref["next_token"])
    np.testing.assert_allclose(out["k_all"], ref["k_all"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out["v_all"], ref["v_all"], rtol=1e-5, atol=1e-5)


def test_small_bank_config_raises_memory_error() -> None:
    graph, _ = build_tiny_voice_decoder_graph(d_model=32, d_ff=64, vocab_size=128)
    hw = AcceleratorConfig(num_banks=2, bank_size_bytes=256)

    try:
        compile_graph(graph, hw)
    except MemoryError:
        return
    raise AssertionError("Expected MemoryError for tiny SRAM bank size")
