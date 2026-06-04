from __future__ import annotations

import pytest

from opentsp.benchmark import benchmark_tiny_gpt_adapter


def test_tiny_gpt_benchmark_smoke() -> None:
    report = benchmark_tiny_gpt_adapter(warmup=0, repeats=2, clock_mhz_values=(50, 100, 200))

    assert report.model_name == "tiny_gpt_single_block_decode"
    assert report.prompt_token_ids == (4, 12, 9, 3, 21)
    assert report.token_match
    assert report.timeline_events > 0
    assert report.timeline_cycles > 0
    assert report.matmul_micro_ops > 0
    assert report.attention_micro_ops > 0
    assert report.max_abs_logits_error < 0.05
    assert len(report.timings) == 3
    assert len(report.clock_estimates) == 3


def test_clock_estimate_formula() -> None:
    report = benchmark_tiny_gpt_adapter(warmup=0, repeats=1, clock_mhz_values=(100,))
    estimate = report.clock_estimates[0]

    assert estimate.clock_mhz == 100
    assert estimate.latency_us == pytest.approx(report.timeline_cycles / 100.0)
    assert estimate.tokens_per_second == pytest.approx(100_000_000.0 / report.timeline_cycles)


def test_timing_sections_are_named() -> None:
    report = benchmark_tiny_gpt_adapter(warmup=0, repeats=1, clock_mhz_values=(100,))
    names = {timing.name for timing in report.timings}

    assert names == {
        "tiny_gpt_fp32_reference",
        "opentsp_int8_python_sim",
        "timeline_build",
    }
