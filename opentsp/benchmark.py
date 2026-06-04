from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from time import perf_counter
from typing import Callable, Iterable, Sequence

from .accelerator_runtime import Int8RuntimeConfig, Int8RuntimeResult, run_schedule_int8_tiled
from .attention_schedule import AttentionScheduleConfig
from .compiler import compile_graph
from .hardware import AcceleratorConfig
from .tiled_matmul import TiledMatmulConfig
from .timeline import TimelineResult, build_unified_timeline
from .tiny_gpt import TinyGPTConfig, build_tiny_gpt_step, tiny_gpt_fp32_reference
from .tiny_gpt_adapter import TinyGPTAdapterResult, run_tiny_gpt_opentsp


@dataclass(frozen=True)
class TimingSummary:
    """Wall-clock timing summary for one CPU-side benchmark section."""

    name: str
    min_ms: float
    mean_ms: float
    std_ms: float
    repeats: int


@dataclass(frozen=True)
class ClockEstimate:
    """Cycle-derived accelerator estimate at one assumed clock speed."""

    clock_mhz: int
    latency_us: float
    tokens_per_second: float


@dataclass(frozen=True)
class TinyGPTBenchmarkReport:
    """Benchmark report for the tiny GPT adapter path."""

    model_name: str
    prompt_token_ids: tuple[int, ...]
    vocab_size: int
    d_model: int
    d_ff: int
    cache_len: int
    fp32_next_token: int
    int8_next_token: int
    token_match: bool
    max_abs_logits_error: float
    timeline_events: int
    timeline_cycles: int
    matmul_micro_ops: int
    attention_micro_ops: int
    matmul_cycles: int
    attention_cycles: int
    timings: tuple[TimingSummary, ...]
    clock_estimates: tuple[ClockEstimate, ...]

    @property
    def timing_by_name(self) -> dict[str, TimingSummary]:
        return {timing.name: timing for timing in self.timings}

    @property
    def estimate_by_clock_mhz(self) -> dict[int, ClockEstimate]:
        return {estimate.clock_mhz: estimate for estimate in self.clock_estimates}


def default_benchmark_runtime_config() -> Int8RuntimeConfig:
    """Return the accelerator-style config used by the benchmark."""

    return Int8RuntimeConfig(
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


def _time_section(name: str, fn: Callable[[], object], *, warmup: int, repeats: int) -> TimingSummary:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        fn()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        samples_ms.append((perf_counter() - start) * 1_000.0)

    return TimingSummary(
        name=name,
        min_ms=min(samples_ms),
        mean_ms=mean(samples_ms),
        std_ms=pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        repeats=repeats,
    )


def _clock_estimates(total_cycles: int, clock_mhz_values: Iterable[int]) -> tuple[ClockEstimate, ...]:
    if total_cycles <= 0:
        raise ValueError("total_cycles must be positive")

    estimates: list[ClockEstimate] = []
    for clock_mhz in clock_mhz_values:
        if clock_mhz <= 0:
            raise ValueError("clock speeds must be positive")
        latency_us = total_cycles / float(clock_mhz)
        tokens_per_second = (clock_mhz * 1_000_000.0) / float(total_cycles)
        estimates.append(
            ClockEstimate(
                clock_mhz=int(clock_mhz),
                latency_us=latency_us,
                tokens_per_second=tokens_per_second,
            )
        )
    return tuple(estimates)


def benchmark_tiny_gpt_adapter(
    prompt_token_ids: Sequence[int] = (4, 12, 9, 3, 21),
    *,
    config: TinyGPTConfig | None = None,
    seed: int = 123,
    warmup: int = 3,
    repeats: int = 20,
    clock_mhz_values: Sequence[int] = (50, 100, 200),
) -> TinyGPTBenchmarkReport:
    """Benchmark the tiny GPT adapter and report CPU timings plus cycle estimates.

    The wall-clock timings are real local CPU timings for the FP32 NumPy path and
    the OpenTSP Python simulator path. The accelerator latency and throughput
    numbers are derived from the deterministic timeline's cycle count; they are
    estimates until a controller, memory system, and compute tile are deployed on
    real hardware.
    """

    cfg = config or TinyGPTConfig(vocab_size=64, d_model=16, d_ff=32, max_seq_len=16, cache_len=4)
    prompt = tuple(int(token) for token in prompt_token_ids)
    runtime_config = default_benchmark_runtime_config()
    hardware = AcceleratorConfig(clock_mhz=100, mac_lanes=32, elem_lanes=16)

    step = build_tiny_gpt_step(prompt, config=cfg, seed=seed)
    program = compile_graph(step.graph, hardware)

    adapter_once: TinyGPTAdapterResult = run_tiny_gpt_opentsp(
        prompt,
        config=cfg,
        seed=seed,
        hardware=hardware,
        runtime_config=runtime_config,
    )
    int8_once: Int8RuntimeResult = adapter_once.int8_result
    timeline_once: TimelineResult = adapter_once.timeline

    timings = (
        _time_section(
            "tiny_gpt_fp32_reference",
            lambda: tiny_gpt_fp32_reference(step),
            warmup=warmup,
            repeats=repeats,
        ),
        _time_section(
            "opentsp_int8_python_sim",
            lambda: run_schedule_int8_tiled(program, step.values, runtime_config),
            warmup=warmup,
            repeats=repeats,
        ),
        _time_section(
            "timeline_build",
            lambda: build_unified_timeline(program, int8_once.matmul_stats, int8_once.attention_stats),
            warmup=warmup,
            repeats=repeats,
        ),
    )

    return TinyGPTBenchmarkReport(
        model_name=program.graph_name,
        prompt_token_ids=prompt,
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        cache_len=cfg.cache_len,
        fp32_next_token=adapter_once.fp32_next_token,
        int8_next_token=adapter_once.int8_next_token,
        token_match=adapter_once.token_match,
        max_abs_logits_error=adapter_once.max_abs_logits_error,
        timeline_events=timeline_once.event_count,
        timeline_cycles=timeline_once.total_cycles,
        matmul_micro_ops=int8_once.total_tiled_events,
        attention_micro_ops=int8_once.total_attention_events,
        matmul_cycles=int8_once.total_tiled_cycles,
        attention_cycles=int8_once.total_attention_cycles,
        timings=timings,
        clock_estimates=_clock_estimates(timeline_once.total_cycles, clock_mhz_values),
    )
