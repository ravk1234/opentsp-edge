from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .accelerator_runtime import Int8RuntimeConfig, Int8RuntimeResult, run_schedule_int8_tiled
from .compiler import CompiledProgram, compile_graph
from .hardware import AcceleratorConfig
from .timeline import TimelineResult, build_unified_timeline
from .tiny_gpt import TinyGPTConfig, TinyGPTStep, build_tiny_gpt_step, tiny_gpt_fp32_reference


@dataclass(frozen=True)
class TinyGPTAdapterResult:
    """Result of running a tiny GPT step through FP32 and OpenTSP paths."""

    step: TinyGPTStep
    program: CompiledProgram
    fp32_values: dict[str, np.ndarray]
    int8_result: Int8RuntimeResult
    timeline: TimelineResult
    max_abs_logits_error: float

    @property
    def fp32_next_token(self) -> int:
        return int(np.asarray(self.fp32_values["next_token"]).reshape(-1)[0])

    @property
    def int8_next_token(self) -> int:
        return int(np.asarray(self.int8_result.values["next_token"]).reshape(-1)[0])

    @property
    def token_match(self) -> bool:
        return self.fp32_next_token == self.int8_next_token

    @property
    def matmul_op_count(self) -> int:
        return len(self.int8_result.matmul_stats)

    @property
    def attention_op_count(self) -> int:
        return len(self.int8_result.attention_stats)


def run_tiny_gpt_opentsp(
    prompt_token_ids: Sequence[int],
    config: TinyGPTConfig | None = None,
    *,
    seed: int = 123,
    hardware: AcceleratorConfig | None = None,
    runtime_config: Int8RuntimeConfig | None = None,
) -> TinyGPTAdapterResult:
    """Run a tiny GPT-style decode step through FP32 and OpenTSP INT8 paths."""

    step = build_tiny_gpt_step(prompt_token_ids, config=config, seed=seed)
    program = compile_graph(step.graph, hardware or AcceleratorConfig())

    fp32_values = tiny_gpt_fp32_reference(step)
    int8_result = run_schedule_int8_tiled(program, step.values, runtime_config or Int8RuntimeConfig())
    timeline = build_unified_timeline(program, int8_result.matmul_stats, int8_result.attention_stats)

    fp32_logits = np.asarray(fp32_values["logits"], dtype=np.float32)
    int8_logits = np.asarray(int8_result.values["logits"], dtype=np.float32)
    max_abs_logits_error = float(np.max(np.abs(fp32_logits - int8_logits)))

    return TinyGPTAdapterResult(
        step=step,
        program=program,
        fp32_values=fp32_values,
        int8_result=int8_result,
        timeline=timeline,
        max_abs_logits_error=max_abs_logits_error,
    )
