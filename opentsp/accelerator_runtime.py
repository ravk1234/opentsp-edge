from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from .attention_schedule import AttentionScheduleConfig, AttentionScheduleEvent, schedule_attention_decode
from .compiler import CompiledProgram, ScheduledOp
from .quant import QuantizedTensor, quantize_symmetric_int8
from .simulator import _execute_op as _execute_fp32_op  # Reuse the FP32 op semantics for non-matmul ops.
from .tiled_matmul import TiledMatmulConfig, TiledMatmulEvent, int8_tiled_matmul


ArrayMap = Dict[str, np.ndarray]


@dataclass(frozen=True)
class Int8MatmulStats:
    """Per-matmul execution metadata from the INT8 tiled backend."""

    op_name: str
    input_name: str
    weight_name: str
    output_name: str
    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_scale: float
    weight_scale: float
    events: tuple[TiledMatmulEvent, ...]
    cycles: int

    @property
    def event_count(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class AttentionDecodeStats:
    """Per-attention execution metadata from the KV-cache scheduler."""

    op_name: str
    q_name: str
    k_cache_name: str
    v_cache_name: str
    output_name: str
    q_shape: tuple[int, ...]
    k_cache_shape: tuple[int, ...]
    v_cache_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    events: tuple[AttentionScheduleEvent, ...]
    cycles: int

    @property
    def event_count(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class Int8RuntimeConfig:
    """Configuration for graph execution with accelerator-style backends.

    By default every `matmul` op is replaced with tiled INT8 matmul and every
    `attention_decode` op is replaced with an explicit KV-cache attention
    schedule. Use the optional name sets to restrict acceleration to selected
    ops.
    """

    tiled_matmul: TiledMatmulConfig = field(default_factory=TiledMatmulConfig)
    attention: AttentionScheduleConfig = field(default_factory=AttentionScheduleConfig)
    accelerate_matmul_names: Optional[frozenset[str]] = None
    accelerate_attention_names: Optional[frozenset[str]] = None

    @classmethod
    def only(cls, names: Iterable[str], tiled_matmul: Optional[TiledMatmulConfig] = None) -> "Int8RuntimeConfig":
        return cls(
            tiled_matmul=tiled_matmul or TiledMatmulConfig(),
            accelerate_matmul_names=frozenset(names),
            # Preserve the old helper behavior: only selected matmuls are
            # accelerated, while attention falls back to the baseline FP32 op.
            accelerate_attention_names=frozenset(),
        )

    @classmethod
    def only_attention(
        cls,
        names: Iterable[str],
        attention: Optional[AttentionScheduleConfig] = None,
    ) -> "Int8RuntimeConfig":
        return cls(
            attention=attention or AttentionScheduleConfig(),
            accelerate_matmul_names=frozenset(),
            accelerate_attention_names=frozenset(names),
        )


@dataclass(frozen=True)
class Int8RuntimeResult:
    """Outputs and metadata from deterministic accelerator-style execution."""

    values: ArrayMap
    matmul_stats: tuple[Int8MatmulStats, ...]
    attention_stats: tuple[AttentionDecodeStats, ...]

    @property
    def accelerated_op_names(self) -> tuple[str, ...]:
        return tuple(stat.op_name for stat in self.matmul_stats)

    @property
    def attention_op_names(self) -> tuple[str, ...]:
        return tuple(stat.op_name for stat in self.attention_stats)

    @property
    def total_tiled_events(self) -> int:
        return sum(stat.event_count for stat in self.matmul_stats)

    @property
    def total_tiled_cycles(self) -> int:
        return sum(stat.cycles for stat in self.matmul_stats)

    @property
    def total_attention_events(self) -> int:
        return sum(stat.event_count for stat in self.attention_stats)

    @property
    def total_attention_cycles(self) -> int:
        return sum(stat.cycles for stat in self.attention_stats)

    @property
    def total_accelerator_events(self) -> int:
        return self.total_tiled_events + self.total_attention_events

    @property
    def total_accelerator_cycles(self) -> int:
        return self.total_tiled_cycles + self.total_attention_cycles


def _should_accelerate_matmul(op: ScheduledOp, config: Int8RuntimeConfig) -> bool:
    if op.kind != "matmul":
        return False
    if config.accelerate_matmul_names is None:
        return True
    return op.name in config.accelerate_matmul_names


def _should_schedule_attention(op: ScheduledOp, config: Int8RuntimeConfig) -> bool:
    if op.kind != "attention_decode":
        return False
    if config.accelerate_attention_names is None:
        return True
    return op.name in config.accelerate_attention_names


def _run_attention_decode_op(op: ScheduledOp, values: ArrayMap, config: Int8RuntimeConfig) -> AttentionDecodeStats:
    q_name, k_name, v_name = op.inputs[0], op.inputs[1], op.inputs[2]
    out_name = op.outputs[0]

    q = np.asarray(values[q_name], dtype=np.float32)
    k_cache = np.asarray(values[k_name], dtype=np.float32)
    v_cache = np.asarray(values[v_name], dtype=np.float32)

    scheduled = schedule_attention_decode(q, k_cache, v_cache, config.attention)
    values[out_name] = scheduled.context.astype(np.float32)

    return AttentionDecodeStats(
        op_name=op.name,
        q_name=q_name,
        k_cache_name=k_name,
        v_cache_name=v_name,
        output_name=out_name,
        q_shape=tuple(q.shape),
        k_cache_shape=tuple(k_cache.shape),
        v_cache_shape=tuple(v_cache.shape),
        output_shape=tuple(scheduled.context.shape),
        events=tuple(scheduled.events),
        cycles=scheduled.total_cycles,
    )


def _run_int8_matmul_op(op: ScheduledOp, values: ArrayMap, config: Int8RuntimeConfig) -> Int8MatmulStats:
    a_name, b_name = op.inputs[0], op.inputs[1]
    out_name = op.outputs[0]

    a_fp32 = np.asarray(values[a_name], dtype=np.float32)
    b_fp32 = np.asarray(values[b_name], dtype=np.float32)

    if a_fp32.ndim != 2 or b_fp32.ndim != 2:
        raise ValueError(f"INT8 tiled matmul expects 2D tensors for op {op.name}")

    qa: QuantizedTensor = quantize_symmetric_int8(a_fp32)
    qb: QuantizedTensor = quantize_symmetric_int8(b_fp32)
    tiled = int8_tiled_matmul(qa, qb, config.tiled_matmul)

    values[out_name] = tiled.out_fp32.astype(np.float32)

    return Int8MatmulStats(
        op_name=op.name,
        input_name=a_name,
        weight_name=b_name,
        output_name=out_name,
        input_shape=tuple(a_fp32.shape),
        weight_shape=tuple(b_fp32.shape),
        output_shape=tuple(tiled.out_fp32.shape),
        input_scale=float(qa.scale),
        weight_scale=float(qb.scale),
        events=tuple(tiled.events),
        cycles=tiled.total_cycles,
    )


def run_schedule_int8_tiled(
    program: CompiledProgram,
    inputs_and_weights: Mapping[str, np.ndarray],
    config: Optional[Int8RuntimeConfig] = None,
) -> Int8RuntimeResult:
    """Run a compiled graph with deterministic accelerator-style backends.

    Matmul ops can be replaced with tiled INT8 matmul. Attention decode ops can
    be replaced with an explicit KV-cache schedule. Remaining operations execute
    with the same FP32 semantics as the baseline simulator.
    """

    cfg = config or Int8RuntimeConfig()
    values: ArrayMap = {k: np.asarray(v).copy() for k, v in inputs_and_weights.items()}
    matmul_stats: list[Int8MatmulStats] = []
    attention_stats: list[AttentionDecodeStats] = []

    for op in program.schedule:
        if _should_accelerate_matmul(op, cfg):
            matmul_stats.append(_run_int8_matmul_op(op, values, cfg))
        elif _should_schedule_attention(op, cfg):
            attention_stats.append(_run_attention_decode_op(op, values, cfg))
        else:
            _execute_fp32_op(op, values)

    return Int8RuntimeResult(
        values=values,
        matmul_stats=tuple(matmul_stats),
        attention_stats=tuple(attention_stats),
    )
