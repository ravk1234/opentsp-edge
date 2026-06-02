from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Set

import numpy as np

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
class Int8RuntimeConfig:
    """Configuration for graph execution with tiled INT8 matmul.

    By default every `matmul` op in the graph is accelerated. Use
    `accelerate_matmul_names` to accelerate only selected op names.
    """

    tiled_matmul: TiledMatmulConfig = field(default_factory=TiledMatmulConfig)
    accelerate_matmul_names: Optional[frozenset[str]] = None

    @classmethod
    def only(cls, names: Iterable[str], tiled_matmul: Optional[TiledMatmulConfig] = None) -> "Int8RuntimeConfig":
        return cls(
            tiled_matmul=tiled_matmul or TiledMatmulConfig(),
            accelerate_matmul_names=frozenset(names),
        )


@dataclass(frozen=True)
class Int8RuntimeResult:
    """Outputs and metadata from deterministic INT8 graph execution."""

    values: ArrayMap
    matmul_stats: tuple[Int8MatmulStats, ...]

    @property
    def accelerated_op_names(self) -> tuple[str, ...]:
        return tuple(stat.op_name for stat in self.matmul_stats)

    @property
    def total_tiled_events(self) -> int:
        return sum(stat.event_count for stat in self.matmul_stats)

    @property
    def total_tiled_cycles(self) -> int:
        return sum(stat.cycles for stat in self.matmul_stats)


def _should_accelerate_matmul(op: ScheduledOp, config: Int8RuntimeConfig) -> bool:
    if op.kind != "matmul":
        return False
    if config.accelerate_matmul_names is None:
        return True
    return op.name in config.accelerate_matmul_names


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
    """Run a compiled graph while replacing matmul ops with tiled INT8 matmul.

    Non-matmul operations still execute with the same FP32 semantics as the
    baseline simulator. This is intentional for Milestone 3: it isolates the
    accelerator-style INT8 matmul path while preserving the rest of the decoder.
    """

    cfg = config or Int8RuntimeConfig()
    values: ArrayMap = {k: np.asarray(v).copy() for k, v in inputs_and_weights.items()}
    stats: list[Int8MatmulStats] = []

    for op in program.schedule:
        if _should_accelerate_matmul(op, cfg):
            stats.append(_run_int8_matmul_op(op, values, cfg))
        else:
            _execute_fp32_op(op, values)

    return Int8RuntimeResult(values=values, matmul_stats=tuple(stats))
