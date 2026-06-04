from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .accelerator_runtime import Int8RuntimeConfig
from .attention_schedule import schedule_attention_decode
from .compiler import CompiledProgram, ScheduledOp
from .quant import QuantizedTensor, quantize_symmetric_int8
from .rtl_test_vectors import SystolicTileVector, make_systolic_tile_vector
from .simulator import _execute_op as _execute_fp32_op
from .tiled_matmul import TiledMatmulConfig, TiledMatmulEvent, int8_tiled_matmul


@dataclass(frozen=True)
class ScheduleVectorGroup:
    """RTL vectors extracted from one scheduled INT8 matmul op."""

    op_name: str
    input_name: str
    weight_name: str
    output_name: str
    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    vectors: tuple[SystolicTileVector, ...]
    schedule_event_count: int
    schedule_cycles: int

    @property
    def vector_count(self) -> int:
        return len(self.vectors)

    @property
    def k_tile_count(self) -> int:
        return sum(v.k_tiles for v in self.vectors)


@dataclass(frozen=True)
class ScheduleVectorExtractionResult:
    """All RTL vectors extracted while executing a compiled graph."""

    values: dict[str, np.ndarray]
    groups: tuple[ScheduleVectorGroup, ...]

    @property
    def total_vectors(self) -> int:
        return sum(group.vector_count for group in self.groups)

    @property
    def total_k_tiles(self) -> int:
        return sum(group.k_tile_count for group in self.groups)

    @property
    def op_names(self) -> tuple[str, ...]:
        return tuple(group.op_name for group in self.groups)


def _zero_padded_a_tile(a_i8: np.ndarray, row0: int, k0: int, m_limit: int, event_k_stop: int) -> np.ndarray:
    tile = np.zeros((2, 2), dtype=np.int8)
    row1 = min(row0 + 2, m_limit)
    k1 = min(k0 + 2, event_k_stop)
    tile[: row1 - row0, : k1 - k0] = a_i8[row0:row1, k0:k1]
    return tile


def _zero_padded_b_tile(b_i8: np.ndarray, k0: int, col0: int, event_k_stop: int, n_limit: int) -> np.ndarray:
    tile = np.zeros((2, 2), dtype=np.int8)
    k1 = min(k0 + 2, event_k_stop)
    col1 = min(col0 + 2, n_limit)
    tile[: k1 - k0, : col1 - col0] = b_i8[k0:k1, col0:col1]
    return tile


def _mac_events_by_output_tile(events: Sequence[TiledMatmulEvent]) -> dict[tuple[int, int, int, int], list[TiledMatmulEvent]]:
    grouped: dict[tuple[int, int, int, int], list[TiledMatmulEvent]] = {}
    for event in events:
        if event.kind != "mac_tile":
            continue
        key = (event.m_range[0], event.m_range[1], event.n_range[0], event.n_range[1])
        grouped.setdefault(key, []).append(event)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda item: item.index)
    return grouped


def systolic_vectors_from_tiled_matmul_schedule(
    op_name: str,
    a: QuantizedTensor,
    b: QuantizedTensor,
    events: Sequence[TiledMatmulEvent],
) -> list[SystolicTileVector]:
    """Extract 2x2 RTL vectors from real tiled-matmul micro-op events.

    The Python tiled matmul can use larger tiles such as M=2, N=8, K=8. The
    current RTL compute primitive is a 2x2 tile that consumes K in chunks of 2.
    This function decomposes each scheduled `mac_tile` region into those exact
    2x2 RTL-compatible steps while preserving the event order from the schedule.

    Edge tiles are zero padded, matching the RTL/Python contract introduced in
    earlier milestones.
    """

    if a.q.ndim != 2 or b.q.ndim != 2:
        raise ValueError("expected 2D quantized tensors")
    m_total, k_total = a.q.shape
    k_b, n_total = b.q.shape
    if k_total != k_b:
        raise ValueError(f"matmul shape mismatch: {a.q.shape} x {b.q.shape}")

    a_i8 = np.asarray(a.q, dtype=np.int8)
    b_i8 = np.asarray(b.q, dtype=np.int8)
    vectors: list[SystolicTileVector] = []

    for (m0, m1, n0, n1), mac_events in sorted(_mac_events_by_output_tile(events).items()):
        for row0 in range(m0, m1, 2):
            for col0 in range(n0, n1, 2):
                a_tiles: list[np.ndarray] = []
                b_tiles: list[np.ndarray] = []
                for event in mac_events:
                    k0, k1 = event.k_range
                    for kk in range(k0, k1, 2):
                        a_tiles.append(_zero_padded_a_tile(a_i8, row0, kk, m_total, k1))
                        b_tiles.append(_zero_padded_b_tile(b_i8, kk, col0, k1, n_total))

                if not a_tiles:
                    continue
                valid_rows = min(row0 + 2, m_total) - row0
                valid_cols = min(col0 + 2, n_total) - col0
                vectors.append(
                    make_systolic_tile_vector(
                        name=f"{op_name}_m{row0}_{row0 + valid_rows}_n{col0}_{col0 + valid_cols}",
                        a_tiles=a_tiles,
                        b_tiles=b_tiles,
                        description=(
                            f"Extracted from scheduled matmul {op_name}: "
                            f"M({row0},{row0 + valid_rows}) N({col0},{col0 + valid_cols})."
                        ),
                    )
                )
    return vectors


def generate_systolic_vectors_from_int8_matmul(
    op_name: str,
    a_fp32: np.ndarray,
    b_fp32: np.ndarray,
    cfg: TiledMatmulConfig | None = None,
) -> tuple[list[SystolicTileVector], np.ndarray]:
    """Quantize, schedule, and emit RTL vectors for one real tiled matmul."""

    qa = quantize_symmetric_int8(np.asarray(a_fp32, dtype=np.float32))
    qb = quantize_symmetric_int8(np.asarray(b_fp32, dtype=np.float32))
    tiled = int8_tiled_matmul(qa, qb, cfg or TiledMatmulConfig())
    vectors = systolic_vectors_from_tiled_matmul_schedule(op_name, qa, qb, tiled.events)
    return vectors, tiled.acc_int32


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


def collect_schedule_vectors_from_program(
    program: CompiledProgram,
    inputs_and_weights: Mapping[str, np.ndarray],
    config: Int8RuntimeConfig | None = None,
) -> ScheduleVectorExtractionResult:
    """Execute a graph and emit RTL vectors from each actual matmul schedule.

    This mirrors the INT8 runtime: accelerated matmul ops use quantized tiled
    matmul, attention ops use the deterministic KV-cache scheduler, and all
    remaining ops use the baseline FP32 semantics. During each accelerated
    matmul, the function emits 2x2 RTL vectors from the exact tiled micro-op
    events for that op.
    """

    cfg = config or Int8RuntimeConfig()
    values: dict[str, np.ndarray] = {k: np.asarray(v).copy() for k, v in inputs_and_weights.items()}
    groups: list[ScheduleVectorGroup] = []

    for op in program.schedule:
        if _should_accelerate_matmul(op, cfg):
            a_name, b_name = op.inputs[0], op.inputs[1]
            out_name = op.outputs[0]
            a_fp32 = np.asarray(values[a_name], dtype=np.float32)
            b_fp32 = np.asarray(values[b_name], dtype=np.float32)
            qa = quantize_symmetric_int8(a_fp32)
            qb = quantize_symmetric_int8(b_fp32)
            tiled = int8_tiled_matmul(qa, qb, cfg.tiled_matmul)
            values[out_name] = tiled.out_fp32.astype(np.float32)
            vectors = systolic_vectors_from_tiled_matmul_schedule(op.name, qa, qb, tiled.events)
            groups.append(
                ScheduleVectorGroup(
                    op_name=op.name,
                    input_name=a_name,
                    weight_name=b_name,
                    output_name=out_name,
                    input_shape=tuple(a_fp32.shape),
                    weight_shape=tuple(b_fp32.shape),
                    output_shape=tuple(tiled.out_fp32.shape),
                    vectors=tuple(vectors),
                    schedule_event_count=len(tiled.events),
                    schedule_cycles=tiled.total_cycles,
                )
            )
        elif _should_schedule_attention(op, cfg):
            q_name, k_name, v_name = op.inputs[0], op.inputs[1], op.inputs[2]
            out_name = op.outputs[0]
            scheduled = schedule_attention_decode(
                np.asarray(values[q_name], dtype=np.float32),
                np.asarray(values[k_name], dtype=np.float32),
                np.asarray(values[v_name], dtype=np.float32),
                cfg.attention,
            )
            values[out_name] = scheduled.context.astype(np.float32)
        else:
            _execute_fp32_op(op, values)

    return ScheduleVectorExtractionResult(values=values, groups=tuple(groups))
