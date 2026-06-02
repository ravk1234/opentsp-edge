from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class AttentionScheduleConfig:
    """Deterministic local hardware model for one-token KV-cache attention.

    This is a software simulator for the schedule that a small accelerator could
    execute. It does not try to be fast Python. It makes the cache movement and
    score/value phases visible and cycle-counted.
    """

    cache_tile: int = 2
    vector_lanes: int = 16
    mac_lanes: int = 16
    load_lanes: int = 16
    store_lanes: int = 8
    load_overhead_cycles: int = 2
    compute_overhead_cycles: int = 1
    reduce_overhead_cycles: int = 3
    store_overhead_cycles: int = 2
    clock_mhz: int = 100

    def validate(self) -> None:
        fields = {
            "cache_tile": self.cache_tile,
            "vector_lanes": self.vector_lanes,
            "mac_lanes": self.mac_lanes,
            "load_lanes": self.load_lanes,
            "store_lanes": self.store_lanes,
        }
        for name, value in fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class AttentionScheduleEvent:
    index: int
    kind: str
    token_range: Tuple[int, int]
    dim_range: Tuple[int, int]
    start_cycle: int
    end_cycle: int
    cycles: int
    bytes_moved: int = 0
    macs: int = 0


@dataclass(frozen=True)
class AttentionScheduleResult:
    context: np.ndarray
    scores: np.ndarray
    probs: np.ndarray
    events: List[AttentionScheduleEvent]
    total_cycles: int

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def latency_us(self) -> float:
        # Kept simple for consistency with the rest of the local simulator.
        return float(self.total_cycles)


def _ceil_div_work(work: int, lanes: int) -> int:
    return ceil(work / lanes)


def _load_cycles(elements: int, cfg: AttentionScheduleConfig) -> int:
    return cfg.load_overhead_cycles + _ceil_div_work(elements, cfg.load_lanes)


def _store_cycles(elements: int, cfg: AttentionScheduleConfig) -> int:
    return cfg.store_overhead_cycles + _ceil_div_work(elements, cfg.store_lanes)


def _compute_cycles(macs: int, cfg: AttentionScheduleConfig) -> int:
    return cfg.compute_overhead_cycles + _ceil_div_work(macs, cfg.mac_lanes)


def _vector_cycles(elements: int, overhead: int, cfg: AttentionScheduleConfig) -> int:
    return overhead + _ceil_div_work(elements, cfg.vector_lanes)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    ex = np.exp(shifted)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def schedule_attention_decode(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    cfg: AttentionScheduleConfig | None = None,
) -> AttentionScheduleResult:
    """Run a deterministic KV-cache attention schedule for one decode token.

    Shapes:
        q:       [1, D]
        k_cache: [T, D]
        v_cache: [T, D]
        context: [1, D]

    Schedule phases:
        1. Load the query vector once.
        2. Tile over cached keys to compute attention scores.
        3. Run softmax as explicit max/sum/normalize phases.
        4. Tile over cached values to accumulate the context vector.
        5. Store the context vector.
    """

    cfg = cfg or AttentionScheduleConfig()
    cfg.validate()

    q = np.asarray(q, dtype=np.float32)
    k_cache = np.asarray(k_cache, dtype=np.float32)
    v_cache = np.asarray(v_cache, dtype=np.float32)

    if q.ndim != 2 or q.shape[0] != 1:
        raise ValueError("q must have shape [1, D]")
    if k_cache.ndim != 2 or v_cache.ndim != 2:
        raise ValueError("k_cache and v_cache must have shape [T, D]")
    if k_cache.shape != v_cache.shape:
        raise ValueError(f"k_cache and v_cache shapes must match: {k_cache.shape} vs {v_cache.shape}")
    if q.shape[1] != k_cache.shape[1]:
        raise ValueError(f"q/cache dimension mismatch: {q.shape} vs {k_cache.shape}")

    tokens, dim = k_cache.shape
    events: List[AttentionScheduleEvent] = []
    cycle = 0
    event_index = 0

    def add_event(kind: str, token_range: Tuple[int, int], dim_range: Tuple[int, int], cycles: int, bytes_moved: int = 0, macs: int = 0) -> None:
        nonlocal cycle, event_index
        events.append(
            AttentionScheduleEvent(
                index=event_index,
                kind=kind,
                token_range=token_range,
                dim_range=dim_range,
                start_cycle=cycle,
                end_cycle=cycle + cycles,
                cycles=cycles,
                bytes_moved=bytes_moved,
                macs=macs,
            )
        )
        cycle += cycles
        event_index += 1

    # Query stays resident while scanning the KV cache.
    add_event(
        "load_q_vector",
        token_range=(0, 1),
        dim_range=(0, dim),
        cycles=_load_cycles(dim, cfg),
        bytes_moved=dim * 4,
    )

    scores = np.zeros((1, tokens), dtype=np.float32)
    scale = np.float32(1.0 / sqrt(float(dim)))

    # Score phase: q dot each cached key tile.
    for t0 in range(0, tokens, cfg.cache_tile):
        t1 = min(t0 + cfg.cache_tile, tokens)
        tile_tokens = t1 - t0

        add_event(
            "load_k_tile",
            token_range=(t0, t1),
            dim_range=(0, dim),
            cycles=_load_cycles(tile_tokens * dim, cfg),
            bytes_moved=tile_tokens * dim * 4,
        )
        add_event(
            "score_tile",
            token_range=(t0, t1),
            dim_range=(0, dim),
            cycles=_compute_cycles(tile_tokens * dim, cfg),
            macs=tile_tokens * dim,
        )
        scores[:, t0:t1] = (q @ k_cache[t0:t1, :].T) * scale

    # Softmax is separated into deterministic reduction/vector phases so it can
    # later be lowered into a real controller instead of one opaque Python call.
    add_event(
        "softmax_max_reduce",
        token_range=(0, tokens),
        dim_range=(0, tokens),
        cycles=_vector_cycles(tokens, cfg.reduce_overhead_cycles, cfg),
    )
    add_event(
        "softmax_exp_sum",
        token_range=(0, tokens),
        dim_range=(0, tokens),
        cycles=_vector_cycles(tokens, cfg.reduce_overhead_cycles, cfg),
    )
    add_event(
        "softmax_normalize",
        token_range=(0, tokens),
        dim_range=(0, tokens),
        cycles=_vector_cycles(tokens, cfg.compute_overhead_cycles, cfg),
    )
    probs = _softmax(scores).astype(np.float32)

    context = np.zeros((1, dim), dtype=np.float32)

    # Value phase: weighted sum over cached value tiles.
    for t0 in range(0, tokens, cfg.cache_tile):
        t1 = min(t0 + cfg.cache_tile, tokens)
        tile_tokens = t1 - t0

        add_event(
            "load_v_tile",
            token_range=(t0, t1),
            dim_range=(0, dim),
            cycles=_load_cycles(tile_tokens * dim, cfg),
            bytes_moved=tile_tokens * dim * 4,
        )
        add_event(
            "value_tile",
            token_range=(t0, t1),
            dim_range=(0, dim),
            cycles=_compute_cycles(tile_tokens * dim, cfg),
            macs=tile_tokens * dim,
        )
        context += probs[:, t0:t1] @ v_cache[t0:t1, :]

    add_event(
        "store_context",
        token_range=(0, 1),
        dim_range=(0, dim),
        cycles=_store_cycles(dim, cfg),
        bytes_moved=dim * 4,
    )

    return AttentionScheduleResult(
        context=context.astype(np.float32),
        scores=scores.astype(np.float32),
        probs=probs.astype(np.float32),
        events=events,
        total_cycles=cycle,
    )
