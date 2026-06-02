from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import List, Tuple

import numpy as np

from .quant import QuantizedTensor


@dataclass(frozen=True)
class TiledMatmulConfig:
    """Abstract local hardware model for tiled INT8 matmul.

    This is not a real FPGA implementation yet. It simulates the kind of work a
    deterministic accelerator would do: load tiles, MAC over tiles, accumulate,
    and store the output tile with a fixed cycle estimate.
    """

    tile_m: int = 2
    tile_n: int = 4
    tile_k: int = 8
    mac_lanes: int = 16
    load_lanes: int = 16
    store_lanes: int = 8
    load_overhead_cycles: int = 2
    compute_overhead_cycles: int = 1
    store_overhead_cycles: int = 2
    clock_mhz: int = 100

    def validate(self) -> None:
        fields = {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "mac_lanes": self.mac_lanes,
            "load_lanes": self.load_lanes,
            "store_lanes": self.store_lanes,
        }
        for name, value in fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class TiledMatmulEvent:
    index: int
    kind: str
    m_range: Tuple[int, int]
    n_range: Tuple[int, int]
    k_range: Tuple[int, int]
    start_cycle: int
    end_cycle: int
    cycles: int
    bytes_moved: int = 0
    macs: int = 0


@dataclass(frozen=True)
class TiledMatmulResult:
    acc_int32: np.ndarray
    out_fp32: np.ndarray
    events: List[TiledMatmulEvent]
    total_cycles: int

    @property
    def latency_us(self) -> float:
        # Use config clock printed externally if needed. Kept for backward-safe API.
        return float(self.total_cycles)


def _span(start: int, tile: int, limit: int) -> Tuple[int, int]:
    return start, min(start + tile, limit)


def _load_cycles(elements: int, cfg: TiledMatmulConfig) -> int:
    return cfg.load_overhead_cycles + ceil(elements / cfg.load_lanes)


def _compute_cycles(m: int, n: int, k: int, cfg: TiledMatmulConfig) -> int:
    return cfg.compute_overhead_cycles + ceil((m * n * k) / cfg.mac_lanes)


def _store_cycles(elements: int, cfg: TiledMatmulConfig) -> int:
    return cfg.store_overhead_cycles + ceil(elements / cfg.store_lanes)


def int8_tiled_matmul(a: QuantizedTensor, b: QuantizedTensor, cfg: TiledMatmulConfig | None = None) -> TiledMatmulResult:
    """Run a deterministic tiled INT8 matmul simulation.

    Shapes:
        A: [M, K] int8
        B: [K, N] int8
        C: [M, N] int32 accumulator, then dequantized to float32

    The implementation intentionally uses explicit Python loops at tile level.
    This lets us inspect a fixed micro-schedule instead of hiding everything
    behind a single `a @ b` call.
    """
    cfg = cfg or TiledMatmulConfig()
    cfg.validate()

    if a.q.ndim != 2 or b.q.ndim != 2:
        raise ValueError("int8_tiled_matmul expects 2D matrices")
    m_total, k_total = a.q.shape
    k_b, n_total = b.q.shape
    if k_total != k_b:
        raise ValueError(f"matmul shape mismatch: {a.q.shape} x {b.q.shape}")

    a_i32 = a.q.astype(np.int32)
    b_i32 = b.q.astype(np.int32)
    acc = np.zeros((m_total, n_total), dtype=np.int32)

    events: List[TiledMatmulEvent] = []
    cycle = 0
    event_index = 0

    for m0 in range(0, m_total, cfg.tile_m):
        m1 = min(m0 + cfg.tile_m, m_total)
        m = m1 - m0
        for n0 in range(0, n_total, cfg.tile_n):
            n1 = min(n0 + cfg.tile_n, n_total)
            n = n1 - n0

            # One output tile accumulates over all K tiles.
            tile_acc = np.zeros((m, n), dtype=np.int32)

            for k0 in range(0, k_total, cfg.tile_k):
                k1 = min(k0 + cfg.tile_k, k_total)
                k = k1 - k0

                # Load A tile.
                a_elems = m * k
                cycles = _load_cycles(a_elems, cfg)
                events.append(
                    TiledMatmulEvent(
                        index=event_index,
                        kind="load_a_tile",
                        m_range=(m0, m1),
                        n_range=(n0, n1),
                        k_range=(k0, k1),
                        start_cycle=cycle,
                        end_cycle=cycle + cycles,
                        cycles=cycles,
                        bytes_moved=a_elems,  # int8, so 1 byte/element
                    )
                )
                cycle += cycles
                event_index += 1

                # Load B tile.
                b_elems = k * n
                cycles = _load_cycles(b_elems, cfg)
                events.append(
                    TiledMatmulEvent(
                        index=event_index,
                        kind="load_b_tile",
                        m_range=(m0, m1),
                        n_range=(n0, n1),
                        k_range=(k0, k1),
                        start_cycle=cycle,
                        end_cycle=cycle + cycles,
                        cycles=cycles,
                        bytes_moved=b_elems,
                    )
                )
                cycle += cycles
                event_index += 1

                # Compute tile MACs.
                cycles = _compute_cycles(m, n, k, cfg)
                events.append(
                    TiledMatmulEvent(
                        index=event_index,
                        kind="mac_tile",
                        m_range=(m0, m1),
                        n_range=(n0, n1),
                        k_range=(k0, k1),
                        start_cycle=cycle,
                        end_cycle=cycle + cycles,
                        cycles=cycles,
                        macs=m * n * k,
                    )
                )
                cycle += cycles
                event_index += 1

                tile_acc += a_i32[m0:m1, k0:k1] @ b_i32[k0:k1, n0:n1]

            # Store final output tile after all K partials are accumulated.
            out_elems = m * n
            cycles = _store_cycles(out_elems, cfg)
            events.append(
                TiledMatmulEvent(
                    index=event_index,
                    kind="store_c_tile",
                    m_range=(m0, m1),
                    n_range=(n0, n1),
                    k_range=(0, k_total),
                    start_cycle=cycle,
                    end_cycle=cycle + cycles,
                    cycles=cycles,
                    bytes_moved=out_elems * 4,  # int32 output accumulator
                )
            )
            cycle += cycles
            event_index += 1

            acc[m0:m1, n0:n1] = tile_acc

    out = acc.astype(np.float32) * np.float32(a.scale * b.scale)
    return TiledMatmulResult(acc_int32=acc, out_fp32=out, events=events, total_cycles=cycle)
