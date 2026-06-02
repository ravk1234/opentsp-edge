from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


INT8_MIN = -128
INT8_MAX = 127


@dataclass(frozen=True)
class SystolicTile2x2Contract:
    """Software contract for rtl/systolic_tile_2x2.sv.

    The RTL block consumes one signed INT8 A[2,2] tile and one signed INT8
    B[2,2] tile per valid cycle and accumulates into an INT32 C[2,2] tile:

        C += A @ B
    """

    tile_m: int = 2
    tile_n: int = 2
    tile_k: int = 2
    input_dtype: str = "int8"
    accumulator_dtype: str = "int32"


def _as_int8_array(name: str, value: Sequence[Sequence[int]] | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must contain integer values")
    if int(arr.min()) < INT8_MIN or int(arr.max()) > INT8_MAX:
        raise ValueError(f"{name} values must fit signed INT8 range [{INT8_MIN}, {INT8_MAX}]")
    return arr.astype(np.int8)


def _as_int32_acc(value: Sequence[Sequence[int]] | np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((2, 2), dtype=np.int32)
    arr = np.asarray(value)
    if arr.shape != (2, 2):
        raise ValueError(f"acc must have shape (2, 2), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("acc must contain integer values")
    return arr.astype(np.int32)


def systolic_tile_2x2_step(
    a_tile: Sequence[Sequence[int]] | np.ndarray,
    b_tile: Sequence[Sequence[int]] | np.ndarray,
    acc: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> np.ndarray:
    """Apply one RTL-equivalent 2x2 signed INT8 tile MAC step.

    Args:
        a_tile: signed INT8-compatible 2x2 A tile.
        b_tile: signed INT8-compatible 2x2 B tile.
        acc: optional existing INT32 2x2 accumulator.

    Returns:
        INT32 2x2 accumulator after `acc + a_tile @ b_tile`.
    """

    a = _as_int8_array("a_tile", a_tile, (2, 2)).astype(np.int32)
    b = _as_int8_array("b_tile", b_tile, (2, 2)).astype(np.int32)
    c = _as_int32_acc(acc)
    return c + (a @ b).astype(np.int32)


def systolic_tile_2x2_sequence(
    tile_pairs: Iterable[tuple[Sequence[Sequence[int]] | np.ndarray, Sequence[Sequence[int]] | np.ndarray]],
    acc: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate a deterministic sequence of RTL-equivalent 2x2 tile steps."""

    c = _as_int32_acc(acc)
    for a_tile, b_tile in tile_pairs:
        c = systolic_tile_2x2_step(a_tile, b_tile, c)
    return c


def matmul_2xk_by_kx2_via_rtl_contract(
    a: Sequence[Sequence[int]] | np.ndarray,
    b: Sequence[Sequence[int]] | np.ndarray,
    *,
    pad_odd_k: bool = True,
) -> np.ndarray:
    """Compute a 2xK by Kx2 INT8 matmul using the 2x2 RTL tile contract.

    The hardware tile consumes K in chunks of 2. When K is odd and
    `pad_odd_k=True`, the final K slice is zero padded, which is equivalent to
    the mathematical matmul for the original unpadded inputs.
    """

    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError("a and b must be 2D matrices")
    if a_arr.shape[0] != 2:
        raise ValueError(f"a must have 2 rows, got {a_arr.shape}")
    if b_arr.shape[1] != 2:
        raise ValueError(f"b must have 2 columns, got {b_arr.shape}")
    if a_arr.shape[1] != b_arr.shape[0]:
        raise ValueError(f"shape mismatch: {a_arr.shape} x {b_arr.shape}")

    k_total = a_arr.shape[1]
    if k_total % 2 != 0 and not pad_odd_k:
        raise ValueError("K must be even when pad_odd_k=False")

    if not np.issubdtype(a_arr.dtype, np.integer) or not np.issubdtype(b_arr.dtype, np.integer):
        raise TypeError("a and b must contain integer values")
    if int(a_arr.min()) < INT8_MIN or int(a_arr.max()) > INT8_MAX:
        raise ValueError("a values must fit signed INT8 range")
    if int(b_arr.min()) < INT8_MIN or int(b_arr.max()) > INT8_MAX:
        raise ValueError("b values must fit signed INT8 range")

    c = np.zeros((2, 2), dtype=np.int32)
    for k0 in range(0, k_total, 2):
        a_tile = np.zeros((2, 2), dtype=np.int8)
        b_tile = np.zeros((2, 2), dtype=np.int8)
        k1 = min(k0 + 2, k_total)
        width = k1 - k0
        a_tile[:, :width] = a_arr[:, k0:k1].astype(np.int8)
        b_tile[:width, :] = b_arr[k0:k1, :].astype(np.int8)
        c = systolic_tile_2x2_step(a_tile, b_tile, c)
    return c


def direct_int8_matmul_2xk_by_kx2(
    a: Sequence[Sequence[int]] | np.ndarray,
    b: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Direct INT32 reference for a 2xK by Kx2 signed INT8 matmul."""

    a_arr = np.asarray(a).astype(np.int32)
    b_arr = np.asarray(b).astype(np.int32)
    if a_arr.ndim != 2 or b_arr.ndim != 2 or a_arr.shape[0] != 2 or b_arr.shape[1] != 2:
        raise ValueError("expected shapes 2xK and Kx2")
    if a_arr.shape[1] != b_arr.shape[0]:
        raise ValueError(f"shape mismatch: {a_arr.shape} x {b_arr.shape}")
    return (a_arr @ b_arr).astype(np.int32)
