from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class QuantizedTensor:
    """Simple per-tensor symmetric INT8 quantization container.

    We deliberately keep this small for the local MVP:
    - symmetric signed INT8 range: [-127, 127]
    - zero point is always 0
    - one scale per tensor
    """

    q: np.ndarray
    scale: float
    zero_point: int = 0
    dtype: str = "int8"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.q.shape


def symmetric_int8_scale(x: np.ndarray, eps: float = 1e-12) -> float:
    """Return one symmetric INT8 scale for a float tensor."""
    x = np.asarray(x, dtype=np.float32)
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    return max(max_abs / 127.0, eps)


def quantize_symmetric_int8(x: np.ndarray, scale: Optional[float] = None) -> QuantizedTensor:
    """Quantize float32 values into signed INT8 using symmetric per-tensor scale."""
    x = np.asarray(x, dtype=np.float32)
    used_scale = symmetric_int8_scale(x) if scale is None else float(scale)
    if used_scale <= 0:
        raise ValueError("scale must be positive")

    q = np.round(x / used_scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return QuantizedTensor(q=q, scale=used_scale)


def dequantize_symmetric_int8(t: QuantizedTensor) -> np.ndarray:
    """Convert a QuantizedTensor back to float32."""
    if t.zero_point != 0:
        raise ValueError("Only symmetric zero_point=0 quantization is supported")
    return t.q.astype(np.float32) * np.float32(t.scale)


def int8_matmul_reference(a: QuantizedTensor, b: QuantizedTensor) -> Tuple[np.ndarray, np.ndarray]:
    """Reference INT8 matmul.

    Returns:
        acc_int32: raw integer accumulator output
        out_fp32: dequantized output using a.scale * b.scale
    """
    if a.q.ndim != 2 or b.q.ndim != 2:
        raise ValueError("int8_matmul_reference expects 2D matrices")
    if a.q.shape[1] != b.q.shape[0]:
        raise ValueError(f"matmul shape mismatch: {a.q.shape} x {b.q.shape}")

    acc = a.q.astype(np.int32) @ b.q.astype(np.int32)
    out = acc.astype(np.float32) * np.float32(a.scale * b.scale)
    return acc, out
