from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .compiler import CompiledProgram, ScheduledOp


ArrayMap = Dict[str, np.ndarray]


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(shifted)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _append_cache(cache: np.ndarray, new_item: np.ndarray, max_len: int) -> np.ndarray:
    if new_item.ndim != 2 or new_item.shape[0] != 1:
        raise ValueError("new cache item must have shape [1, d]")
    out = np.concatenate([cache, new_item], axis=0)
    if out.shape[0] > max_len:
        out = out[-max_len:, :]
    return out


def _attention_decode(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
    d = q.shape[-1]
    scores = (q @ k_cache.T) / np.sqrt(float(d))
    probs = _softmax(scores, axis=-1)
    return probs @ v_cache


def _execute_op(op: ScheduledOp, values: ArrayMap) -> None:
    kind = op.kind

    if kind == "matmul":
        a = values[op.inputs[0]]
        b = values[op.inputs[1]]
        values[op.outputs[0]] = (a @ b).astype(np.float32)
        return

    if kind == "add":
        values[op.outputs[0]] = (values[op.inputs[0]] + values[op.inputs[1]]).astype(np.float32)
        return

    if kind == "silu":
        values[op.outputs[0]] = _silu(values[op.inputs[0]]).astype(np.float32)
        return

    if kind == "rmsnorm":
        eps = float(op.attrs.get("eps", 1e-6))
        values[op.outputs[0]] = _rmsnorm(values[op.inputs[0]], values[op.inputs[1]], eps).astype(np.float32)
        return

    if kind == "append_cache":
        max_len = int(op.attrs["max_len"])
        values[op.outputs[0]] = _append_cache(values[op.inputs[0]], values[op.inputs[1]], max_len).astype(np.float32)
        return

    if kind == "attention_decode":
        values[op.outputs[0]] = _attention_decode(
            values[op.inputs[0]], values[op.inputs[1]], values[op.inputs[2]]
        ).astype(np.float32)
        return

    if kind == "softmax":
        axis = int(op.attrs.get("axis", -1))
        values[op.outputs[0]] = _softmax(values[op.inputs[0]], axis=axis).astype(np.float32)
        return

    if kind == "argmax":
        axis = int(op.attrs.get("axis", -1))
        values[op.outputs[0]] = np.asarray(np.argmax(values[op.inputs[0]], axis=axis), dtype=np.int32)
        return

    raise ValueError(f"Unsupported op kind {kind}")


def run_schedule(program: CompiledProgram, inputs_and_weights: Mapping[str, np.ndarray]) -> ArrayMap:
    """Run a compiled program exactly in schedule order.

    The returned map contains all intermediate tensors as well as graph outputs.
    """

    values: ArrayMap = {k: np.asarray(v).copy() for k, v in inputs_and_weights.items()}
    for op in program.schedule:
        _execute_op(op, values)
    return values
