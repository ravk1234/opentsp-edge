from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .ir import Graph, OpSpec, TensorSpec


ArrayMap = Dict[str, np.ndarray]


def _w(rng: np.random.Generator, shape: Tuple[int, ...], scale: float = 0.02) -> np.ndarray:
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


def build_tiny_voice_decoder_graph(
    d_model: int = 16,
    d_ff: int = 32,
    vocab_size: int = 64,
    cache_len: int = 4,
    seed: int = 7,
) -> Tuple[Graph, ArrayMap]:
    """Build a tiny single-token decoder step.

    This is a toy block that looks like a simplified autoregressive voice-token decoder:

    x -> Q/K/V projection -> append KV cache -> attention -> output projection
      -> residual + RMSNorm -> FFN -> RMSNorm -> vocab logits -> argmax token

    It is intentionally tiny so it can run locally and eventually map to small RTL blocks.
    """

    rng = np.random.default_rng(seed)
    tensors = {}

    def add(name: str, shape: Tuple[int, ...], is_weight: bool = False) -> None:
        tensors[name] = TensorSpec(name=name, shape=shape, dtype="float32", is_weight=is_weight)

    # Runtime inputs.
    add("x", (1, d_model))
    add("k_cache", (cache_len, d_model))
    add("v_cache", (cache_len, d_model))

    # Weights.
    for name in ["w_q", "w_k", "w_v", "w_o"]:
        add(name, (d_model, d_model), is_weight=True)
    add("gamma_1", (d_model,), is_weight=True)
    add("w_ff1", (d_model, d_ff), is_weight=True)
    add("w_ff2", (d_ff, d_model), is_weight=True)
    add("gamma_2", (d_model,), is_weight=True)
    add("w_vocab", (d_model, vocab_size), is_weight=True)

    # Intermediates.
    for name in ["q", "k_new", "v_new", "attn_ctx", "attn_out", "resid_1", "norm_1", "ff1", "ff_act", "ff2", "resid_2", "norm_2", "logits"]:
        shape = (1, d_model)
        if name == "ff1" or name == "ff_act":
            shape = (1, d_ff)
        if name == "logits":
            shape = (1, vocab_size)
        add(name, shape)
    add("k_all", (cache_len, d_model))
    add("v_all", (cache_len, d_model))
    add("next_token", (1,), is_weight=False)

    ops = [
        OpSpec("q_proj", "matmul", ["x", "w_q"], ["q"]),
        OpSpec("k_proj", "matmul", ["x", "w_k"], ["k_new"]),
        OpSpec("v_proj", "matmul", ["x", "w_v"], ["v_new"]),
        OpSpec("append_k", "append_cache", ["k_cache", "k_new"], ["k_all"], {"max_len": cache_len}),
        OpSpec("append_v", "append_cache", ["v_cache", "v_new"], ["v_all"], {"max_len": cache_len}),
        OpSpec("attn_decode", "attention_decode", ["q", "k_all", "v_all"], ["attn_ctx"]),
        OpSpec("o_proj", "matmul", ["attn_ctx", "w_o"], ["attn_out"]),
        OpSpec("residual_attn", "add", ["x", "attn_out"], ["resid_1"]),
        OpSpec("rmsnorm_1", "rmsnorm", ["resid_1", "gamma_1"], ["norm_1"], {"eps": 1e-6}),
        OpSpec("ffn_up", "matmul", ["norm_1", "w_ff1"], ["ff1"]),
        OpSpec("ffn_silu", "silu", ["ff1"], ["ff_act"]),
        OpSpec("ffn_down", "matmul", ["ff_act", "w_ff2"], ["ff2"]),
        OpSpec("residual_ffn", "add", ["norm_1", "ff2"], ["resid_2"]),
        OpSpec("rmsnorm_2", "rmsnorm", ["resid_2", "gamma_2"], ["norm_2"], {"eps": 1e-6}),
        OpSpec("vocab_logits", "matmul", ["norm_2", "w_vocab"], ["logits"]),
        OpSpec("next_token", "argmax", ["logits"], ["next_token"], {"axis": -1}),
    ]

    graph = Graph(
        name="tiny_voice_decoder_step",
        tensors=tensors,
        ops=ops,
        inputs=["x", "k_cache", "v_cache"],
        outputs=["next_token", "logits", "k_all", "v_all"],
    )

    values: ArrayMap = {
        "x": _w(rng, (1, d_model), scale=0.5),
        "k_cache": _w(rng, (cache_len, d_model), scale=0.3),
        "v_cache": _w(rng, (cache_len, d_model), scale=0.3),
        "w_q": _w(rng, (d_model, d_model)),
        "w_k": _w(rng, (d_model, d_model)),
        "w_v": _w(rng, (d_model, d_model)),
        "w_o": _w(rng, (d_model, d_model)),
        "gamma_1": np.ones((d_model,), dtype=np.float32),
        "w_ff1": _w(rng, (d_model, d_ff)),
        "w_ff2": _w(rng, (d_ff, d_model)),
        "gamma_2": np.ones((d_model,), dtype=np.float32),
        "w_vocab": _w(rng, (d_model, vocab_size)),
    }

    return graph, values


def eager_reference(values: ArrayMap, cache_len: int = 4) -> ArrayMap:
    """Independent eager NumPy implementation for correctness checks."""

    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * gamma

    def softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x, axis=-1, keepdims=True)
        ex = np.exp(z)
        return ex / np.sum(ex, axis=-1, keepdims=True)

    x = values["x"]
    q = x @ values["w_q"]
    k_new = x @ values["w_k"]
    v_new = x @ values["w_v"]

    k_all = np.concatenate([values["k_cache"], k_new], axis=0)[-cache_len:, :]
    v_all = np.concatenate([values["v_cache"], v_new], axis=0)[-cache_len:, :]

    scores = (q @ k_all.T) / np.sqrt(float(q.shape[-1]))
    probs = softmax(scores)
    attn_ctx = probs @ v_all
    attn_out = attn_ctx @ values["w_o"]

    resid_1 = x + attn_out
    norm_1 = rmsnorm(resid_1, values["gamma_1"])
    ff1 = norm_1 @ values["w_ff1"]
    ff_act = silu(ff1)
    ff2 = ff_act @ values["w_ff2"]
    resid_2 = norm_1 + ff2
    norm_2 = rmsnorm(resid_2, values["gamma_2"])
    logits = norm_2 @ values["w_vocab"]
    next_token = np.asarray(np.argmax(logits, axis=-1), dtype=np.int32)

    return {
        "q": q.astype(np.float32),
        "k_new": k_new.astype(np.float32),
        "v_new": v_new.astype(np.float32),
        "k_all": k_all.astype(np.float32),
        "v_all": v_all.astype(np.float32),
        "attn_ctx": attn_ctx.astype(np.float32),
        "attn_out": attn_out.astype(np.float32),
        "resid_1": resid_1.astype(np.float32),
        "norm_1": norm_1.astype(np.float32),
        "ff1": ff1.astype(np.float32),
        "ff_act": ff_act.astype(np.float32),
        "ff2": ff2.astype(np.float32),
        "resid_2": resid_2.astype(np.float32),
        "norm_2": norm_2.astype(np.float32),
        "logits": logits.astype(np.float32),
        "next_token": next_token,
    }
