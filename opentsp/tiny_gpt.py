from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .ir import Graph, OpSpec, TensorSpec
from .models import eager_reference


ArrayMap = Dict[str, np.ndarray]


@dataclass(frozen=True)
class TinyGPTConfig:
    """Configuration for a very small GPT-style single-block decoder.

    This is intentionally tiny so it can run on CPU and map into the current
    OpenTSP single-token decoder backend. Embedding lookup and KV-cache prefill
    are done by the adapter; the compiled graph starts at the current hidden
    state and runs one autoregressive decode step.
    """

    vocab_size: int = 64
    d_model: int = 16
    d_ff: int = 32
    max_seq_len: int = 16
    cache_len: int = 4


@dataclass(frozen=True)
class TinyGPTStep:
    """Prepared single-token decode step for the OpenTSP graph runtime."""

    config: TinyGPTConfig
    prompt_token_ids: tuple[int, ...]
    graph: Graph
    values: ArrayMap
    token_embedding: np.ndarray
    position_embedding: np.ndarray


def _w(rng: np.random.Generator, shape: Tuple[int, ...], scale: float = 0.02) -> np.ndarray:
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


def _validate_tokens(token_ids: Sequence[int], config: TinyGPTConfig) -> tuple[int, ...]:
    if not token_ids:
        raise ValueError("prompt_token_ids must contain at least one token")
    if len(token_ids) > config.max_seq_len:
        raise ValueError(
            f"prompt length {len(token_ids)} exceeds max_seq_len={config.max_seq_len}"
        )
    out = tuple(int(t) for t in token_ids)
    bad = [t for t in out if t < 0 or t >= config.vocab_size]
    if bad:
        raise ValueError(f"token ids must be in [0, {config.vocab_size}); got {bad}")
    return out


def _build_graph(config: TinyGPTConfig) -> Graph:
    tensors: dict[str, TensorSpec] = {}

    def add(name: str, shape: Tuple[int, ...], is_weight: bool = False) -> None:
        tensors[name] = TensorSpec(name=name, shape=shape, dtype="float32", is_weight=is_weight)

    d_model = config.d_model
    d_ff = config.d_ff
    cache_len = config.cache_len
    vocab_size = config.vocab_size

    add("x", (1, d_model))
    add("k_cache", (cache_len, d_model))
    add("v_cache", (cache_len, d_model))

    for name in ["w_q", "w_k", "w_v", "w_o"]:
        add(name, (d_model, d_model), is_weight=True)
    add("gamma_1", (d_model,), is_weight=True)
    add("w_ff1", (d_model, d_ff), is_weight=True)
    add("w_ff2", (d_ff, d_model), is_weight=True)
    add("gamma_2", (d_model,), is_weight=True)
    add("w_vocab", (d_model, vocab_size), is_weight=True)

    for name in ["q", "k_new", "v_new", "attn_ctx", "attn_out", "resid_1", "norm_1", "ff1", "ff_act", "ff2", "resid_2", "norm_2", "logits"]:
        shape = (1, d_model)
        if name in {"ff1", "ff_act"}:
            shape = (1, d_ff)
        if name == "logits":
            shape = (1, vocab_size)
        add(name, shape)

    add("k_all", (cache_len, d_model))
    add("v_all", (cache_len, d_model))
    add("next_token", (1,))

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

    return Graph(
        name="tiny_gpt_single_block_decode",
        tensors=tensors,
        ops=ops,
        inputs=["x", "k_cache", "v_cache"],
        outputs=["next_token", "logits", "k_all", "v_all"],
    )


def _positioned_hidden(token_ids: tuple[int, ...], token_embedding: np.ndarray, position_embedding: np.ndarray) -> np.ndarray:
    positions = np.arange(len(token_ids), dtype=np.int64)
    return (token_embedding[np.asarray(token_ids, dtype=np.int64)] + position_embedding[positions]).astype(np.float32)


def build_tiny_gpt_step(
    prompt_token_ids: Sequence[int],
    config: TinyGPTConfig | None = None,
    *,
    seed: int = 123,
) -> TinyGPTStep:
    """Build a deterministic tiny GPT-style decode step.

    The prompt is converted into the current hidden state plus prefilled K/V
    caches. The graph then represents a real single-block GPT-style token step:
    QKV projection, cache append, decode attention, MLP, vocab logits, argmax.
    """

    cfg = config or TinyGPTConfig()
    tokens = _validate_tokens(prompt_token_ids, cfg)
    rng = np.random.default_rng(seed)

    token_embedding = _w(rng, (cfg.vocab_size, cfg.d_model), scale=0.35)
    position_embedding = _w(rng, (cfg.max_seq_len, cfg.d_model), scale=0.05)

    weights: ArrayMap = {
        "w_q": _w(rng, (cfg.d_model, cfg.d_model)),
        "w_k": _w(rng, (cfg.d_model, cfg.d_model)),
        "w_v": _w(rng, (cfg.d_model, cfg.d_model)),
        "w_o": _w(rng, (cfg.d_model, cfg.d_model)),
        "gamma_1": np.ones((cfg.d_model,), dtype=np.float32),
        "w_ff1": _w(rng, (cfg.d_model, cfg.d_ff)),
        "w_ff2": _w(rng, (cfg.d_ff, cfg.d_model)),
        "gamma_2": np.ones((cfg.d_model,), dtype=np.float32),
        "w_vocab": _w(rng, (cfg.d_model, cfg.vocab_size)),
    }

    hidden = _positioned_hidden(tokens, token_embedding, position_embedding)
    x = hidden[-1:, :].astype(np.float32)

    past_hidden = hidden[:-1]
    k_cache = np.zeros((cfg.cache_len, cfg.d_model), dtype=np.float32)
    v_cache = np.zeros((cfg.cache_len, cfg.d_model), dtype=np.float32)
    if past_hidden.shape[0] > 0:
        k_past = past_hidden @ weights["w_k"]
        v_past = past_hidden @ weights["w_v"]
        k_tail = k_past[-cfg.cache_len :, :]
        v_tail = v_past[-cfg.cache_len :, :]
        k_cache[-k_tail.shape[0] :, :] = k_tail
        v_cache[-v_tail.shape[0] :, :] = v_tail

    values: ArrayMap = {
        "x": x,
        "k_cache": k_cache,
        "v_cache": v_cache,
        **weights,
    }

    return TinyGPTStep(
        config=cfg,
        prompt_token_ids=tokens,
        graph=_build_graph(cfg),
        values=values,
        token_embedding=token_embedding,
        position_embedding=position_embedding,
    )


def tiny_gpt_fp32_reference(step: TinyGPTStep) -> ArrayMap:
    """Run the independent FP32 reference for the prepared decode step."""

    return eager_reference(step.values, cache_len=step.config.cache_len)
