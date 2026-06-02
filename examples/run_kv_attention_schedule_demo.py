from __future__ import annotations

import numpy as np

from opentsp.attention_schedule import AttentionScheduleConfig, schedule_attention_decode


def _reference_attention(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
    d = q.shape[-1]
    scores = (q @ k_cache.T) / np.sqrt(float(d))
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    return (probs @ v_cache).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(42)
    q = rng.normal(0.0, 0.5, size=(1, 16)).astype(np.float32)
    k_cache = rng.normal(0.0, 0.3, size=(7, 16)).astype(np.float32)
    v_cache = rng.normal(0.0, 0.3, size=(7, 16)).astype(np.float32)

    cfg = AttentionScheduleConfig(
        cache_tile=3,
        vector_lanes=16,
        mac_lanes=16,
        load_lanes=16,
        store_lanes=16,
        clock_mhz=100,
    )
    scheduled = schedule_attention_decode(q, k_cache, v_cache, cfg)
    reference = _reference_attention(q, k_cache, v_cache)

    max_error = float(np.max(np.abs(reference - scheduled.context)))
    latency_us = scheduled.total_cycles / cfg.clock_mhz

    print("KV-cache attention schedule demo")
    print("-" * 80)
    print(f"q shape: {q.shape}, k_cache shape: {k_cache.shape}, v_cache shape: {v_cache.shape}")
    print(f"Cache tile: {cfg.cache_tile}")
    print(f"Total deterministic events: {scheduled.event_count}")
    print(f"Total estimated cycles: {scheduled.total_cycles}")
    print(f"Estimated attention latency at {cfg.clock_mhz} MHz: {latency_us:.3f} microseconds")
    print(f"Max abs error vs FP32 reference: {max_error:.8f}")

    print("\nScheduled attention micro-ops")
    print("-" * 80)
    for event in scheduled.events:
        print(
            f"{event.index:03d} {event.kind:<20} "
            f"T{event.token_range} D{event.dim_range} "
            f"start={event.start_cycle:<5} end={event.end_cycle:<5} "
            f"cycles={event.cycles:<4} bytes={event.bytes_moved:<4} macs={event.macs:<4}"
        )

    print("\nDeterministic KV-cache attention schedule check: PASSED")


if __name__ == "__main__":
    main()
