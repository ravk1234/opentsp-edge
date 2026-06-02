from __future__ import annotations

import numpy as np

from opentsp.quant import int8_matmul_reference, quantize_symmetric_int8
from opentsp.tiled_matmul import TiledMatmulConfig, int8_tiled_matmul


def main() -> None:
    rng = np.random.default_rng(7)

    # Small matrix sizes on purpose: easy to inspect, but not aligned perfectly
    # with tile sizes, so edge tiles are also tested.
    a = rng.normal(0.0, 0.7, size=(5, 13)).astype(np.float32)
    b = rng.normal(0.0, 0.5, size=(13, 9)).astype(np.float32)

    qa = quantize_symmetric_int8(a)
    qb = quantize_symmetric_int8(b)

    cfg = TiledMatmulConfig(
        tile_m=2,
        tile_n=4,
        tile_k=8,
        mac_lanes=16,
        load_lanes=16,
        store_lanes=8,
        clock_mhz=100,
    )

    tiled = int8_tiled_matmul(qa, qb, cfg)
    ref_acc, ref_out = int8_matmul_reference(qa, qb)
    fp32_ref = a @ b

    np.testing.assert_array_equal(tiled.acc_int32, ref_acc)
    np.testing.assert_allclose(tiled.out_fp32, ref_out, rtol=0, atol=0)

    print("INT8 tiled matmul demo")
    print("-" * 80)
    print(f"A shape: {a.shape}, B shape: {b.shape}")
    print(f"A scale: {qa.scale:.8f}, B scale: {qb.scale:.8f}")
    print(f"Tile config: M={cfg.tile_m}, N={cfg.tile_n}, K={cfg.tile_k}")
    print(f"Total deterministic events: {len(tiled.events)}")
    print(f"Total estimated cycles: {tiled.total_cycles}")
    print(f"Estimated latency at {cfg.clock_mhz} MHz: {tiled.total_cycles / cfg.clock_mhz:.3f} microseconds")
    print(f"Max abs error vs FP32 matmul: {np.max(np.abs(tiled.out_fp32 - fp32_ref)):.6f}")

    print("\nFirst 16 scheduled micro-ops")
    print("-" * 80)
    for ev in tiled.events[:16]:
        print(
            f"{ev.index:03d} {ev.kind:<14} "
            f"M{ev.m_range} N{ev.n_range} K{ev.k_range} "
            f"start={ev.start_cycle:<5} end={ev.end_cycle:<5} cycles={ev.cycles:<4} "
            f"bytes={ev.bytes_moved:<4} macs={ev.macs:<4}"
        )

    print("\nDeterministic tiled INT8 check: PASSED")


if __name__ == "__main__":
    main()
