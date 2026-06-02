from __future__ import annotations

import numpy as np

from opentsp.quant import dequantize_symmetric_int8, int8_matmul_reference, quantize_symmetric_int8
from opentsp.tiled_matmul import TiledMatmulConfig, int8_tiled_matmul


def test_quantize_roundtrip_has_bounded_error() -> None:
    x = np.array([-1.0, -0.25, 0.0, 0.25, 1.0], dtype=np.float32)
    qx = quantize_symmetric_int8(x)
    dx = dequantize_symmetric_int8(qx)

    assert qx.q.dtype == np.int8
    assert qx.scale > 0
    assert np.max(np.abs(dx - x)) <= qx.scale


def test_tiled_matmul_matches_int8_reference() -> None:
    rng = np.random.default_rng(123)
    a = rng.normal(size=(4, 8)).astype(np.float32)
    b = rng.normal(size=(8, 6)).astype(np.float32)

    qa = quantize_symmetric_int8(a)
    qb = quantize_symmetric_int8(b)

    cfg = TiledMatmulConfig(tile_m=2, tile_n=3, tile_k=4, mac_lanes=8)
    tiled = int8_tiled_matmul(qa, qb, cfg)
    ref_acc, ref_out = int8_matmul_reference(qa, qb)

    np.testing.assert_array_equal(tiled.acc_int32, ref_acc)
    np.testing.assert_allclose(tiled.out_fp32, ref_out, rtol=0, atol=0)
    assert tiled.total_cycles > 0
    assert len(tiled.events) > 0


def test_tiled_matmul_handles_edge_tiles() -> None:
    rng = np.random.default_rng(456)
    a = rng.normal(size=(5, 13)).astype(np.float32)
    b = rng.normal(size=(13, 7)).astype(np.float32)

    qa = quantize_symmetric_int8(a)
    qb = quantize_symmetric_int8(b)

    # Deliberately choose tile sizes that do not divide matrix dimensions.
    cfg = TiledMatmulConfig(tile_m=2, tile_n=4, tile_k=6, mac_lanes=12)
    tiled = int8_tiled_matmul(qa, qb, cfg)
    ref_acc, _ = int8_matmul_reference(qa, qb)

    np.testing.assert_array_equal(tiled.acc_int32, ref_acc)
    assert tiled.acc_int32.shape == (5, 7)


def test_tiled_schedule_is_deterministic() -> None:
    rng = np.random.default_rng(789)
    a = rng.normal(size=(3, 5)).astype(np.float32)
    b = rng.normal(size=(5, 4)).astype(np.float32)

    qa = quantize_symmetric_int8(a)
    qb = quantize_symmetric_int8(b)
    cfg = TiledMatmulConfig(tile_m=2, tile_n=2, tile_k=3, mac_lanes=4)

    first = int8_tiled_matmul(qa, qb, cfg)
    second = int8_tiled_matmul(qa, qb, cfg)

    assert first.total_cycles == second.total_cycles
    assert first.events == second.events
    np.testing.assert_array_equal(first.acc_int32, second.acc_int32)


def test_invalid_tiled_matmul_shape_raises() -> None:
    qa = quantize_symmetric_int8(np.ones((2, 3), dtype=np.float32))
    qb = quantize_symmetric_int8(np.ones((4, 2), dtype=np.float32))

    try:
        int8_tiled_matmul(qa, qb)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid matmul shapes")
