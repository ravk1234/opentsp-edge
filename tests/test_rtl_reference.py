from __future__ import annotations

import numpy as np
import pytest

from opentsp.quant import quantize_symmetric_int8
from opentsp.rtl_reference import (
    SystolicTile2x2Contract,
    direct_int8_matmul_2xk_by_kx2,
    matmul_2xk_by_kx2_via_rtl_contract,
    systolic_tile_2x2_sequence,
    systolic_tile_2x2_step,
)
from opentsp.tiled_matmul import TiledMatmulConfig, int8_tiled_matmul


def test_contract_metadata_matches_rtl_tile_shape() -> None:
    contract = SystolicTile2x2Contract()
    assert contract.tile_m == 2
    assert contract.tile_n == 2
    assert contract.tile_k == 2
    assert contract.input_dtype == "int8"
    assert contract.accumulator_dtype == "int32"


def test_single_step_matches_direct_2x2_matmul() -> None:
    a = np.array([[1, -2], [3, 4]], dtype=np.int8)
    b = np.array([[5, 6], [-7, 8]], dtype=np.int8)
    expected = a.astype(np.int32) @ b.astype(np.int32)
    np.testing.assert_array_equal(systolic_tile_2x2_step(a, b), expected)


def test_sequence_accumulates_multiple_k_tiles() -> None:
    tiles = [
        (
            np.array([[1, -2], [3, 4]], dtype=np.int8),
            np.array([[5, 6], [-7, 8]], dtype=np.int8),
        ),
        (
            np.array([[-3, 2], [1, -5]], dtype=np.int8),
            np.array([[-4, 9], [6, -2]], dtype=np.int8),
        ),
    ]
    expected = sum((a.astype(np.int32) @ b.astype(np.int32) for a, b in tiles), np.zeros((2, 2), dtype=np.int32))
    np.testing.assert_array_equal(systolic_tile_2x2_sequence(tiles), expected)


def test_2xk_by_kx2_contract_matches_direct_reference_even_k() -> None:
    rng = np.random.default_rng(2026)
    a = rng.integers(-32, 32, size=(2, 6), dtype=np.int16).astype(np.int8)
    b = rng.integers(-32, 32, size=(6, 2), dtype=np.int16).astype(np.int8)
    np.testing.assert_array_equal(matmul_2xk_by_kx2_via_rtl_contract(a, b), direct_int8_matmul_2xk_by_kx2(a, b))


def test_2xk_by_kx2_contract_pads_odd_k() -> None:
    rng = np.random.default_rng(7)
    a = rng.integers(-16, 16, size=(2, 5), dtype=np.int16).astype(np.int8)
    b = rng.integers(-16, 16, size=(5, 2), dtype=np.int16).astype(np.int8)
    np.testing.assert_array_equal(matmul_2xk_by_kx2_via_rtl_contract(a, b), direct_int8_matmul_2xk_by_kx2(a, b))


def test_contract_matches_tiled_matmul_for_one_2x2_output_tile() -> None:
    rng = np.random.default_rng(99)
    a_fp = rng.normal(size=(2, 8)).astype(np.float32)
    b_fp = rng.normal(size=(8, 2)).astype(np.float32)
    qa = quantize_symmetric_int8(a_fp)
    qb = quantize_symmetric_int8(b_fp)

    tiled = int8_tiled_matmul(qa, qb, TiledMatmulConfig(tile_m=2, tile_n=2, tile_k=2, mac_lanes=4))
    contract_acc = matmul_2xk_by_kx2_via_rtl_contract(qa.q, qb.q)

    np.testing.assert_array_equal(tiled.acc_int32, contract_acc)


def test_rejects_out_of_range_int8_values() -> None:
    with pytest.raises(ValueError):
        systolic_tile_2x2_step([[128, 0], [0, 0]], [[1, 0], [0, 1]])
