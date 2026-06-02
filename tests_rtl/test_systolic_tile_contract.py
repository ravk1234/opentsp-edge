"""RTL/Python contract tests for rtl/systolic_tile_2x2.sv.

These tests connect the Python tiled-matmul assumptions to the RTL tile
behavior. The same signed INT8 2x2 tiles are evaluated by the SystemVerilog
DUT and by opentsp.rtl_reference.
"""

from __future__ import annotations

import random

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

from opentsp.rtl_reference import matmul_2xk_by_kx2_via_rtl_contract, systolic_tile_2x2_step


IN_WIDTH = 8


def twos(value: int, width: int = IN_WIDTH) -> int:
    """Encode a signed integer into two's-complement bits."""
    return value & ((1 << width) - 1)


def signed(value) -> int:
    """Read a cocotb signal as a signed integer."""
    return int(value.signed_integer)


def read_c(dut) -> np.ndarray:
    """Read the 2x2 INT32 accumulator output tile."""
    return np.array(
        [
            [signed(dut.c00_o.value), signed(dut.c01_o.value)],
            [signed(dut.c10_o.value), signed(dut.c11_o.value)],
        ],
        dtype=np.int32,
    )


async def start_and_reset(dut) -> None:
    """Start clock and reset the tile."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value = 0
    dut.valid_i.value = 0
    dut.clear_i.value = 0
    for name in [
        "a00_i", "a01_i", "a10_i", "a11_i",
        "b00_i", "b01_i", "b10_i", "b11_i",
    ]:
        getattr(dut, name).value = 0

    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")

    assert int(dut.valid_o.value) == 0
    np.testing.assert_array_equal(read_c(dut), np.zeros((2, 2), dtype=np.int32))


async def apply_tile_and_check(dut, a_tile: np.ndarray, b_tile: np.ndarray, expected: np.ndarray) -> None:
    """Drive one valid tile into RTL and compare against expected C."""
    dut.valid_i.value = 1
    dut.clear_i.value = 0

    dut.a00_i.value = twos(int(a_tile[0, 0]))
    dut.a01_i.value = twos(int(a_tile[0, 1]))
    dut.a10_i.value = twos(int(a_tile[1, 0]))
    dut.a11_i.value = twos(int(a_tile[1, 1]))

    dut.b00_i.value = twos(int(b_tile[0, 0]))
    dut.b01_i.value = twos(int(b_tile[0, 1]))
    dut.b10_i.value = twos(int(b_tile[1, 0]))
    dut.b11_i.value = twos(int(b_tile[1, 1]))

    await RisingEdge(dut.clk)
    await Timer(1, units="ns")

    assert int(dut.valid_o.value) == 1
    np.testing.assert_array_equal(read_c(dut), expected)
    dut.valid_i.value = 0


def k_tile_pair(a: np.ndarray, b: np.ndarray, k0: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract or zero-pad one K=2 tile pair from A[2,K] and B[K,2]."""
    a_tile = np.zeros((2, 2), dtype=np.int8)
    b_tile = np.zeros((2, 2), dtype=np.int8)
    k1 = min(k0 + 2, a.shape[1])
    width = k1 - k0
    a_tile[:, :width] = a[:, k0:k1]
    b_tile[:width, :] = b[k0:k1, :]
    return a_tile, b_tile


@cocotb.test()
async def rtl_matches_python_contract_single_step(dut):
    """One RTL valid cycle matches the Python 2x2 tile contract."""
    await start_and_reset(dut)

    a = np.array([[1, -2], [3, 4]], dtype=np.int8)
    b = np.array([[5, 6], [-7, 8]], dtype=np.int8)
    expected = systolic_tile_2x2_step(a, b)
    await apply_tile_and_check(dut, a, b, expected)


@cocotb.test()
async def rtl_matches_python_contract_even_k_matmul(dut):
    """RTL tile sequence matches the Python contract for a 2x4 by 4x2 matmul."""
    await start_and_reset(dut)

    a = np.array([[1, -2, 3, 4], [5, 6, -7, 8]], dtype=np.int8)
    b = np.array([[2, -1], [3, 4], [-5, 6], [7, -8]], dtype=np.int8)
    expected_final = matmul_2xk_by_kx2_via_rtl_contract(a, b)

    acc = np.zeros((2, 2), dtype=np.int32)
    for k0 in range(0, a.shape[1], 2):
        a_tile, b_tile = k_tile_pair(a, b, k0)
        acc = systolic_tile_2x2_step(a_tile, b_tile, acc)
        await apply_tile_and_check(dut, a_tile, b_tile, acc)

    np.testing.assert_array_equal(read_c(dut), expected_final)


@cocotb.test()
async def rtl_matches_python_contract_odd_k_with_padding(dut):
    """A final zero-padded K tile matches the odd-K Python contract."""
    await start_and_reset(dut)

    a = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.int8)
    b = np.array([[7, -8], [9, 10], [-11, 12]], dtype=np.int8)
    expected_final = matmul_2xk_by_kx2_via_rtl_contract(a, b)

    acc = np.zeros((2, 2), dtype=np.int32)
    for k0 in range(0, a.shape[1], 2):
        a_tile, b_tile = k_tile_pair(a, b, k0)
        acc = systolic_tile_2x2_step(a_tile, b_tile, acc)
        await apply_tile_and_check(dut, a_tile, b_tile, acc)

    np.testing.assert_array_equal(read_c(dut), expected_final)


@cocotb.test()
async def rtl_matches_python_contract_random_sequence(dut):
    """Deterministic random K-tile sequence matches Python contract step-by-step."""
    await start_and_reset(dut)

    rng = random.Random(4112)
    acc = np.zeros((2, 2), dtype=np.int32)

    for _ in range(20):
        a_tile = np.array([[rng.randint(-32, 31), rng.randint(-32, 31)], [rng.randint(-32, 31), rng.randint(-32, 31)]], dtype=np.int8)
        b_tile = np.array([[rng.randint(-32, 31), rng.randint(-32, 31)], [rng.randint(-32, 31), rng.randint(-32, 31)]], dtype=np.int8)
        acc = systolic_tile_2x2_step(a_tile, b_tile, acc)
        await apply_tile_and_check(dut, a_tile, b_tile, acc)
