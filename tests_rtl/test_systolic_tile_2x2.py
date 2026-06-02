"""cocotb tests for rtl/systolic_tile_2x2.sv.

The tile computes and accumulates a 2x2 matrix product:

    C += A x B

where A and B are 2x2 signed INT8 tiles and C is a 2x2 signed INT32
accumulator tile. This is the first RTL step from a scalar MAC unit toward a
small systolic-array-style matmul primitive.
"""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer


IN_WIDTH = 8


def twos(value: int, width: int = IN_WIDTH) -> int:
    """Encode a signed integer into two's-complement bits."""
    return value & ((1 << width) - 1)


def signed(value) -> int:
    """Read a cocotb signal as a signed integer."""
    return int(value.signed_integer)


def matmul2(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    """Reference 2x2 matrix multiply."""
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def add2(x: list[list[int]], y: list[list[int]]) -> list[list[int]]:
    """Elementwise add for 2x2 matrices."""
    return [[x[r][c] + y[r][c] for c in range(2)] for r in range(2)]


def read_c(dut) -> list[list[int]]:
    """Read the 2x2 accumulator output tile."""
    return [
        [signed(dut.c00_o.value), signed(dut.c01_o.value)],
        [signed(dut.c10_o.value), signed(dut.c11_o.value)],
    ]


async def start_and_reset(dut) -> None:
    """Start the clock and reset the DUT."""
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
    assert read_c(dut) == [[0, 0], [0, 0]]


async def apply_clear(dut) -> None:
    """Clear the output accumulator tile."""
    dut.valid_i.value = 0
    dut.clear_i.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.clear_i.value = 0

    assert read_c(dut) == [[0, 0], [0, 0]]


async def apply_tile(
    dut,
    a: list[list[int]],
    b: list[list[int]],
    expected_c: list[list[int]],
) -> None:
    """Apply one valid 2x2 tile multiply-accumulate operation."""
    dut.valid_i.value = 1
    dut.clear_i.value = 0

    dut.a00_i.value = twos(a[0][0])
    dut.a01_i.value = twos(a[0][1])
    dut.a10_i.value = twos(a[1][0])
    dut.a11_i.value = twos(a[1][1])

    dut.b00_i.value = twos(b[0][0])
    dut.b01_i.value = twos(b[0][1])
    dut.b10_i.value = twos(b[1][0])
    dut.b11_i.value = twos(b[1][1])

    await RisingEdge(dut.clk)
    await Timer(1, units="ns")

    assert int(dut.valid_o.value) == 1
    assert read_c(dut) == expected_c

    dut.valid_i.value = 0


@cocotb.test()
async def reset_outputs_zero(dut):
    """After reset, accumulator tile and valid output are zero."""
    await start_and_reset(dut)


@cocotb.test()
async def computes_single_2x2_product(dut):
    """The tile computes one signed 2x2 matrix product."""
    await start_and_reset(dut)

    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    expected = matmul2(a, b)

    await apply_tile(dut, a, b, expected)


@cocotb.test()
async def accumulates_two_k_tiles(dut):
    """Multiple valid cycles accumulate partial products across K tiles."""
    await start_and_reset(dut)

    acc = [[0, 0], [0, 0]]
    tiles = [
        ([[1, -2], [3, 4]], [[5, 6], [-7, 8]]),
        ([[-3, 2], [1, -5]], [[-4, 9], [6, -2]]),
    ]

    for a, b in tiles:
        acc = add2(acc, matmul2(a, b))
        await apply_tile(dut, a, b, acc)


@cocotb.test()
async def clear_resets_accumulator_between_products(dut):
    """clear_i resets the accumulated C tile between independent products."""
    await start_and_reset(dut)

    first_a = [[10, -3], [2, 7]]
    first_b = [[-5, 4], [6, -8]]
    await apply_tile(dut, first_a, first_b, matmul2(first_a, first_b))

    await apply_clear(dut)

    second_a = [[-1, -2], [-3, -4]]
    second_b = [[8, -7], [6, -5]]
    await apply_tile(dut, second_a, second_b, matmul2(second_a, second_b))


@cocotb.test()
async def deterministic_random_tile_sequence(dut):
    """A deterministic random tile sequence matches the Python reference."""
    await start_and_reset(dut)

    rng = random.Random(2026)
    acc = [[0, 0], [0, 0]]

    for _ in range(24):
        a = [[rng.randint(-16, 15), rng.randint(-16, 15)], [rng.randint(-16, 15), rng.randint(-16, 15)]]
        b = [[rng.randint(-16, 15), rng.randint(-16, 15)], [rng.randint(-16, 15), rng.randint(-16, 15)]]
        acc = add2(acc, matmul2(a, b))
        await apply_tile(dut, a, b, acc)
