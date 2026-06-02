"""cocotb tests for rtl/mac_unit.sv.

These tests verify the first real RTL primitive in OpenTSP:
    acc_o = acc_o + signed_int8(a_i) * signed_int8(b_i)

The tests run in a software simulator through Verilator, so no FPGA board
or cloud instance is required.
"""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer


IN_WIDTH = 8
ACC_WIDTH = 32


def twos(value: int, width: int) -> int:
    """Encode a signed integer into two's-complement bits."""
    return value & ((1 << width) - 1)


def signed(value) -> int:
    """Read a cocotb signal as a signed integer."""
    return int(value.signed_integer)


async def start_and_reset(dut) -> None:
    """Start the clock and reset the DUT."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value = 0
    dut.valid_i.value = 0
    dut.clear_i.value = 0
    dut.a_i.value = 0
    dut.b_i.value = 0

    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")

    assert signed(dut.acc_o.value) == 0
    assert int(dut.valid_o.value) == 0


async def apply_mac(dut, a: int, b: int, expected_acc: int) -> None:
    """Apply one valid MAC operation and check the accumulated result."""
    dut.valid_i.value = 1
    dut.clear_i.value = 0
    dut.a_i.value = twos(a, IN_WIDTH)
    dut.b_i.value = twos(b, IN_WIDTH)

    await RisingEdge(dut.clk)
    await Timer(1, units="ns")

    assert int(dut.valid_o.value) == 1
    assert signed(dut.acc_o.value) == expected_acc

    dut.valid_i.value = 0
    dut.a_i.value = 0
    dut.b_i.value = 0


async def apply_clear(dut) -> None:
    """Clear the accumulator."""
    dut.valid_i.value = 0
    dut.clear_i.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.clear_i.value = 0

    assert signed(dut.acc_o.value) == 0


@cocotb.test()
async def reset_outputs_zero(dut):
    """After reset, accumulator and valid output are zero."""
    await start_and_reset(dut)


@cocotb.test()
async def accumulates_signed_int8_products(dut):
    """The MAC unit accumulates positive and negative signed INT8 products."""
    await start_and_reset(dut)

    expected = 0
    vectors = [
        (3, 4),       # +12
        (5, -2),      # -10
        (-7, -6),     # +42
        (-128, 2),    # -256
        (127, -1),    # -127
        (-8, 9),      # -72
    ]

    for a, b in vectors:
        expected += a * b
        await apply_mac(dut, a, b, expected)


@cocotb.test()
async def clear_resets_accumulator_between_sequences(dut):
    """clear_i resets accumulated state between two MAC sequences."""
    await start_and_reset(dut)

    expected = 0
    for a, b in [(10, 3), (-4, 7)]:
        expected += a * b
        await apply_mac(dut, a, b, expected)

    await apply_clear(dut)

    expected = 0
    for a, b in [(-5, -5), (6, -3)]:
        expected += a * b
        await apply_mac(dut, a, b, expected)


@cocotb.test()
async def deterministic_random_vector_sequence(dut):
    """A deterministic random sequence matches the Python reference model."""
    await start_and_reset(dut)

    rng = random.Random(1234)
    expected = 0

    for _ in range(32):
        a = rng.randint(-128, 127)
        b = rng.randint(-128, 127)
        expected += a * b
        await apply_mac(dut, a, b, expected)
