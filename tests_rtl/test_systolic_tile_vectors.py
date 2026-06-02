"""Run rtl/systolic_tile_2x2.sv against generated Python test vectors."""

from __future__ import annotations

from pathlib import Path
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# tests_rtl is one level below the repo root. Add the repo root so cocotb can
# import the normal OpenTSP Python modules while running from tests_rtl/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentsp.rtl_test_vectors import generate_systolic_tile_vectors  # noqa: E402


def twos(value: int, width: int = 8) -> int:
    return value & ((1 << width) - 1)


def signed(signal) -> int:
    return int(signal.value.signed_integer)


async def reset_dut(dut) -> None:
    dut.rst_n.value = 0
    dut.valid_i.value = 0
    dut.clear_i.value = 0
    for name in [
        "a00_i",
        "a01_i",
        "a10_i",
        "a11_i",
        "b00_i",
        "b01_i",
        "b10_i",
        "b11_i",
    ]:
        getattr(dut, name).value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")


def drive_tile(dut, a: list[list[int]], b: list[list[int]]) -> None:
    dut.a00_i.value = twos(a[0][0])
    dut.a01_i.value = twos(a[0][1])
    dut.a10_i.value = twos(a[1][0])
    dut.a11_i.value = twos(a[1][1])
    dut.b00_i.value = twos(b[0][0])
    dut.b01_i.value = twos(b[0][1])
    dut.b10_i.value = twos(b[1][0])
    dut.b11_i.value = twos(b[1][1])


async def clear_accumulator(dut) -> None:
    dut.valid_i.value = 0
    dut.clear_i.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.clear_i.value = 0


def read_c(dut) -> list[list[int]]:
    return [[signed(dut.c00_o), signed(dut.c01_o)], [signed(dut.c10_o), signed(dut.c11_o)]]


@cocotb.test()
async def generated_vectors_match_rtl(dut):
    """Generated Python vectors match the RTL tile exactly."""

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    vectors = generate_systolic_tile_vectors(seed=2026, random_count=8)
    assert len(vectors) >= 8

    for vector in vectors:
        await clear_accumulator(dut)
        for a_tile, b_tile in zip(vector.a_tiles, vector.b_tiles):
            drive_tile(dut, a_tile, b_tile)
            dut.valid_i.value = 1
            await RisingEdge(dut.clk)
            await Timer(1, units="ns")
        dut.valid_i.value = 0
        assert read_c(dut) == vector.expected_c, vector.name
