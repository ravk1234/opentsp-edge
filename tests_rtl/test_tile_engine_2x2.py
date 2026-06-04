"""cocotb tests for rtl/tile_engine_2x2.sv.

This verifies the first combined RTL datapath: the instruction controller decodes
an OpenTSP MAC_TILE instruction and drives the 2x2 systolic tile's valid input.
LOAD/STORE/ATTENTION are still control-only signals at this milestone.
"""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

OP_NOP = 0
OP_LOAD_A = 1
OP_LOAD_B = 2
OP_MAC_TILE = 3
OP_STORE_C = 4
OP_ATTENTION = 5
OP_BASELINE = 6


def twos(value: int, width: int = 8) -> int:
    return value & ((1 << width) - 1)


def signed(value) -> int:
    return int(value.signed_integer)


def matmul2(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def scale2(x: list[list[int]], factor: int) -> list[list[int]]:
    return [[factor * x[0][0], factor * x[0][1]], [factor * x[1][0], factor * x[1][1]]]


def read_c(dut) -> list[list[int]]:
    return [[signed(dut.c00_o.value), signed(dut.c01_o.value)], [signed(dut.c10_o.value), signed(dut.c11_o.value)]]


async def init_dut(dut) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.host_we_i.value = 0
    dut.host_addr_i.value = 0
    dut.host_opcode_i.value = 0
    dut.host_cycles_i.value = 0
    dut.start_i.value = 0
    dut.program_len_i.value = 0
    dut.clear_tile_i.value = 0
    set_tiles(dut, [[0, 0], [0, 0]], [[0, 0], [0, 0]])
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")


def set_tiles(dut, a: list[list[int]], b: list[list[int]]) -> None:
    dut.a00_i.value = twos(a[0][0])
    dut.a01_i.value = twos(a[0][1])
    dut.a10_i.value = twos(a[1][0])
    dut.a11_i.value = twos(a[1][1])
    dut.b00_i.value = twos(b[0][0])
    dut.b01_i.value = twos(b[0][1])
    dut.b10_i.value = twos(b[1][0])
    dut.b11_i.value = twos(b[1][1])


async def write_instruction(dut, addr: int, opcode: int, cycles: int) -> None:
    dut.host_addr_i.value = addr
    dut.host_opcode_i.value = opcode
    dut.host_cycles_i.value = cycles
    dut.host_we_i.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.host_we_i.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")


async def start_program(dut, length: int) -> None:
    dut.program_len_i.value = length
    dut.start_i.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.start_i.value = 0


async def wait_done(dut, max_cycles: int = 1000) -> None:
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        if int(dut.done_o.value) == 1:
            return
    raise AssertionError("Timed out waiting for tile engine done_o")

async def wait_idle(dut, max_cycles: int = 100) -> None:
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        if int(dut.busy_o.value) == 0:
            return
    raise AssertionError("Timed out waiting for tile engine busy_o to go low")    


@cocotb.test()
async def reset_state_is_idle_and_tile_zero(dut):
    """After reset, controller is idle and tile output is zero."""
    await init_dut(dut)
    assert int(dut.busy_o.value) == 0
    assert int(dut.instr_valid_o.value) == 0
    assert int(dut.tile_valid_o.value) == 0
    assert read_c(dut) == [[0, 0], [0, 0]]


@cocotb.test()
async def controller_mac_instruction_drives_tile_once(dut):
    """A one-cycle MAC_TILE instruction drives exactly one 2x2 tile product."""
    await init_dut(dut)
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    set_tiles(dut, a, b)

    await write_instruction(dut, 0, OP_LOAD_A, 1)
    await write_instruction(dut, 1, OP_LOAD_B, 1)
    await write_instruction(dut, 2, OP_MAC_TILE, 1)
    await write_instruction(dut, 3, OP_STORE_C, 1)

    await start_program(dut, 4)
    await wait_done(dut)

    assert read_c(dut) == matmul2(a, b)
    assert int(dut.done_o.value) == 1


@cocotb.test()
async def multi_cycle_mac_accumulates_repeated_tile(dut):
    """A MAC_TILE instruction held for N cycles accumulates N identical tile products."""
    await init_dut(dut)
    a = [[-2, 3], [4, -5]]
    b = [[6, -7], [8, 9]]
    set_tiles(dut, a, b)

    await write_instruction(dut, 0, OP_MAC_TILE, 3)
    await write_instruction(dut, 1, OP_STORE_C, 1)

    await start_program(dut, 2)
    await wait_done(dut)

    assert read_c(dut) == scale2(matmul2(a, b), 3)


@cocotb.test()
async def non_mac_instructions_do_not_change_tile(dut):
    """LOAD/STORE/BASELINE/ATTENTION decoded signals do not trigger tile accumulation."""
    await init_dut(dut)
    a = [[10, 11], [12, 13]]
    b = [[2, 3], [4, 5]]
    set_tiles(dut, a, b)

    program = [OP_LOAD_A, OP_LOAD_B, OP_STORE_C, OP_ATTENTION, OP_BASELINE]
    for i, op in enumerate(program):
        await write_instruction(dut, i, op, 1)

    await start_program(dut, len(program))
    await wait_done(dut)

    assert read_c(dut) == [[0, 0], [0, 0]]


@cocotb.test()
async def clear_tile_resets_between_programs(dut):
    """clear_tile_i clears accumulated C so a second program starts fresh."""
    await init_dut(dut)
    a1 = [[1, 0], [0, 1]]
    b1 = [[9, 8], [7, 6]]
    set_tiles(dut, a1, b1)
    await write_instruction(dut, 0, OP_MAC_TILE, 1)
    await start_program(dut, 1)
    await wait_done(dut)
    assert read_c(dut) == matmul2(a1, b1)

    dut.clear_tile_i.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.clear_tile_i.value = 0
    assert read_c(dut) == [[0, 0], [0, 0]]

    a2 = [[-1, 2], [3, -4]]
    b2 = [[5, -6], [7, 8]]
    set_tiles(dut, a2, b2)
    await start_program(dut, 1)
    await wait_done(dut)
    assert read_c(dut) == matmul2(a2, b2)


@cocotb.test()
async def deterministic_random_program_matches_python_reference(dut):
    """Random signed tiles accumulated by repeated MAC instructions match Python."""
    await init_dut(dut)
    rng = random.Random(20260514)

    for i in range(6):
        await write_instruction(dut, i, OP_MAC_TILE, 1)

    expected = [[0, 0], [0, 0]]
    await start_program(dut, 6)

    for _ in range(6):
        a = [[rng.randint(-8, 8), rng.randint(-8, 8)], [rng.randint(-8, 8), rng.randint(-8, 8)]]
        b = [[rng.randint(-8, 8), rng.randint(-8, 8)], [rng.randint(-8, 8), rng.randint(-8, 8)]]
        set_tiles(dut, a, b)
        product = matmul2(a, b)
        expected[0][0] += product[0][0]
        expected[0][1] += product[0][1]
        expected[1][0] += product[1][0]
        expected[1][1] += product[1][1]
        # Let the current MAC_TILE cycle consume this tile.
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")

    await wait_idle(dut)
    assert read_c(dut) == expected
