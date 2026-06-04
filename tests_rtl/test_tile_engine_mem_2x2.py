"""cocotb tests for rtl/tile_engine_mem_2x2.sv.

This verifies the first data-routing step: LOAD_A and LOAD_B fetch 2x2 tiles
from tiny register-file memories, MAC_TILE drives the 2x2 compute tile, and
STORE_C captures the accumulated C tile.
"""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

OP_NOP = 0
OP_LOAD_A = 1
OP_LOAD_B = 2
OP_MAC_TILE = 3
OP_STORE_C = 4
OP_ATTENTION = 5
OP_BASELINE = 6

BANK_A = 0
BANK_B = 1


def matmul2(a, b):
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def add2(x, y):
    return [[x[0][0] + y[0][0], x[0][1] + y[0][1]], [x[1][0] + y[1][0], x[1][1] + y[1][1]]]


def read_current_a(dut):
    return [[int(dut.a00_o.value.signed_integer), int(dut.a01_o.value.signed_integer)], [int(dut.a10_o.value.signed_integer), int(dut.a11_o.value.signed_integer)]]


def read_current_b(dut):
    return [[int(dut.b00_o.value.signed_integer), int(dut.b01_o.value.signed_integer)], [int(dut.b10_o.value.signed_integer), int(dut.b11_o.value.signed_integer)]]


def read_c(dut):
    return [[int(dut.c00_o.value.signed_integer), int(dut.c01_o.value.signed_integer)], [int(dut.c10_o.value.signed_integer), int(dut.c11_o.value.signed_integer)]]


def read_stored_c(dut):
    return [
        [int(dut.stored_c00_o.value.signed_integer), int(dut.stored_c01_o.value.signed_integer)],
        [int(dut.stored_c10_o.value.signed_integer), int(dut.stored_c11_o.value.signed_integer)],
    ]


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


async def reset(dut):
    dut.rst_n.value = 0
    dut.host_we_i.value = 0
    dut.host_addr_i.value = 0
    dut.host_opcode_i.value = 0
    dut.host_cycles_i.value = 0
    dut.data_we_i.value = 0
    dut.data_bank_i.value = 0
    dut.data_addr_i.value = 0
    dut.data00_i.value = 0
    dut.data01_i.value = 0
    dut.data10_i.value = 0
    dut.data11_i.value = 0
    dut.start_i.value = 0
    dut.program_len_i.value = 0
    dut.clear_tile_i.value = 0
    await tick(dut, 2)
    dut.rst_n.value = 1
    await tick(dut, 2)


async def write_instr(dut, addr, opcode, cycles=1):
    dut.host_addr_i.value = addr
    dut.host_opcode_i.value = opcode
    dut.host_cycles_i.value = cycles
    dut.host_we_i.value = 1
    await tick(dut)
    dut.host_we_i.value = 0
    await tick(dut)


async def write_tile(dut, bank, addr, tile):
    dut.data_bank_i.value = bank
    dut.data_addr_i.value = addr
    dut.data00_i.value = tile[0][0]
    dut.data01_i.value = tile[0][1]
    dut.data10_i.value = tile[1][0]
    dut.data11_i.value = tile[1][1]
    dut.data_we_i.value = 1
    await tick(dut)
    dut.data_we_i.value = 0
    await tick(dut)


async def start_program(dut, program_len):
    dut.program_len_i.value = program_len
    dut.start_i.value = 1
    await tick(dut)
    dut.start_i.value = 0
    await tick(dut)


async def wait_idle(dut, max_cycles=200):
    for _ in range(max_cycles):
        if int(dut.busy_o.value) == 0:
            return
        await tick(dut)
    raise AssertionError("Timed out waiting for tile engine memory wrapper to go idle")


@cocotb.test()
async def reset_outputs_are_zero(dut):
    """After reset, loaded tiles, C tile, and stored output are zero."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    assert int(dut.busy_o.value) == 0
    assert int(dut.stored_valid_o.value) == 0
    assert read_current_a(dut) == [[0, 0], [0, 0]]
    assert read_current_b(dut) == [[0, 0], [0, 0]]
    assert read_c(dut) == [[0, 0], [0, 0]]


@cocotb.test()
async def load_a_and_load_b_fetch_tiles_by_program_counter(dut):
    """LOAD_A and LOAD_B fetch A/B memory entries indexed by current PC."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    a = [[1, -2], [3, 4]]
    b = [[-5, 6], [7, -8]]
    await write_instr(dut, 0, OP_LOAD_A, 1)
    await write_instr(dut, 1, OP_LOAD_B, 1)
    await write_instr(dut, 2, OP_BASELINE, 1)
    await write_tile(dut, BANK_A, 0, a)
    await write_tile(dut, BANK_B, 1, b)

    await start_program(dut, 3)
    await wait_idle(dut)

    assert read_current_a(dut) == a
    assert read_current_b(dut) == b
    assert read_c(dut) == [[0, 0], [0, 0]]


@cocotb.test()
async def loaded_tiles_compute_and_store_c(dut):
    """LOAD_A, LOAD_B, MAC_TILE, STORE_C compute and capture one product."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    await write_instr(dut, 0, OP_LOAD_A, 1)
    await write_instr(dut, 1, OP_LOAD_B, 1)
    await write_instr(dut, 2, OP_MAC_TILE, 1)
    await write_instr(dut, 3, OP_STORE_C, 1)
    await write_tile(dut, BANK_A, 0, a)
    await write_tile(dut, BANK_B, 1, b)

    await start_program(dut, 4)
    await wait_idle(dut)

    expected = matmul2(a, b)
    assert int(dut.stored_valid_o.value) == 1
    assert read_stored_c(dut) == expected
    assert read_c(dut) == expected


@cocotb.test()
async def two_k_tiles_accumulate_from_memory(dut):
    """Two LOAD/MAC groups accumulate two K tiles before STORE_C."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    a0 = [[1, 0], [2, -1]]
    b0 = [[3, 4], [5, 6]]
    a1 = [[-2, 1], [0, 3]]
    b1 = [[7, -1], [2, 5]]

    program = [OP_LOAD_A, OP_LOAD_B, OP_MAC_TILE, OP_LOAD_A, OP_LOAD_B, OP_MAC_TILE, OP_STORE_C]
    for addr, opcode in enumerate(program):
        await write_instr(dut, addr, opcode, 1)
    await write_tile(dut, BANK_A, 0, a0)
    await write_tile(dut, BANK_B, 1, b0)
    await write_tile(dut, BANK_A, 3, a1)
    await write_tile(dut, BANK_B, 4, b1)

    await start_program(dut, len(program))
    await wait_idle(dut)

    expected = add2(matmul2(a0, b0), matmul2(a1, b1))
    assert read_stored_c(dut) == expected
    assert read_c(dut) == expected


@cocotb.test()
async def deterministic_random_memory_program_matches_python(dut):
    """Random tiles loaded from memory and accumulated by MAC_TILE match Python."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    rng = random.Random(123)

    program = []
    expected = [[0, 0], [0, 0]]
    pc = 0
    for _ in range(4):
        a = [[rng.randint(-8, 8), rng.randint(-8, 8)], [rng.randint(-8, 8), rng.randint(-8, 8)]]
        b = [[rng.randint(-8, 8), rng.randint(-8, 8)], [rng.randint(-8, 8), rng.randint(-8, 8)]]
        program.extend([OP_LOAD_A, OP_LOAD_B, OP_MAC_TILE])
        await write_tile(dut, BANK_A, pc, a)
        await write_tile(dut, BANK_B, pc + 1, b)
        expected = add2(expected, matmul2(a, b))
        pc += 3
    program.append(OP_STORE_C)

    for addr, opcode in enumerate(program):
        await write_instr(dut, addr, opcode, 1)

    await start_program(dut, len(program))
    await wait_idle(dut)

    assert int(dut.stored_valid_o.value) == 1
    assert read_stored_c(dut) == expected
