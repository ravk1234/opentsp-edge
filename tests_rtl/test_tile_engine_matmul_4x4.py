"""Run a full 4x4 INT8 matmul through rtl/tile_engine_mem_2x2.sv.

The current memory-backed tile engine stores one 2x2 C tile per program run. This
cocotb test executes four deterministic 2x2 output-tile programs and reconstructs
the full 4x4 matrix output.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentsp.rtl_matmul_program import generate_default_matmul_4x4_program  # noqa: E402


async def tick(dut, n: int = 1) -> None:
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


async def reset(dut) -> None:
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


async def write_instr(dut, addr: int, opcode: int, cycles: int = 1) -> None:
    dut.host_addr_i.value = addr
    dut.host_opcode_i.value = opcode
    dut.host_cycles_i.value = cycles
    dut.host_we_i.value = 1
    await tick(dut)
    dut.host_we_i.value = 0
    await tick(dut)


async def write_tile(dut, bank: int, addr: int, values: list[list[int]]) -> None:
    dut.data_bank_i.value = bank
    dut.data_addr_i.value = addr
    dut.data00_i.value = values[0][0]
    dut.data01_i.value = values[0][1]
    dut.data10_i.value = values[1][0]
    dut.data11_i.value = values[1][1]
    dut.data_we_i.value = 1
    await tick(dut)
    dut.data_we_i.value = 0
    await tick(dut)


async def start_program(dut, program_len: int) -> None:
    dut.program_len_i.value = program_len
    dut.start_i.value = 1
    await tick(dut)
    dut.start_i.value = 0
    await tick(dut)


async def wait_idle(dut, max_cycles: int = 200) -> None:
    for _ in range(max_cycles):
        if int(dut.busy_o.value) == 0:
            return
        await tick(dut)
    raise AssertionError("Timed out waiting for tile engine to go idle")


def read_stored_c(dut) -> list[list[int]]:
    return [
        [int(dut.stored_c00_o.value.signed_integer), int(dut.stored_c01_o.value.signed_integer)],
        [int(dut.stored_c10_o.value.signed_integer), int(dut.stored_c11_o.value.signed_integer)],
    ]


async def run_tile_program(dut, tile_program) -> list[list[int]]:
    # Overwrite the tiny instruction and data memories for each 2x2 output tile.
    for addr, instr in enumerate(tile_program.instructions):
        await write_instr(dut, addr, instr.opcode, instr.cycles)
    for data_tile in tile_program.data_tiles:
        await write_tile(dut, data_tile.bank, data_tile.addr, data_tile.values)

    await start_program(dut, tile_program.program_len)
    await wait_idle(dut)

    assert int(dut.stored_valid_o.value) == 1, tile_program.name
    return read_stored_c(dut)


@cocotb.test()
async def full_4x4_matmul_matches_python_reference(dut):
    """Four 2x2 engine runs reconstruct the full 4x4 INT8 matmul output."""

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    program = generate_default_matmul_4x4_program()
    observed = [[0 for _ in range(4)] for _ in range(4)]

    for tile_program in program.tile_programs:
        tile_c = await run_tile_program(dut, tile_program)
        assert tile_c == tile_program.expected_c, tile_program.name
        r = tile_program.c_row
        c = tile_program.c_col
        observed[r][c] = tile_c[0][0]
        observed[r][c + 1] = tile_c[0][1]
        observed[r + 1][c] = tile_c[1][0]
        observed[r + 1][c + 1] = tile_c[1][1]

    assert observed == program.expected_c
