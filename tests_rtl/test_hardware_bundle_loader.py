"""Load exported hardware bundle files into rtl/tile_engine_mem_2x2.sv.

This is the local simulation version of a future host/FPGA loader flow. The test
reads instructions.hex, a_memory.hex, b_memory.hex, and expected_c.json files,
programs the RTL tile engine, and checks the reconstructed 4x4 output matrix.
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

from opentsp.hardware_export import export_default_matmul_4x4_hardware_bundle  # noqa: E402
from opentsp.hardware_loader import load_hardware_bundle, reconstruct_matrix_from_tile_outputs  # noqa: E402

BANK_A = 0
BANK_B = 1


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


async def write_tile(dut, bank: int, addr: int, values) -> None:
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
    raise AssertionError("Timed out waiting for hardware-bundle tile program to finish")


def read_stored_c(dut) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (int(dut.stored_c00_o.value.signed_integer), int(dut.stored_c01_o.value.signed_integer)),
        (int(dut.stored_c10_o.value.signed_integer), int(dut.stored_c11_o.value.signed_integer)),
    )


async def run_loaded_tile_program(dut, tile_program) -> tuple[tuple[int, int], tuple[int, int]]:
    await reset(dut)

    for addr, instr in enumerate(tile_program.instructions):
        await write_instr(dut, addr, instr.opcode, instr.cycles)
    for addr, tile in enumerate(tile_program.a_memory):
        await write_tile(dut, BANK_A, addr, tile)
    for addr, tile in enumerate(tile_program.b_memory):
        await write_tile(dut, BANK_B, addr, tile)

    await start_program(dut, tile_program.program_len)
    await wait_idle(dut)

    assert int(dut.stored_valid_o.value) == 1, tile_program.name
    return read_stored_c(dut)


@cocotb.test()
async def exported_hardware_bundle_runs_on_tile_engine(dut):
    """Exported hex files load into RTL and reconstruct the expected 4x4 C matrix."""

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    bundle_dir = REPO_ROOT / "artifacts" / "hardware_bundle" / "matmul_4x4"
    export_default_matmul_4x4_hardware_bundle(bundle_dir)
    bundle = load_hardware_bundle(bundle_dir)

    observed_tiles: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for tile_program in bundle.tile_programs:
        observed = await run_loaded_tile_program(dut, tile_program)
        assert observed == tile_program.expected_c, tile_program.name
        observed_tiles[tile_program.name] = observed

    reconstructed = reconstruct_matrix_from_tile_outputs(bundle.tile_programs, observed_tiles)
    assert reconstructed == bundle.expected_c
