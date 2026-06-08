"""Run an exported hardware bundle through rtl/host_tile_engine_2x2.sv.

This verifies the future host-flow shape: exported bundle -> host register writes
-> RTL tile engine -> host reads C outputs.
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
from opentsp.host_driver import (  # noqa: E402
    ADDR_C00,
    ADDR_C01,
    ADDR_C10,
    ADDR_C11,
    ADDR_STATUS,
    STATUS_DONE,
    STATUS_STORED_VALID,
    HostWrite,
    build_tile_program_host_writes,
)


async def tick(dut, n: int = 1) -> None:
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


async def reset(dut) -> None:
    dut.rst_n.value = 0
    dut.reg_we_i.value = 0
    dut.reg_re_i.value = 0
    dut.reg_addr_i.value = 0
    dut.reg_wdata_i.value = 0
    await tick(dut, 2)
    dut.rst_n.value = 1
    await tick(dut, 2)


async def write_reg(dut, addr: int, value: int) -> None:
    dut.reg_addr_i.value = int(addr)
    dut.reg_wdata_i.value = int(value) & 0xFFFFFFFF
    dut.reg_we_i.value = 1
    dut.reg_re_i.value = 0
    await tick(dut)
    dut.reg_we_i.value = 0
    await tick(dut)


async def read_reg(dut, addr: int) -> int:
    dut.reg_addr_i.value = int(addr)
    dut.reg_we_i.value = 0
    dut.reg_re_i.value = 1
    await Timer(1, units="ns")
    value = int(dut.reg_rdata_o.value)
    dut.reg_re_i.value = 0
    await tick(dut)
    return value


async def apply_host_writes(dut, writes: tuple[HostWrite, ...]) -> None:
    for write in writes:
        await write_reg(dut, write.addr, write.value)


async def wait_done(dut, max_cycles: int = 300) -> int:
    for _ in range(max_cycles):
        status = await read_reg(dut, ADDR_STATUS)
        if status & STATUS_DONE:
            return status
        await tick(dut)
    raise AssertionError("Timed out waiting for host bundle runner STATUS.done")


def to_signed32(value: int) -> int:
    value = int(value) & 0xFFFFFFFF
    if value & 0x80000000:
        return value - 0x100000000
    return value


async def read_c_tile(dut) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (to_signed32(await read_reg(dut, ADDR_C00)), to_signed32(await read_reg(dut, ADDR_C01))),
        (to_signed32(await read_reg(dut, ADDR_C10)), to_signed32(await read_reg(dut, ADDR_C11))),
    )


@cocotb.test()
async def exported_bundle_runs_through_host_register_interface(dut):
    """Host writes exported bundle programs through registers and reconstructs 4x4 C."""

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    bundle_dir = REPO_ROOT / "artifacts" / "hardware_bundle" / "matmul_4x4"
    export_default_matmul_4x4_hardware_bundle(bundle_dir)
    bundle = load_hardware_bundle(bundle_dir)

    observed_tiles: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for tile_program in bundle.tile_programs:
        writes = build_tile_program_host_writes(tile_program)
        await apply_host_writes(dut, writes)
        status = await wait_done(dut)
        assert status & STATUS_STORED_VALID, tile_program.name

        observed = await read_c_tile(dut)
        assert observed == tile_program.expected_c, tile_program.name
        observed_tiles[tile_program.name] = observed

    reconstructed = reconstruct_matrix_from_tile_outputs(bundle.tile_programs, observed_tiles)
    assert reconstructed == bundle.expected_c
