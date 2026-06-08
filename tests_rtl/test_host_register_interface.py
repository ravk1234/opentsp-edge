"""cocotb tests for rtl/host_tile_engine_2x2.sv."""

from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

ADDR_CONTROL = 0x00
ADDR_STATUS = 0x04
ADDR_PROGRAM_LEN = 0x08
ADDR_PC = 0x0C
ADDR_INSTR_ADDR = 0x10
ADDR_INSTR_WORD = 0x14
ADDR_DATA_ADDR = 0x18
ADDR_DATA_BANK = 0x1C
ADDR_DATA_WORD = 0x20
ADDR_C00 = 0x30
ADDR_C01 = 0x34
ADDR_C10 = 0x38
ADDR_C11 = 0x3C

OP_LOAD_A = 1
OP_LOAD_B = 2
OP_MAC_TILE = 3
OP_STORE_C = 4
OP_BASELINE = 6

BANK_A = 0
BANK_B = 1

STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1
STATUS_STORED_VALID = 1 << 2


def pack_instr(opcode: int, cycles: int = 1) -> int:
    return (int(opcode) & 0xFF) | ((int(cycles) & 0xFFFF) << 8)


def pack_i8(value: int) -> int:
    return int(value) & 0xFF


def pack_tile(tile) -> int:
    return (
        pack_i8(tile[0][0])
        | (pack_i8(tile[0][1]) << 8)
        | (pack_i8(tile[1][0]) << 16)
        | (pack_i8(tile[1][1]) << 24)
    )


def to_signed32(value: int) -> int:
    value = int(value) & 0xFFFFFFFF
    if value & 0x80000000:
        return value - 0x100000000
    return value


def matmul2(a, b):
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


async def reset(dut):
    dut.rst_n.value = 0
    dut.reg_we_i.value = 0
    dut.reg_re_i.value = 0
    dut.reg_addr_i.value = 0
    dut.reg_wdata_i.value = 0
    await tick(dut, 2)
    dut.rst_n.value = 1
    await tick(dut, 2)


async def write_reg(dut, addr: int, value: int):
    dut.reg_addr_i.value = addr
    dut.reg_wdata_i.value = value & 0xFFFFFFFF
    dut.reg_we_i.value = 1
    dut.reg_re_i.value = 0
    await tick(dut)
    dut.reg_we_i.value = 0
    await tick(dut)


async def read_reg(dut, addr: int) -> int:
    dut.reg_addr_i.value = addr
    dut.reg_we_i.value = 0
    dut.reg_re_i.value = 1
    await Timer(1, units="ns")
    value = int(dut.reg_rdata_o.value)
    dut.reg_re_i.value = 0
    await tick(dut)
    return value


async def write_instruction(dut, addr: int, opcode: int, cycles: int = 1):
    await write_reg(dut, ADDR_INSTR_ADDR, addr)
    await write_reg(dut, ADDR_INSTR_WORD, pack_instr(opcode, cycles))


async def write_tile(dut, bank: int, addr: int, tile):
    await write_reg(dut, ADDR_DATA_BANK, bank)
    await write_reg(dut, ADDR_DATA_ADDR, addr)
    await write_reg(dut, ADDR_DATA_WORD, pack_tile(tile))


async def start(dut, program_len: int):
    await write_reg(dut, ADDR_PROGRAM_LEN, program_len)
    await write_reg(dut, ADDR_CONTROL, 0x1)


async def wait_done(dut, max_cycles: int = 200):
    for _ in range(max_cycles):
        status = await read_reg(dut, ADDR_STATUS)
        if status & STATUS_DONE:
            return status
        await tick(dut)
    raise AssertionError("Timed out waiting for host STATUS.done")


async def read_c_tile(dut):
    return [
        [to_signed32(await read_reg(dut, ADDR_C00)), to_signed32(await read_reg(dut, ADDR_C01))],
        [to_signed32(await read_reg(dut, ADDR_C10)), to_signed32(await read_reg(dut, ADDR_C11))],
    ]


@cocotb.test()
async def reset_status_is_idle(dut):
    """After reset, host status is idle and output registers are zero."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    status = await read_reg(dut, ADDR_STATUS)
    assert status & STATUS_BUSY == 0
    assert status & STATUS_DONE == 0
    assert status & STATUS_STORED_VALID == 0
    assert await read_reg(dut, ADDR_C00) == 0
    assert await read_reg(dut, ADDR_C01) == 0
    assert await read_reg(dut, ADDR_C10) == 0
    assert await read_reg(dut, ADDR_C11) == 0


@cocotb.test()
async def config_registers_roundtrip(dut):
    """Basic host configuration registers round-trip through readback."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    await write_reg(dut, ADDR_PROGRAM_LEN, 7)
    await write_reg(dut, ADDR_INSTR_ADDR, 3)
    await write_reg(dut, ADDR_DATA_ADDR, 5)
    await write_reg(dut, ADDR_DATA_BANK, 1)

    assert await read_reg(dut, ADDR_PROGRAM_LEN) == 7
    assert await read_reg(dut, ADDR_INSTR_ADDR) == 3
    assert await read_reg(dut, ADDR_DATA_ADDR) == 5
    assert await read_reg(dut, ADDR_DATA_BANK) == 1


@cocotb.test()
async def host_loads_program_and_reads_c_registers(dut):
    """Host writes instructions/data, starts the engine, and reads C output."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    expected = matmul2(a, b)

    await write_instruction(dut, 0, OP_LOAD_A, 1)
    await write_instruction(dut, 1, OP_LOAD_B, 1)
    await write_instruction(dut, 2, OP_MAC_TILE, 1)
    await write_instruction(dut, 3, OP_STORE_C, 1)
    await write_tile(dut, BANK_A, 0, a)
    await write_tile(dut, BANK_B, 1, b)

    await start(dut, 4)
    status = await wait_done(dut)

    assert status & STATUS_STORED_VALID
    assert await read_c_tile(dut) == expected


@cocotb.test()
async def host_can_clear_and_reuse_engine(dut):
    """CONTROL.clear resets sticky done/output-valid so a second program can run cleanly."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    program = [OP_LOAD_A, OP_LOAD_B, OP_MAC_TILE, OP_STORE_C]
    for addr, opcode in enumerate(program):
        await write_instruction(dut, addr, opcode, 1)

    a0 = [[2, 0], [0, 2]]
    b0 = [[3, 1], [4, 5]]
    await write_tile(dut, BANK_A, 0, a0)
    await write_tile(dut, BANK_B, 1, b0)
    await start(dut, 4)
    await wait_done(dut)
    assert await read_c_tile(dut) == matmul2(a0, b0)

    await write_reg(dut, ADDR_CONTROL, 0x2)
    status = await read_reg(dut, ADDR_STATUS)
    assert status & STATUS_DONE == 0

    a1 = [[-1, 3], [2, 1]]
    b1 = [[4, -2], [0, 5]]
    await write_tile(dut, BANK_A, 0, a1)
    await write_tile(dut, BANK_B, 1, b1)
    await start(dut, 4)
    await wait_done(dut)
    assert await read_c_tile(dut) == matmul2(a1, b1)
