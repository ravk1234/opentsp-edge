# SPDX-License-Identifier: Apache-2.0
"""cocotb tests for the AXI-lite-controlled OpenTSP 2x2 tile engine."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

REG_CONTROL = 0x00
REG_STATUS = 0x04
REG_PROGRAM_LEN = 0x08
REG_INSTR_ADDR = 0x10
REG_INSTR_WORD = 0x14
REG_DATA_ADDR = 0x18
REG_DATA_BANK = 0x1C
REG_DATA_WORD = 0x20
REG_C00 = 0x30
REG_C01 = 0x34
REG_C10 = 0x38
REG_C11 = 0x3C


def u32_to_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return value - 0x100000000
    return value


async def reset(dut):
    dut.rst_ni.value = 0
    dut.awaddr_i.value = 0
    dut.awvalid_i.value = 0
    dut.wdata_i.value = 0
    dut.wvalid_i.value = 0
    dut.bready_i.value = 0
    dut.araddr_i.value = 0
    dut.arvalid_i.value = 0
    dut.rready_i.value = 0
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)
    await Timer(1, units="ns")


async def write_reg(dut, addr: int, data: int, max_cycles: int = 40) -> None:
    dut.awaddr_i.value = addr
    dut.wdata_i.value = data
    dut.awvalid_i.value = 1
    dut.wvalid_i.value = 1
    dut.bready_i.value = 0

    for _ in range(max_cycles):
        await RisingEdge(dut.clk_i)
        await Timer(1, units="ns")
        if int(dut.awready_o.value) == 1 and int(dut.wready_o.value) == 1:
            dut.awvalid_i.value = 0
            dut.wvalid_i.value = 0
            break
    else:
        raise AssertionError("Timed out waiting for AXI-lite write ready")

    for _ in range(max_cycles):
        await Timer(1, units="ns")
        if int(dut.bvalid_o.value) == 1:
            dut.bready_i.value = 1
            await RisingEdge(dut.clk_i)
            await Timer(1, units="ns")
            dut.bready_i.value = 0
            return
        await RisingEdge(dut.clk_i)

    raise AssertionError("Timed out waiting for AXI-lite write response")


async def read_reg(dut, addr: int, max_cycles: int = 40) -> int:
    dut.araddr_i.value = addr
    dut.arvalid_i.value = 1
    dut.rready_i.value = 0

    for _ in range(max_cycles):
        await RisingEdge(dut.clk_i)
        await Timer(1, units="ns")
        if int(dut.arready_o.value) == 1:
            dut.arvalid_i.value = 0
            break
    else:
        raise AssertionError("Timed out waiting for AXI-lite read address ready")

    for _ in range(max_cycles):
        await Timer(1, units="ns")
        if int(dut.rvalid_o.value) == 1:
            value = int(dut.rdata_o.value)
            dut.rready_i.value = 1
            await RisingEdge(dut.clk_i)
            await Timer(1, units="ns")
            dut.rready_i.value = 0
            return value
        await RisingEdge(dut.clk_i)

    raise AssertionError("Timed out waiting for AXI-lite read data")


async def wait_done(dut, max_cycles: int = 200) -> None:
    for _ in range(max_cycles):
        status = await read_reg(dut, REG_STATUS)
        done = (status >> 1) & 1
        if done:
            return
        await RisingEdge(dut.clk_i)
        await Timer(1, units="ns")
    raise AssertionError("Timed out waiting for AXI-lite tile engine done")


async def load_program(dut):
    # Program: LOAD_A, LOAD_B, MAC_TILE, LOAD_A, LOAD_B, MAC_TILE, STORE_C
    instructions = [
        0x00000101,
        0x00000102,
        0x00000103,
        0x00000101,
        0x00000102,
        0x00000103,
        0x00000104,
    ]
    for idx, word in enumerate(instructions):
        await write_reg(dut, REG_INSTR_ADDR, idx)
        await write_reg(dut, REG_INSTR_WORD, word)
    await write_reg(dut, REG_PROGRAM_LEN, len(instructions))


async def load_tile_data(dut):
    # Data words are from the exported c00 tile in the 4x4 matmul bundle.
    # A memory entries used by LOAD_A at pc=0 and pc=3.
    a_words = {
        0: 0x0504FE01,
        3: 0x02FF0003,
    }
    # B memory entries used by LOAD_B at pc=1 and pc=4.
    b_words = {
        1: 0x05FD0002,
        4: 0x0401FE06,
    }

    await write_reg(dut, REG_DATA_BANK, 0)
    for addr, word in a_words.items():
        await write_reg(dut, REG_DATA_ADDR, addr)
        await write_reg(dut, REG_DATA_WORD, word)

    await write_reg(dut, REG_DATA_BANK, 1)
    for addr, word in b_words.items():
        await write_reg(dut, REG_DATA_ADDR, addr)
        await write_reg(dut, REG_DATA_WORD, word)


@cocotb.test()
async def axi_lite_writes_run_one_tile_program(dut):
    """AXI-lite writes load one tile program, run it, and read the expected C tile."""
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset(dut)

    # Clear previous state.
    await write_reg(dut, REG_CONTROL, 0x2)
    await load_program(dut)
    await load_tile_data(dut)

    await write_reg(dut, REG_CONTROL, 0x1)
    await wait_done(dut)

    c00 = u32_to_i32(await read_reg(dut, REG_C00))
    c01 = u32_to_i32(await read_reg(dut, REG_C01))
    c10 = u32_to_i32(await read_reg(dut, REG_C10))
    c11 = u32_to_i32(await read_reg(dut, REG_C11))

    assert [[c00, c01], [c10, c11]] == [[26, -16], [-11, 35]]
    assert int(dut.stored_valid_o.value) == 1
