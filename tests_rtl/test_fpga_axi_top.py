# SPDX-License-Identifier: Apache-2.0
"""cocotb test for the FPGA-facing AXI top simulation wrapper."""

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

OP_LOAD_A = 0x01
OP_LOAD_B = 0x02
OP_MAC_TILE = 0x03
OP_STORE_C = 0x04


def instr_word(opcode: int, cycles: int = 1) -> int:
    return ((cycles & 0xFF) << 8) | (opcode & 0xFF)


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


async def write_reg(dut, addr: int, data: int, max_cycles: int = 50) -> None:
    dut.awaddr_i.value = addr
    dut.wdata_i.value = data & 0xFFFFFFFF
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


async def read_reg(dut, addr: int, max_cycles: int = 50) -> int:
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
        if status & 0x2:
            return
        await RisingEdge(dut.clk_i)
    raise AssertionError("Timed out waiting for top-level done status")


async def write_instruction_program(dut) -> None:
    program = [
        instr_word(OP_LOAD_A),
        instr_word(OP_LOAD_B),
        instr_word(OP_MAC_TILE),
        instr_word(OP_LOAD_A),
        instr_word(OP_LOAD_B),
        instr_word(OP_MAC_TILE),
        instr_word(OP_STORE_C),
    ]
    for idx, word in enumerate(program):
        await write_reg(dut, REG_INSTR_ADDR, idx)
        await write_reg(dut, REG_INSTR_WORD, word)
    await write_reg(dut, REG_PROGRAM_LEN, len(program))


async def write_memory_word(dut, bank: int, addr: int, word: int) -> None:
    await write_reg(dut, REG_DATA_BANK, bank)
    await write_reg(dut, REG_DATA_ADDR, addr)
    await write_reg(dut, REG_DATA_WORD, word)


async def write_c00_tile_memories(dut) -> None:
    # This data is the c00 tile program from the exported 4x4 matmul bundle.
    a_words = [0x0504FE01, 0x00000000, 0x00000000, 0x02FF0003]
    b_words = [0x00000000, 0x05FD0002, 0x00000000, 0x00000000, 0x0401FE06]

    for addr, word in enumerate(a_words):
        await write_memory_word(dut, bank=0, addr=addr, word=word)

    for addr, word in enumerate(b_words):
        await write_memory_word(dut, bank=1, addr=addr, word=word)


@cocotb.test()
async def fpga_axi_top_runs_single_exported_tile_program(dut):
    """The FPGA-facing AXI top runs a real exported c00 tile program."""
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset(dut)

    await write_reg(dut, REG_CONTROL, 0x2)  # clear
    await write_instruction_program(dut)
    await write_c00_tile_memories(dut)

    await write_reg(dut, REG_CONTROL, 0x1)  # start
    await wait_done(dut)

    c00 = u32_to_i32(await read_reg(dut, REG_C00))
    c01 = u32_to_i32(await read_reg(dut, REG_C01))
    c10 = u32_to_i32(await read_reg(dut, REG_C10))
    c11 = u32_to_i32(await read_reg(dut, REG_C11))

    assert [[c00, c01], [c10, c11]] == [[26, -16], [-11, 35]]
