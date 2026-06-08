# SPDX-License-Identifier: Apache-2.0
"""cocotb tests for the AXI-lite-style host register prototype."""

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
    dut.engine_busy_i.value = 0
    dut.engine_done_i.value = 0
    dut.c00_i.value = 0
    dut.c01_i.value = 0
    dut.c10_i.value = 0
    dut.c11_i.value = 0
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)
    await Timer(1, units="ns")


async def write_reg(dut, addr: int, data: int, max_cycles: int = 20) -> None:
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
            # Keep bready high so the caller's next clock edge clears the response.
            # This also lets one-cycle sideband pulses still be visible to the caller.
            dut.bready_i.value = 1
            return

        await RisingEdge(dut.clk_i)

    raise AssertionError("Timed out waiting for AXI-lite write response")

async def read_reg(dut, addr: int, max_cycles: int = 20) -> int:
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


@cocotb.test()
async def reset_state_is_clear(dut):
    """After reset, output pulses and register state are clear."""
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset(dut)

    assert int(dut.start_o.value) == 0
    assert int(dut.clear_o.value) == 0
    assert int(dut.instr_write_o.value) == 0
    assert int(dut.data_write_o.value) == 0
    assert int(dut.program_len_o.value) == 0


@cocotb.test()
async def writes_decode_to_control_and_memory_sidebands(dut):
    """Register writes decode into clean one-cycle sideband signals."""
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset(dut)

    await write_reg(dut, REG_PROGRAM_LEN, 7)
    assert int(dut.program_len_o.value) == 7

    await write_reg(dut, REG_INSTR_ADDR, 3)
    assert int(dut.instr_addr_o.value) == 3
    await write_reg(dut, REG_INSTR_WORD, 0x00000103)
    assert int(dut.instr_word_o.value) == 0x00000103
    assert int(dut.instr_write_o.value) == 1
    await RisingEdge(dut.clk_i)
    await Timer(1, units="ns")
    assert int(dut.instr_write_o.value) == 0

    await write_reg(dut, REG_DATA_BANK, 1)
    await write_reg(dut, REG_DATA_ADDR, 5)
    await write_reg(dut, REG_DATA_WORD, 0x0504FE01)
    assert int(dut.data_bank_o.value) == 1
    assert int(dut.data_addr_o.value) == 5
    assert int(dut.data_word_o.value) == 0x0504FE01
    assert int(dut.data_write_o.value) == 1
    await RisingEdge(dut.clk_i)
    await Timer(1, units="ns")
    assert int(dut.data_write_o.value) == 0

    await write_reg(dut, REG_CONTROL, 0x3)
    assert int(dut.start_o.value) == 1
    assert int(dut.clear_o.value) == 1
    await RisingEdge(dut.clk_i)
    await Timer(1, units="ns")
    assert int(dut.start_o.value) == 0
    assert int(dut.clear_o.value) == 0


@cocotb.test()
async def reads_status_and_c_output_registers(dut):
    """Read path returns status and C output registers."""
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset(dut)

    dut.engine_busy_i.value = 1
    dut.engine_done_i.value = 1
    assert await read_reg(dut, REG_STATUS) == 0x3

    dut.c00_i.value = 0x0000001A
    dut.c01_i.value = 0xFFFFFFF0
    dut.c10_i.value = 0xFFFFFFF5
    dut.c11_i.value = 0x00000023

    assert await read_reg(dut, REG_C00) == 0x0000001A
    assert await read_reg(dut, REG_C01) == 0xFFFFFFF0
    assert await read_reg(dut, REG_C10) == 0xFFFFFFF5
    assert await read_reg(dut, REG_C11) == 0x00000023


@cocotb.test()
async def unknown_register_access_is_safe(dut):
    """Unknown accesses are acknowledged and read as zero."""
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset(dut)

    await write_reg(dut, 0x7C, 0xDEADBEEF)
    assert int(dut.instr_write_o.value) == 0
    assert int(dut.data_write_o.value) == 0
    assert await read_reg(dut, 0x7C) == 0
