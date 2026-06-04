"""cocotb tests for rtl/instruction_controller.sv."""

from __future__ import annotations

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


async def start_and_reset(dut) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.host_we_i.value = 0
    dut.host_addr_i.value = 0
    dut.host_opcode_i.value = 0
    dut.host_cycles_i.value = 0
    dut.start_i.value = 0
    dut.program_len_i.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")


async def write_instr(dut, addr: int, opcode: int, cycles: int) -> None:
    dut.host_addr_i.value = addr
    dut.host_opcode_i.value = opcode
    dut.host_cycles_i.value = cycles
    dut.host_we_i.value = 1
    await RisingEdge(dut.clk)
    dut.host_we_i.value = 0
    await Timer(1, units="ns")


async def start_program(dut, program_len: int) -> None:
    dut.program_len_i.value = program_len
    dut.start_i.value = 1
    await RisingEdge(dut.clk)
    dut.start_i.value = 0
    await Timer(1, units="ns")


def state(dut) -> tuple[int, int, int, int, int]:
    return (
        int(dut.pc_o.value),
        int(dut.opcode_o.value),
        int(dut.cycles_left_o.value),
        int(dut.valid_o.value),
        int(dut.global_cycle_o.value),
    )


@cocotb.test()
async def reset_state_is_idle(dut):
    """After reset, the controller is idle and has no valid instruction."""
    await start_and_reset(dut)
    assert int(dut.busy_o.value) == 0
    assert int(dut.done_o.value) == 0
    assert int(dut.valid_o.value) == 0
    assert int(dut.opcode_o.value) == OP_NOP
    assert int(dut.global_cycle_o.value) == 0


@cocotb.test()
async def executes_program_with_cycle_counts(dut):
    """The controller holds each instruction for its programmed cycle count."""
    await start_and_reset(dut)
    program = [
        (OP_LOAD_A, 2),
        (OP_LOAD_B, 1),
        (OP_MAC_TILE, 3),
        (OP_STORE_C, 1),
    ]
    for addr, (opcode, cycles) in enumerate(program):
        await write_instr(dut, addr, opcode, cycles)

    await start_program(dut, len(program))
    assert state(dut) == (0, OP_LOAD_A, 2, 1, 0)
    assert int(dut.load_a_o.value) == 1

    expected_after_edges = [
        (0, OP_LOAD_A, 1, 1, 1),
        (1, OP_LOAD_B, 1, 1, 2),
        (2, OP_MAC_TILE, 3, 1, 3),
        (2, OP_MAC_TILE, 2, 1, 4),
        (2, OP_MAC_TILE, 1, 1, 5),
        (3, OP_STORE_C, 1, 1, 6),
    ]
    for expected in expected_after_edges:
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        assert state(dut) == expected

    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    assert int(dut.busy_o.value) == 0
    assert int(dut.valid_o.value) == 0
    assert int(dut.done_o.value) == 1
    assert int(dut.global_cycle_o.value) == 7


@cocotb.test()
async def zero_cycle_instruction_is_normalized_to_one_cycle(dut):
    """A programmed cycle count of zero is treated as one cycle."""
    await start_and_reset(dut)
    await write_instr(dut, 0, OP_BASELINE, 0)
    await start_program(dut, 1)
    assert state(dut) == (0, OP_BASELINE, 1, 1, 0)
    assert int(dut.baseline_o.value) == 1

    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    assert int(dut.done_o.value) == 1
    assert int(dut.busy_o.value) == 0


@cocotb.test()
async def zero_length_program_finishes_immediately(dut):
    """Starting with program_len_i=0 raises done without entering busy state."""
    await start_and_reset(dut)
    await start_program(dut, 0)
    assert int(dut.done_o.value) == 1
    assert int(dut.busy_o.value) == 0
    assert int(dut.valid_o.value) == 0


@cocotb.test()
async def decoded_outputs_match_current_opcode(dut):
    """Decoded one-hot outputs reflect the current instruction opcode."""
    await start_and_reset(dut)
    program = [
        (OP_LOAD_A, 1),
        (OP_LOAD_B, 1),
        (OP_MAC_TILE, 1),
        (OP_STORE_C, 1),
        (OP_ATTENTION, 1),
        (OP_BASELINE, 1),
    ]
    for addr, (opcode, cycles) in enumerate(program):
        await write_instr(dut, addr, opcode, cycles)

    await start_program(dut, len(program))
    checks = [
        (OP_LOAD_A, "load_a_o"),
        (OP_LOAD_B, "load_b_o"),
        (OP_MAC_TILE, "mac_tile_o"),
        (OP_STORE_C, "store_c_o"),
        (OP_ATTENTION, "attention_o"),
        (OP_BASELINE, "baseline_o"),
    ]

    for idx, (opcode, signal_name) in enumerate(checks):
        assert int(dut.opcode_o.value) == opcode
        assert int(getattr(dut, signal_name).value) == 1
        if idx != len(checks) - 1:
            await RisingEdge(dut.clk)
            await Timer(1, units="ns")
