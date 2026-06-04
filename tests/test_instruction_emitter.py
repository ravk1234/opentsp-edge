from __future__ import annotations

from pathlib import Path

from opentsp.instruction import InstructionOpcode
from opentsp.instruction_emitter import (
    emit_instructions_from_timeline,
    read_instruction_program_json,
    write_instruction_program_json,
)
from opentsp.tiny_gpt_adapter import run_tiny_gpt_opentsp


def _program():
    result = run_tiny_gpt_opentsp([4, 12, 9, 3, 21])
    return result, emit_instructions_from_timeline(result.timeline)


def test_instruction_count_matches_timeline_events() -> None:
    result, program = _program()
    assert program.instruction_count == result.timeline.event_count
    assert program.total_cycles == result.timeline.total_cycles


def test_instruction_opcode_counts_include_matmul_attention_and_baseline() -> None:
    _, program = _program()
    counts = program.opcode_counts()
    assert counts[InstructionOpcode.LOAD_A.value] > 0
    assert counts[InstructionOpcode.LOAD_B.value] > 0
    assert counts[InstructionOpcode.MAC_TILE.value] > 0
    assert counts[InstructionOpcode.STORE_C.value] > 0
    assert counts[InstructionOpcode.ATTENTION.value] > 0
    assert counts[InstructionOpcode.BASELINE.value] > 0


def test_instruction_order_preserves_timeline_cycles() -> None:
    _, program = _program()
    starts = [instr.start_cycle for instr in program.instructions]
    assert starts == sorted(starts)
    for instr in program.instructions:
        assert instr.end_cycle == instr.start_cycle + instr.cycles
        assert instr.cycles > 0


def test_first_tiled_matmul_instructions_have_ranges_and_banks() -> None:
    _, program = _program()
    first = program.instructions[:3]
    assert [instr.opcode for instr in first] == [
        InstructionOpcode.LOAD_A,
        InstructionOpcode.LOAD_B,
        InstructionOpcode.MAC_TILE,
    ]
    assert first[0].bank == 0
    assert first[1].bank == 1
    assert first[2].bank is None
    assert first[2].m_range is not None
    assert first[2].n_range is not None
    assert first[2].k_range is not None
    assert first[2].macs > 0


def test_instruction_json_round_trip(tmp_path: Path) -> None:
    _, program = _program()
    out = write_instruction_program_json(program, tmp_path / "instructions.json")
    loaded = read_instruction_program_json(out)
    assert loaded.total_cycles == program.total_cycles
    assert loaded.instruction_count == program.instruction_count
    assert loaded.opcode_counts() == program.opcode_counts()
    assert loaded.instructions[0] == program.instructions[0]
