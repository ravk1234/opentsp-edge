from __future__ import annotations

from pathlib import Path

from opentsp.instruction import InstructionOpcode
from opentsp.instruction_emitter import emit_instructions_from_timeline, write_instruction_program_json
from opentsp.tiny_gpt_adapter import run_tiny_gpt_opentsp


def main() -> None:
    result = run_tiny_gpt_opentsp([4, 12, 9, 3, 21])
    program = emit_instructions_from_timeline(result.timeline)

    out_path = write_instruction_program_json(program, Path("artifacts/instructions/tiny_gpt_instructions.json"))
    counts = program.opcode_counts()

    print("Tiny GPT instruction emission demo")
    print("-" * 80)
    print(f"Graph: {result.program.graph_name}")
    print(f"Output: {out_path}")
    print(f"Instruction count: {program.instruction_count}")
    print(f"Total cycles: {program.total_cycles}")
    print(f"LOAD_A instructions: {counts.get(InstructionOpcode.LOAD_A.value, 0)}")
    print(f"LOAD_B instructions: {counts.get(InstructionOpcode.LOAD_B.value, 0)}")
    print(f"MAC_TILE instructions: {counts.get(InstructionOpcode.MAC_TILE.value, 0)}")
    print(f"STORE_C instructions: {counts.get(InstructionOpcode.STORE_C.value, 0)}")
    print(f"ATTENTION instructions: {counts.get(InstructionOpcode.ATTENTION.value, 0)}")
    print(f"BASELINE instructions: {counts.get(InstructionOpcode.BASELINE.value, 0)}")

    print("\nFirst 24 instructions")
    print("-" * 80)
    for instr in program.instructions[:24]:
        rng = ""
        if instr.m_range is not None:
            rng = f" M{instr.m_range} N{instr.n_range} K{instr.k_range}"
        elif instr.token_range is not None:
            rng = f" T{instr.token_range} D{instr.dim_range}"
        print(
            f"{instr.index:03d} {instr.opcode.value:<10} op={instr.op_name:<14} "
            f"micro={instr.micro_op:<18} start={instr.start_cycle:<5} cycles={instr.cycles:<4} "
            f"bank={str(instr.bank):<4} offset={str(instr.offset):<6} bytes={instr.bytes_moved:<4}{rng}"
        )

    assert program.instruction_count == result.timeline.event_count
    assert counts.get(InstructionOpcode.MAC_TILE.value, 0) > 0
    assert counts.get(InstructionOpcode.ATTENTION.value, 0) > 0
    assert program.total_cycles == result.timeline.total_cycles
    print("\nInstruction emission check: PASSED")


if __name__ == "__main__":
    main()
