from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .instruction import AcceleratorInstruction, InstructionMemoryLayout, InstructionOpcode, InstructionProgram
from .timeline import TimelineEvent, TimelineResult

_PAIR_RE = re.compile(r"([M N K T D])\((\d+),\s*(\d+)\)")


def _parse_detail_ranges(detail: str) -> dict[str, tuple[int, int]]:
    """Parse timeline details such as `M(0, 1) N(0, 8) K(0, 8)`."""

    ranges: dict[str, tuple[int, int]] = {}
    for key, start, end in _PAIR_RE.findall(detail):
        ranges[key] = (int(start), int(end))
    return ranges


def _opcode_for_event(event: TimelineEvent) -> InstructionOpcode:
    if event.source == "tiled_matmul":
        if event.micro_op == "load_a_tile":
            return InstructionOpcode.LOAD_A
        if event.micro_op == "load_b_tile":
            return InstructionOpcode.LOAD_B
        if event.micro_op == "mac_tile":
            return InstructionOpcode.MAC_TILE
        if event.micro_op == "store_c_tile":
            return InstructionOpcode.STORE_C
    if event.source == "kv_attention":
        return InstructionOpcode.ATTENTION
    return InstructionOpcode.BASELINE


def _bank_for_opcode(opcode: InstructionOpcode, layout: InstructionMemoryLayout) -> int | None:
    if opcode == InstructionOpcode.LOAD_A:
        return layout.a_bank
    if opcode == InstructionOpcode.LOAD_B:
        return layout.b_bank
    if opcode == InstructionOpcode.STORE_C:
        return layout.c_bank
    if opcode == InstructionOpcode.ATTENTION:
        return layout.kv_bank
    if opcode == InstructionOpcode.BASELINE:
        return layout.temp_bank
    return None


def emit_instructions_from_timeline(
    timeline: TimelineResult,
    *,
    layout: InstructionMemoryLayout | None = None,
) -> InstructionProgram:
    """Convert a unified timeline into controller-style instructions.

    The resulting stream is deliberately simple and deterministic. It is not a
    final ISA, but it is structured enough for a future RTL controller FSM:
    load A/B tiles, trigger MAC_TILE, store C, and represent attention/baseline
    work as explicit coarse instructions.
    """

    mem = layout or InstructionMemoryLayout()
    bank_offsets: dict[int, int] = {
        mem.a_bank: 0,
        mem.b_bank: 0,
        mem.c_bank: 0,
        mem.kv_bank: 0,
        mem.temp_bank: 0,
    }
    instructions: list[AcceleratorInstruction] = []

    for event in timeline.events:
        opcode = _opcode_for_event(event)
        bank = _bank_for_opcode(opcode, mem)
        offset: int | None = None
        if bank is not None:
            offset = bank_offsets.setdefault(bank, 0)
            # Reserve at least one slot even for compute-only instructions so
            # instruction addresses remain monotonically meaningful per bank.
            bank_offsets[bank] += max(int(event.bytes_moved), 1)

        ranges = _parse_detail_ranges(event.detail)
        instructions.append(
            AcceleratorInstruction(
                index=len(instructions),
                opcode=opcode,
                op_name=event.op_name,
                micro_op=event.micro_op,
                start_cycle=int(event.start_cycle),
                end_cycle=int(event.end_cycle),
                cycles=int(event.cycles),
                source=event.source,
                bank=bank,
                offset=offset,
                bytes_moved=int(event.bytes_moved),
                macs=int(event.macs),
                m_range=ranges.get("M"),
                n_range=ranges.get("N"),
                k_range=ranges.get("K"),
                token_range=ranges.get("T"),
                dim_range=ranges.get("D"),
                metadata={"detail": event.detail},
            )
        )

    return InstructionProgram(instructions=tuple(instructions), total_cycles=int(timeline.total_cycles))


def write_instruction_program_json(program: InstructionProgram, path: str | Path) -> Path:
    """Write an instruction program as stable, pretty JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(program.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_instruction_program_json(path: str | Path) -> InstructionProgram:
    """Read an instruction program written by `write_instruction_program_json`."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    instructions = tuple(AcceleratorInstruction.from_dict(item) for item in raw["instructions"])
    return InstructionProgram(instructions=instructions, total_cycles=int(raw["total_cycles"]))


def summarize_instructions(instructions: Iterable[AcceleratorInstruction]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for instr in instructions:
        counts[instr.opcode.value] = counts.get(instr.opcode.value, 0) + 1
    return tuple(sorted(counts.items()))
