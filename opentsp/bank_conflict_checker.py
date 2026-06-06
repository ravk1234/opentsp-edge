"""Helpers for checking OpenTSP instruction streams against SRAM bank rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .memory_bank import MemoryAccess, MemoryBankConfig, MemoryCheckResult, check_memory_bank_conflicts

_READ_OPCODES = {"LOAD_A", "LOAD_B"}
_WRITE_OPCODES = {"STORE_C"}


def _read_field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def instruction_to_memory_access(instruction: Any, instruction_idx: int) -> MemoryAccess | None:
    """Convert one emitted instruction object/dict into a MemoryAccess when possible.

    LOAD_A and LOAD_B become read accesses. STORE_C becomes a write access.
    MAC_TILE, ATTENTION, BASELINE, and other instructions do not directly touch
    SRAM in this simplified checker and therefore return None.
    """

    opcode = str(_read_field(instruction, "opcode", "kind", "op", default="")).upper()
    if opcode not in _READ_OPCODES and opcode not in _WRITE_OPCODES:
        return None

    bank = _read_field(instruction, "bank", default=None)
    offset = _read_field(instruction, "offset", "offset_bytes", default=None)
    size_bytes = _read_field(instruction, "bytes", "size_bytes", "nbytes", default=None)
    start_cycle = _read_field(instruction, "start_cycle", "start", default=0)
    cycles = _read_field(instruction, "cycles", default=1)

    missing = [
        name
        for name, value in {
            "bank": bank,
            "offset": offset,
            "size_bytes": size_bytes,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"Instruction {instruction_idx} is missing memory fields: {', '.join(missing)}")

    return MemoryAccess(
        start_cycle=int(start_cycle),
        cycles=max(1, int(cycles)),
        bank=int(bank),
        offset=int(offset),
        size_bytes=int(size_bytes),
        kind="read" if opcode in _READ_OPCODES else "write",
        op_name=str(_read_field(instruction, "op_name", "op_name", "layer", default="")),
        micro_op=str(_read_field(instruction, "micro_op", "micro", default=opcode.lower())),
        instruction_idx=instruction_idx,
    )


def instructions_to_memory_accesses(instructions: Iterable[Any]) -> list[MemoryAccess]:
    """Convert a stream of OpenTSP instructions into memory accesses."""

    accesses: list[MemoryAccess] = []
    for idx, instruction in enumerate(instructions):
        access = instruction_to_memory_access(instruction, idx)
        if access is not None:
            accesses.append(access)
    return accesses


def check_instruction_memory_conflicts(
    instructions: Iterable[Any],
    config: MemoryBankConfig | None = None,
) -> MemoryCheckResult:
    """Check OpenTSP instructions for SRAM-bank conflicts."""

    accesses = instructions_to_memory_accesses(instructions)
    return check_memory_bank_conflicts(accesses, config=config)
