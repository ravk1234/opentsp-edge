from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class InstructionOpcode(str, Enum):
    """Small accelerator instruction vocabulary for the OpenTSP simulator."""

    LOAD_A = "LOAD_A"
    LOAD_B = "LOAD_B"
    MAC_TILE = "MAC_TILE"
    STORE_C = "STORE_C"
    ATTENTION = "ATTENTION"
    BASELINE = "BASELINE"


@dataclass(frozen=True)
class InstructionMemoryLayout:
    """Logical SRAM-bank assignment used by the instruction emitter.

    This is still an abstract layout. It gives every instruction deterministic
    bank metadata so a future RTL controller can consume a fixed instruction
    stream. The numbers do not claim a physical SRAM implementation yet.
    """

    a_bank: int = 0
    b_bank: int = 1
    c_bank: int = 2
    kv_bank: int = 3
    temp_bank: int = 4


@dataclass(frozen=True)
class AcceleratorInstruction:
    """One controller-style instruction derived from the deterministic timeline."""

    index: int
    opcode: InstructionOpcode
    op_name: str
    micro_op: str
    start_cycle: int
    cycles: int
    end_cycle: int
    source: str
    bank: int | None = None
    offset: int | None = None
    bytes_moved: int = 0
    macs: int = 0
    m_range: tuple[int, int] | None = None
    n_range: tuple[int, int] | None = None
    k_range: tuple[int, int] | None = None
    token_range: tuple[int, int] | None = None
    dim_range: tuple[int, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        data = asdict(self)
        data["opcode"] = self.opcode.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AcceleratorInstruction":
        """Reconstruct an instruction from a JSON-friendly dictionary."""

        def tup2(value: Any) -> tuple[int, int] | None:
            if value is None:
                return None
            if len(value) != 2:
                raise ValueError(f"expected pair, got {value!r}")
            return (int(value[0]), int(value[1]))

        return cls(
            index=int(data["index"]),
            opcode=InstructionOpcode(data["opcode"]),
            op_name=str(data["op_name"]),
            micro_op=str(data["micro_op"]),
            start_cycle=int(data["start_cycle"]),
            cycles=int(data["cycles"]),
            end_cycle=int(data["end_cycle"]),
            source=str(data["source"]),
            bank=None if data.get("bank") is None else int(data["bank"]),
            offset=None if data.get("offset") is None else int(data["offset"]),
            bytes_moved=int(data.get("bytes_moved", 0)),
            macs=int(data.get("macs", 0)),
            m_range=tup2(data.get("m_range")),
            n_range=tup2(data.get("n_range")),
            k_range=tup2(data.get("k_range")),
            token_range=tup2(data.get("token_range")),
            dim_range=tup2(data.get("dim_range")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class InstructionProgram:
    """A deterministic accelerator instruction stream."""

    instructions: tuple[AcceleratorInstruction, ...]
    total_cycles: int

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)

    def opcode_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for instr in self.instructions:
            counts[instr.opcode.value] = counts.get(instr.opcode.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "instruction_count": self.instruction_count,
            "opcode_counts": self.opcode_counts(),
            "instructions": [instr.to_dict() for instr in self.instructions],
        }
