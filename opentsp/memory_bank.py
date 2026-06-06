"""Memory-bank access model and conflict checker for OpenTSP.

This module is intentionally small and deterministic. It does not simulate data
values; it checks whether a scheduled set of memory accesses can be served by a
simple SRAM-bank model with a fixed number of read/write ports per bank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

AccessKind = Literal["read", "write"]
ConflictKind = Literal[
    "invalid_access",
    "out_of_bounds",
    "read_port_conflict",
    "write_port_conflict",
    "read_write_conflict",
]


@dataclass(frozen=True)
class MemoryBankConfig:
    """Configuration for a banked scratchpad memory model.

    Args:
        num_banks: Number of independently addressable banks.
        bank_size_bytes: Capacity of each bank.
        read_ports_per_bank: Max reads from the same bank in the same cycle.
        write_ports_per_bank: Max writes to the same bank in the same cycle.
        allow_same_cycle_read_write: Whether the same bank can be read and
            written in the same cycle.
    """

    num_banks: int = 4
    bank_size_bytes: int = 4096
    read_ports_per_bank: int = 1
    write_ports_per_bank: int = 1
    allow_same_cycle_read_write: bool = False

    def __post_init__(self) -> None:
        if self.num_banks <= 0:
            raise ValueError("num_banks must be positive")
        if self.bank_size_bytes <= 0:
            raise ValueError("bank_size_bytes must be positive")
        if self.read_ports_per_bank <= 0:
            raise ValueError("read_ports_per_bank must be positive")
        if self.write_ports_per_bank <= 0:
            raise ValueError("write_ports_per_bank must be positive")


@dataclass(frozen=True)
class MemoryAccess:
    """One scheduled SRAM-bank access."""

    start_cycle: int
    cycles: int
    bank: int
    offset: int
    size_bytes: int
    kind: AccessKind
    op_name: str = ""
    micro_op: str = ""
    instruction_idx: int | None = None

    @property
    def end_cycle(self) -> int:
        return self.start_cycle + self.cycles

    def active_cycles(self) -> range:
        return range(self.start_cycle, self.end_cycle)


@dataclass(frozen=True)
class MemoryConflict:
    """A concrete memory conflict or invalid access."""

    conflict_kind: ConflictKind
    cycle: int | None
    bank: int | None
    message: str
    accesses: tuple[MemoryAccess, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MemoryCheckResult:
    """Result returned by the bank conflict checker."""

    accesses: tuple[MemoryAccess, ...]
    conflicts: tuple[MemoryConflict, ...]

    @property
    def ok(self) -> bool:
        return not self.conflicts

    @property
    def total_accesses(self) -> int:
        return len(self.accesses)


def _validate_access(access: MemoryAccess, config: MemoryBankConfig) -> list[MemoryConflict]:
    conflicts: list[MemoryConflict] = []

    if access.kind not in ("read", "write"):
        conflicts.append(
            MemoryConflict(
                conflict_kind="invalid_access",
                cycle=None,
                bank=access.bank,
                message=f"Invalid access kind: {access.kind!r}",
                accesses=(access,),
            )
        )

    if access.start_cycle < 0 or access.cycles <= 0 or access.offset < 0 or access.size_bytes <= 0:
        conflicts.append(
            MemoryConflict(
                conflict_kind="invalid_access",
                cycle=None,
                bank=access.bank,
                message=(
                    "Access must have non-negative start_cycle/offset and "
                    "positive cycles/size_bytes"
                ),
                accesses=(access,),
            )
        )

    if not 0 <= access.bank < config.num_banks:
        conflicts.append(
            MemoryConflict(
                conflict_kind="out_of_bounds",
                cycle=None,
                bank=access.bank,
                message=f"Bank {access.bank} is outside 0..{config.num_banks - 1}",
                accesses=(access,),
            )
        )

    if access.offset + access.size_bytes > config.bank_size_bytes:
        conflicts.append(
            MemoryConflict(
                conflict_kind="out_of_bounds",
                cycle=None,
                bank=access.bank,
                message=(
                    f"Access [{access.offset}, {access.offset + access.size_bytes}) exceeds "
                    f"bank size {config.bank_size_bytes} bytes"
                ),
                accesses=(access,),
            )
        )

    return conflicts


def check_memory_bank_conflicts(
    accesses: Iterable[MemoryAccess],
    config: MemoryBankConfig | None = None,
) -> MemoryCheckResult:
    """Check a list of scheduled memory accesses for SRAM bank conflicts."""

    cfg = config or MemoryBankConfig()
    ordered_accesses = tuple(sorted(accesses, key=lambda a: (a.start_cycle, a.bank, a.offset, a.kind)))
    conflicts: list[MemoryConflict] = []

    for access in ordered_accesses:
        conflicts.extend(_validate_access(access, cfg))

    accesses_by_cycle_bank: dict[tuple[int, int], list[MemoryAccess]] = {}
    for access in ordered_accesses:
        if access.cycles <= 0:
            continue
        for cycle in access.active_cycles():
            accesses_by_cycle_bank.setdefault((cycle, access.bank), []).append(access)

    for (cycle, bank), cycle_accesses in sorted(accesses_by_cycle_bank.items()):
        reads = tuple(a for a in cycle_accesses if a.kind == "read")
        writes = tuple(a for a in cycle_accesses if a.kind == "write")

        if len(reads) > cfg.read_ports_per_bank:
            conflicts.append(
                MemoryConflict(
                    conflict_kind="read_port_conflict",
                    cycle=cycle,
                    bank=bank,
                    message=(
                        f"Bank {bank} has {len(reads)} reads in cycle {cycle}, "
                        f"but only {cfg.read_ports_per_bank} read port(s)"
                    ),
                    accesses=reads,
                )
            )

        if len(writes) > cfg.write_ports_per_bank:
            conflicts.append(
                MemoryConflict(
                    conflict_kind="write_port_conflict",
                    cycle=cycle,
                    bank=bank,
                    message=(
                        f"Bank {bank} has {len(writes)} writes in cycle {cycle}, "
                        f"but only {cfg.write_ports_per_bank} write port(s)"
                    ),
                    accesses=writes,
                )
            )

        if reads and writes and not cfg.allow_same_cycle_read_write:
            conflicts.append(
                MemoryConflict(
                    conflict_kind="read_write_conflict",
                    cycle=cycle,
                    bank=bank,
                    message=f"Bank {bank} has read/write activity in the same cycle {cycle}",
                    accesses=reads + writes,
                )
            )

    return MemoryCheckResult(accesses=ordered_accesses, conflicts=tuple(conflicts))
