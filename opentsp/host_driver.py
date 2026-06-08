from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .hardware_export import pack_i8x4
from .hardware_loader import LoadedHardwareBundle, LoadedTileProgram


ADDR_CONTROL = 0x00
ADDR_STATUS = 0x04
ADDR_PROGRAM_LEN = 0x08
ADDR_PC = 0x0C
ADDR_INSTR_ADDR = 0x10
ADDR_INSTR_WORD = 0x14
ADDR_DATA_ADDR = 0x18
ADDR_DATA_BANK = 0x1C
ADDR_DATA_WORD = 0x20
ADDR_C00 = 0x30
ADDR_C01 = 0x34
ADDR_C10 = 0x38
ADDR_C11 = 0x3C

CONTROL_START = 0x1
CONTROL_CLEAR = 0x2

BANK_A = 0
BANK_B = 1

STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1
STATUS_STORED_VALID = 1 << 2

REGISTER_NAMES = {
    ADDR_CONTROL: "CONTROL",
    ADDR_STATUS: "STATUS",
    ADDR_PROGRAM_LEN: "PROGRAM_LEN",
    ADDR_PC: "PC",
    ADDR_INSTR_ADDR: "INSTR_ADDR",
    ADDR_INSTR_WORD: "INSTR_WORD",
    ADDR_DATA_ADDR: "DATA_ADDR",
    ADDR_DATA_BANK: "DATA_BANK",
    ADDR_DATA_WORD: "DATA_WORD",
    ADDR_C00: "C00",
    ADDR_C01: "C01",
    ADDR_C10: "C10",
    ADDR_C11: "C11",
}


@dataclass(frozen=True)
class HostWrite:
    """One 32-bit host-register write for the simulation register interface."""

    addr: int
    value: int
    description: str = ""

    @property
    def name(self) -> str:
        return REGISTER_NAMES.get(int(self.addr), f"REG_0x{int(self.addr):02x}")


def _write(addr: int, value: int, description: str = "") -> HostWrite:
    return HostWrite(addr=int(addr), value=int(value) & 0xFFFFFFFF, description=description)


def build_tile_program_host_writes(
    tile_program: LoadedTileProgram,
    *,
    clear_first: bool = True,
    start_at_end: bool = True,
) -> tuple[HostWrite, ...]:
    """Convert one loaded 2x2 tile program into host register writes.

    The generated sequence matches ``rtl/host_tile_engine_2x2.sv``:

    - write instruction address + instruction word for each instruction
    - write A and B tile memory words through DATA_BANK/DATA_ADDR/DATA_WORD
    - write PROGRAM_LEN
    - write CONTROL.start
    """

    writes: list[HostWrite] = []

    if clear_first:
        writes.append(_write(ADDR_CONTROL, CONTROL_CLEAR, f"clear before {tile_program.name}"))

    for addr, instruction in enumerate(tile_program.instructions):
        writes.append(_write(ADDR_INSTR_ADDR, addr, f"{tile_program.name}: select instr[{addr}]"))
        writes.append(
            _write(
                ADDR_INSTR_WORD,
                int(instruction.word_hex, 16),
                f"{tile_program.name}: opcode={instruction.opcode} cycles={instruction.cycles}",
            )
        )

    for addr, tile in enumerate(tile_program.a_memory):
        writes.append(_write(ADDR_DATA_BANK, BANK_A, f"{tile_program.name}: select A memory"))
        writes.append(_write(ADDR_DATA_ADDR, addr, f"{tile_program.name}: select A[{addr}]"))
        writes.append(_write(ADDR_DATA_WORD, pack_i8x4(tile), f"{tile_program.name}: write A[{addr}]"))

    for addr, tile in enumerate(tile_program.b_memory):
        writes.append(_write(ADDR_DATA_BANK, BANK_B, f"{tile_program.name}: select B memory"))
        writes.append(_write(ADDR_DATA_ADDR, addr, f"{tile_program.name}: select B[{addr}]"))
        writes.append(_write(ADDR_DATA_WORD, pack_i8x4(tile), f"{tile_program.name}: write B[{addr}]"))

    writes.append(_write(ADDR_PROGRAM_LEN, tile_program.program_len, f"{tile_program.name}: program length"))

    if start_at_end:
        writes.append(_write(ADDR_CONTROL, CONTROL_START, f"start {tile_program.name}"))

    return tuple(writes)


def build_bundle_host_writes(bundle: LoadedHardwareBundle) -> dict[str, tuple[HostWrite, ...]]:
    """Build host writes for every tile program in a loaded hardware bundle."""

    return {tile.name: build_tile_program_host_writes(tile) for tile in bundle.tile_programs}


def summarize_host_writes(writes: tuple[HostWrite, ...]) -> dict[str, int]:
    """Return a stable count by register name for demo/test reporting."""

    counts = Counter(write.name for write in writes)
    return dict(sorted(counts.items()))
