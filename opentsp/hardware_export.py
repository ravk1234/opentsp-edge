from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .rtl_matmul_program import (
    BANK_A,
    BANK_B,
    Matmul4x4Program,
    MatmulTileProgram,
    RtlInstruction,
    TilePayload,
    generate_default_matmul_4x4_program,
    program_to_payload,
)


@dataclass(frozen=True)
class HardwareExportPaths:
    """Paths written by the hardware bundle exporter."""

    root_dir: Path
    manifest_json: Path
    expected_c_json: Path
    tile_dirs: tuple[Path, ...]


def _to_u8(value: int) -> int:
    if not -128 <= int(value) <= 127:
        raise ValueError(f"signed INT8 value out of range: {value}")
    return int(value) & 0xFF


def _from_u8(value: int) -> int:
    value = int(value) & 0xFF
    return value - 256 if value >= 128 else value


def pack_i8x4(values: Sequence[Sequence[int]]) -> int:
    """Pack a signed 2x2 INT8 tile into one 32-bit little-endian word.

    Byte order is deterministic and matches the RTL testbench loading order:

    bits  7:0   -> [0][0]
    bits 15:8   -> [0][1]
    bits 23:16  -> [1][0]
    bits 31:24  -> [1][1]
    """

    if len(values) != 2 or any(len(row) != 2 for row in values):
        raise ValueError("pack_i8x4 expects a 2x2 tile")
    flat = [int(values[0][0]), int(values[0][1]), int(values[1][0]), int(values[1][1])]
    word = 0
    for shift, value in enumerate(flat):
        word |= _to_u8(value) << (8 * shift)
    return word


def unpack_i8x4(word: int) -> list[list[int]]:
    """Unpack a 32-bit tile word back into signed 2x2 INT8 values."""

    word = int(word) & 0xFFFFFFFF
    flat = [_from_u8((word >> (8 * i)) & 0xFF) for i in range(4)]
    return [[flat[0], flat[1]], [flat[2], flat[3]]]


def hex32(value: int) -> str:
    return f"{int(value) & 0xFFFFFFFF:08x}"


def pack_instruction_word(instr: RtlInstruction) -> int:
    """Pack the current tiny RTL instruction fields into a 32-bit word.

    The current RTL controller still accepts opcode and cycle count as separate
    host signals. This packed format is a stable export format for future
    instruction-memory initialization:

    bits  7:0   opcode
    bits 23:8   cycles
    bits 31:24  reserved
    """

    opcode = int(instr.opcode)
    cycles = int(instr.cycles)
    if not 0 <= opcode <= 0xFF:
        raise ValueError(f"opcode must fit 8 bits, got {opcode}")
    if not 0 <= cycles <= 0xFFFF:
        raise ValueError(f"cycles must fit 16 bits, got {cycles}")
    return opcode | (cycles << 8)


def unpack_instruction_word(word: int) -> tuple[int, int]:
    """Return (opcode, cycles) from a packed instruction word."""

    word = int(word) & 0xFFFFFFFF
    opcode = word & 0xFF
    cycles = (word >> 8) & 0xFFFF
    return opcode, cycles


def _memory_words_for_bank(tile_program: MatmulTileProgram, bank: int) -> list[int]:
    data_tiles = [tile for tile in tile_program.data_tiles if tile.bank == bank]
    if not data_tiles:
        return []
    max_addr = max(tile.addr for tile in data_tiles)
    words = [0 for _ in range(max_addr + 1)]
    for tile in data_tiles:
        words[tile.addr] = pack_i8x4(tile.values)
    return words


def _tile_program_payload(tile_program: MatmulTileProgram) -> dict[str, Any]:
    instruction_words = [pack_instruction_word(instr) for instr in tile_program.instructions]
    a_memory_words = _memory_words_for_bank(tile_program, BANK_A)
    b_memory_words = _memory_words_for_bank(tile_program, BANK_B)
    return {
        "name": tile_program.name,
        "c_row": tile_program.c_row,
        "c_col": tile_program.c_col,
        "program_len": tile_program.program_len,
        "instruction_words_hex": [hex32(word) for word in instruction_words],
        "instructions": [
            {
                "pc": pc,
                "opcode": int(instr.opcode),
                "cycles": int(instr.cycles),
                "name": instr.name,
                "word_hex": hex32(instruction_words[pc]),
            }
            for pc, instr in enumerate(tile_program.instructions)
        ],
        "a_memory_words_hex": [hex32(word) for word in a_memory_words],
        "b_memory_words_hex": [hex32(word) for word in b_memory_words],
        "a_memory_tiles": [
            {"addr": tile.addr, "word_hex": hex32(pack_i8x4(tile.values)), "values": tile.values}
            for tile in tile_program.data_tiles
            if tile.bank == BANK_A
        ],
        "b_memory_tiles": [
            {"addr": tile.addr, "word_hex": hex32(pack_i8x4(tile.values)), "values": tile.values}
            for tile in tile_program.data_tiles
            if tile.bank == BANK_B
        ],
        "expected_c": tile_program.expected_c,
    }


def hardware_bundle_payload(program: Matmul4x4Program) -> dict[str, Any]:
    """Build a JSON-serializable hardware export bundle payload."""

    base_payload = program_to_payload(program)
    return {
        "kind": "opentsp.hardware_export.matmul_4x4.v1",
        "source_kind": base_payload["kind"],
        "tile_size": base_payload["tile_size"],
        "a_shape": base_payload["a_shape"],
        "b_shape": base_payload["b_shape"],
        "a": program.a,
        "b": program.b,
        "expected_c": program.expected_c,
        "encoding": {
            "instruction_word": "bits[7:0]=opcode, bits[23:8]=cycles, bits[31:24]=reserved",
            "tile_word": "bits[7:0]=00, bits[15:8]=01, bits[23:16]=10, bits[31:24]=11 signed-int8 two's-complement",
        },
        "tile_programs": [_tile_program_payload(tile) for tile in program.tile_programs],
    }


def _write_hex_lines(path: Path, words: Sequence[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(hex32(word) for word in words) + ("\n" if words else ""), encoding="utf-8")
    return path


def export_hardware_bundle(program: Matmul4x4Program, out_dir: str | Path) -> HardwareExportPaths:
    """Write a hardware-friendly export bundle for the 4x4 matmul workload.

    The bundle contains both a manifest JSON and per-output-tile hex files:

    - instructions.hex
    - a_memory.hex
    - b_memory.hex
    - expected_c.json
    """

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = hardware_bundle_payload(program)

    manifest_path = root / "manifest.json"
    expected_c_path = root / "expected_c.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    expected_c_path.write_text(json.dumps(program.expected_c, indent=2), encoding="utf-8")

    tile_dirs: list[Path] = []
    for tile_payload in payload["tile_programs"]:
        tile_dir = root / tile_payload["name"]
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_dirs.append(tile_dir)

        instruction_words = [int(word, 16) for word in tile_payload["instruction_words_hex"]]
        a_words = [int(word, 16) for word in tile_payload["a_memory_words_hex"]]
        b_words = [int(word, 16) for word in tile_payload["b_memory_words_hex"]]
        _write_hex_lines(tile_dir / "instructions.hex", instruction_words)
        _write_hex_lines(tile_dir / "a_memory.hex", a_words)
        _write_hex_lines(tile_dir / "b_memory.hex", b_words)
        (tile_dir / "expected_c.json").write_text(json.dumps(tile_payload["expected_c"], indent=2), encoding="utf-8")

    return HardwareExportPaths(
        root_dir=root,
        manifest_json=manifest_path,
        expected_c_json=expected_c_path,
        tile_dirs=tuple(tile_dirs),
    )


def export_default_matmul_4x4_hardware_bundle(out_dir: str | Path) -> HardwareExportPaths:
    return export_hardware_bundle(generate_default_matmul_4x4_program(), out_dir)
