from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hardware_export import hex32, unpack_i8x4, unpack_instruction_word


@dataclass(frozen=True)
class LoadedInstruction:
    """Instruction decoded from an exported 32-bit instruction word."""

    opcode: int
    cycles: int
    word_hex: str


@dataclass(frozen=True)
class LoadedTileProgram:
    """One exported 2x2 output-tile program loaded from hex files."""

    name: str
    c_row: int
    c_col: int
    program_len: int
    instructions: tuple[LoadedInstruction, ...]
    a_memory: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    b_memory: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    expected_c: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class LoadedHardwareBundle:
    """A hardware bundle loaded from manifest/hex files."""

    root_dir: Path
    manifest: dict[str, Any]
    expected_c: tuple[tuple[int, ...], ...]
    tile_programs: tuple[LoadedTileProgram, ...]


def _read_hex_words(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(path)
    words: list[int] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            words.append(int(line, 16))
        except ValueError as exc:
            raise ValueError(f"invalid hex word in {path} line {line_no}: {line!r}") from exc
    return words


def _freeze_2x2(values: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(values) != 2 or any(len(row) != 2 for row in values):
        raise ValueError(f"expected 2x2 values, got {values!r}")
    return ((int(values[0][0]), int(values[0][1])), (int(values[1][0]), int(values[1][1])))


def _freeze_matrix(values: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in values)


def _decode_tile_memory(words: list[int]) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return tuple(_freeze_2x2(unpack_i8x4(word)) for word in words)


def load_hardware_bundle(root_dir: str | Path) -> LoadedHardwareBundle:
    """Load a hardware bundle exported by ``hardware_export.py``.

    The loader reads the manifest plus per-tile ``instructions.hex``,
    ``a_memory.hex``, ``b_memory.hex``, and ``expected_c.json`` files.
    """

    root = Path(root_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "opentsp.hardware_export.matmul_4x4.v1":
        raise ValueError(f"unsupported hardware bundle kind: {manifest.get('kind')!r}")

    expected_c_path = root / "expected_c.json"
    expected_c = _freeze_matrix(json.loads(expected_c_path.read_text(encoding="utf-8")))

    loaded_tiles: list[LoadedTileProgram] = []
    for tile_manifest in manifest["tile_programs"]:
        name = tile_manifest["name"]
        tile_dir = root / name
        instruction_words = _read_hex_words(tile_dir / "instructions.hex")
        instructions = tuple(
            LoadedInstruction(opcode=opcode, cycles=cycles, word_hex=hex32(word))
            for word in instruction_words
            for opcode, cycles in [unpack_instruction_word(word)]
        )
        a_memory = _decode_tile_memory(_read_hex_words(tile_dir / "a_memory.hex"))
        b_memory = _decode_tile_memory(_read_hex_words(tile_dir / "b_memory.hex"))
        expected_tile = _freeze_2x2(json.loads((tile_dir / "expected_c.json").read_text(encoding="utf-8")))
        program_len = int(tile_manifest["program_len"])
        if len(instructions) != program_len:
            raise ValueError(f"{name}: instruction count {len(instructions)} != program_len {program_len}")
        loaded_tiles.append(
            LoadedTileProgram(
                name=name,
                c_row=int(tile_manifest["c_row"]),
                c_col=int(tile_manifest["c_col"]),
                program_len=program_len,
                instructions=instructions,
                a_memory=a_memory,
                b_memory=b_memory,
                expected_c=expected_tile,
            )
        )

    return LoadedHardwareBundle(
        root_dir=root,
        manifest=manifest,
        expected_c=expected_c,
        tile_programs=tuple(loaded_tiles),
    )


def reconstruct_matrix_from_tile_outputs(
    tile_programs: tuple[LoadedTileProgram, ...],
    tile_outputs: dict[str, tuple[tuple[int, int], tuple[int, int]]],
    rows: int = 4,
    cols: int = 4,
) -> tuple[tuple[int, ...], ...]:
    """Reconstruct a full matrix from named 2x2 tile outputs."""

    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for tile in tile_programs:
        if tile.name not in tile_outputs:
            raise KeyError(f"missing output for tile program {tile.name}")
        out = tile_outputs[tile.name]
        r = tile.c_row
        c = tile.c_col
        matrix[r][c] = int(out[0][0])
        matrix[r][c + 1] = int(out[0][1])
        matrix[r + 1][c] = int(out[1][0])
        matrix[r + 1][c + 1] = int(out[1][1])
    return _freeze_matrix(matrix)
