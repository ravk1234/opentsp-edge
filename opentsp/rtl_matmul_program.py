from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OP_LOAD_A = 1
OP_LOAD_B = 2
OP_MAC_TILE = 3
OP_STORE_C = 4

BANK_A = 0
BANK_B = 1

TILE_SIZE = 2


@dataclass(frozen=True)
class RtlInstruction:
    """One tiny instruction for the memory-backed 2x2 RTL tile engine."""

    opcode: int
    cycles: int = 1
    name: str = ""


@dataclass(frozen=True)
class TilePayload:
    """A signed INT8 2x2 tile written into the RTL A or B memory."""

    bank: int
    addr: int
    values: list[list[int]]


@dataclass(frozen=True)
class MatmulTileProgram:
    """One RTL program that computes a single 2x2 output C tile.

    The current `tile_engine_mem_2x2` stores one C tile at a time, so a full 4x4
    matmul is represented as four independent tile programs: C00, C01, C10, C11.
    Each program runs two K tiles for a 4-wide inner dimension:

        LOAD_A, LOAD_B, MAC_TILE, LOAD_A, LOAD_B, MAC_TILE, STORE_C
    """

    name: str
    c_row: int
    c_col: int
    instructions: list[RtlInstruction]
    data_tiles: list[TilePayload]
    expected_c: list[list[int]]

    @property
    def program_len(self) -> int:
        return len(self.instructions)


@dataclass(frozen=True)
class Matmul4x4Program:
    """Serializable 4x4 matmul workload for the 2x2 RTL tile engine."""

    a: list[list[int]]
    b: list[list[int]]
    expected_c: list[list[int]]
    tile_programs: list[MatmulTileProgram]

    @property
    def tile_count(self) -> int:
        return len(self.tile_programs)

    @property
    def total_instructions(self) -> int:
        return sum(p.program_len for p in self.tile_programs)


def _validate_int8_matrix(name: str, value: Sequence[Sequence[int]] | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must contain integer values")
    if int(arr.min()) < -128 or int(arr.max()) > 127:
        raise ValueError(f"{name} values must fit signed INT8 range")
    return arr.astype(np.int8)


def _tile_to_list(tile: np.ndarray) -> list[list[int]]:
    if tile.shape != (2, 2):
        raise ValueError(f"expected 2x2 tile, got {tile.shape}")
    return [[int(v) for v in row] for row in tile.astype(np.int64).tolist()]


def _int32_matrix_to_list(matrix: np.ndarray) -> list[list[int]]:
    return [[int(v) for v in row] for row in matrix.astype(np.int64).tolist()]


def make_matmul_4x4_program(
    a: Sequence[Sequence[int]] | np.ndarray,
    b: Sequence[Sequence[int]] | np.ndarray,
) -> Matmul4x4Program:
    """Build four 2x2 RTL tile-engine programs for a signed INT8 4x4 matmul."""

    a_arr = _validate_int8_matrix("a", a, (4, 4))
    b_arr = _validate_int8_matrix("b", b, (4, 4))
    expected = (a_arr.astype(np.int32) @ b_arr.astype(np.int32)).astype(np.int32)

    instructions = [
        RtlInstruction(OP_LOAD_A, 1, "LOAD_A_k0"),
        RtlInstruction(OP_LOAD_B, 1, "LOAD_B_k0"),
        RtlInstruction(OP_MAC_TILE, 1, "MAC_TILE_k0"),
        RtlInstruction(OP_LOAD_A, 1, "LOAD_A_k1"),
        RtlInstruction(OP_LOAD_B, 1, "LOAD_B_k1"),
        RtlInstruction(OP_MAC_TILE, 1, "MAC_TILE_k1"),
        RtlInstruction(OP_STORE_C, 1, "STORE_C"),
    ]

    tile_programs: list[MatmulTileProgram] = []
    for c_row in range(0, 4, TILE_SIZE):
        for c_col in range(0, 4, TILE_SIZE):
            data_tiles: list[TilePayload] = []
            for k0, load_a_pc, load_b_pc in [(0, 0, 1), (2, 3, 4)]:
                a_tile = a_arr[c_row : c_row + 2, k0 : k0 + 2]
                b_tile = b_arr[k0 : k0 + 2, c_col : c_col + 2]
                data_tiles.append(TilePayload(BANK_A, load_a_pc, _tile_to_list(a_tile)))
                data_tiles.append(TilePayload(BANK_B, load_b_pc, _tile_to_list(b_tile)))

            expected_tile = expected[c_row : c_row + 2, c_col : c_col + 2]
            tile_programs.append(
                MatmulTileProgram(
                    name=f"c{c_row // 2}{c_col // 2}",
                    c_row=c_row,
                    c_col=c_col,
                    instructions=list(instructions),
                    data_tiles=data_tiles,
                    expected_c=_tile_to_list(expected_tile),
                )
            )

    return Matmul4x4Program(
        a=_tile_to_list(a_arr[:2, :2])[:0] + [[int(v) for v in row] for row in a_arr.astype(np.int64).tolist()],
        b=[[int(v) for v in row] for row in b_arr.astype(np.int64).tolist()],
        expected_c=_int32_matrix_to_list(expected),
        tile_programs=tile_programs,
    )


def generate_default_matmul_4x4_program() -> Matmul4x4Program:
    """Generate a deterministic signed INT8 4x4 matmul workload."""

    a = np.array(
        [
            [1, -2, 3, 0],
            [4, 5, -1, 2],
            [-3, 1, 2, -4],
            [0, 6, -5, 3],
        ],
        dtype=np.int8,
    )
    b = np.array(
        [
            [2, 0, -1, 4],
            [-3, 5, 2, 1],
            [6, -2, 0, -3],
            [1, 4, -5, 2],
        ],
        dtype=np.int8,
    )
    return make_matmul_4x4_program(a, b)


def reconstruct_c_from_tile_programs(program: Matmul4x4Program) -> list[list[int]]:
    """Rebuild the full 4x4 C matrix from the four expected 2x2 output tiles."""

    c = np.zeros((4, 4), dtype=np.int32)
    for tile_program in program.tile_programs:
        r = tile_program.c_row
        col = tile_program.c_col
        c[r : r + 2, col : col + 2] = np.asarray(tile_program.expected_c, dtype=np.int32)
    return _int32_matrix_to_list(c)


def program_to_payload(program: Matmul4x4Program) -> dict[str, Any]:
    return {
        "kind": "opentsp.rtl_matmul_4x4_program.v1",
        "tile_size": TILE_SIZE,
        "a_shape": [4, 4],
        "b_shape": [4, 4],
        "a": program.a,
        "b": program.b,
        "expected_c": program.expected_c,
        "tile_programs": [
            {
                "name": tile.name,
                "c_row": tile.c_row,
                "c_col": tile.c_col,
                "program_len": tile.program_len,
                "instructions": [asdict(instr) for instr in tile.instructions],
                "data_tiles": [asdict(data) for data in tile.data_tiles],
                "expected_c": tile.expected_c,
            }
            for tile in program.tile_programs
        ],
    }


def save_matmul_4x4_program(path: str | Path, program: Matmul4x4Program) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(program_to_payload(program), indent=2), encoding="utf-8")
    return out_path
