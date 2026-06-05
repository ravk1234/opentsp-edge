from __future__ import annotations

import json

import numpy as np
import pytest

from opentsp.rtl_matmul_program import (
    BANK_A,
    BANK_B,
    OP_LOAD_A,
    OP_LOAD_B,
    OP_MAC_TILE,
    OP_STORE_C,
    generate_default_matmul_4x4_program,
    make_matmul_4x4_program,
    program_to_payload,
    reconstruct_c_from_tile_programs,
    save_matmul_4x4_program,
)


def test_default_program_matches_direct_int32_matmul() -> None:
    program = generate_default_matmul_4x4_program()
    a = np.asarray(program.a, dtype=np.int32)
    b = np.asarray(program.b, dtype=np.int32)
    expected = (a @ b).astype(np.int32).tolist()
    assert program.expected_c == expected
    assert reconstruct_c_from_tile_programs(program) == expected


def test_program_contains_four_output_tile_programs() -> None:
    program = generate_default_matmul_4x4_program()
    assert program.tile_count == 4
    assert program.total_instructions == 28
    assert {(p.c_row, p.c_col) for p in program.tile_programs} == {(0, 0), (0, 2), (2, 0), (2, 2)}


@pytest.mark.parametrize("tile_program", generate_default_matmul_4x4_program().tile_programs)
def test_each_tile_program_has_expected_load_mac_store_shape(tile_program) -> None:
    assert [instr.opcode for instr in tile_program.instructions] == [
        OP_LOAD_A,
        OP_LOAD_B,
        OP_MAC_TILE,
        OP_LOAD_A,
        OP_LOAD_B,
        OP_MAC_TILE,
        OP_STORE_C,
    ]
    assert [instr.cycles for instr in tile_program.instructions] == [1, 1, 1, 1, 1, 1, 1]
    assert {(d.bank, d.addr) for d in tile_program.data_tiles} == {
        (BANK_A, 0),
        (BANK_B, 1),
        (BANK_A, 3),
        (BANK_B, 4),
    }


def test_custom_program_validates_shape_and_int8_range() -> None:
    with pytest.raises(ValueError):
        make_matmul_4x4_program(np.zeros((2, 4), dtype=np.int8), np.zeros((4, 4), dtype=np.int8))
    with pytest.raises(ValueError):
        make_matmul_4x4_program(np.full((4, 4), 200, dtype=np.int16), np.zeros((4, 4), dtype=np.int8))


def test_payload_and_save_round_trip(tmp_path) -> None:
    program = generate_default_matmul_4x4_program()
    payload = program_to_payload(program)
    assert payload["kind"] == "opentsp.rtl_matmul_4x4_program.v1"
    assert payload["expected_c"] == program.expected_c
    assert len(payload["tile_programs"]) == 4

    out_path = save_matmul_4x4_program(tmp_path / "program.json", program)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == payload
