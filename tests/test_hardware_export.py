from __future__ import annotations

import json

from opentsp.hardware_export import (
    export_hardware_bundle,
    hardware_bundle_payload,
    pack_i8x4,
    pack_instruction_word,
    unpack_i8x4,
    unpack_instruction_word,
)
from opentsp.rtl_matmul_program import OP_LOAD_A, OP_MAC_TILE, RtlInstruction, generate_default_matmul_4x4_program


def test_pack_i8x4_roundtrips_signed_values() -> None:
    tile = [[1, -2], [127, -128]]
    word = pack_i8x4(tile)
    assert f"{word:08x}" == "807ffe01"
    assert unpack_i8x4(word) == tile


def test_pack_i8x4_rejects_invalid_shape() -> None:
    try:
        pack_i8x4([[1, 2, 3], [4, 5, 6]])
    except ValueError as exc:
        assert "2x2" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-2x2 tile")


def test_instruction_word_roundtrip() -> None:
    instr = RtlInstruction(opcode=OP_MAC_TILE, cycles=9, name="MAC_TILE")
    word = pack_instruction_word(instr)
    assert f"{word:08x}" == "00000903"
    assert unpack_instruction_word(word) == (OP_MAC_TILE, 9)


def test_hardware_bundle_payload_contains_four_tile_programs() -> None:
    program = generate_default_matmul_4x4_program()
    payload = hardware_bundle_payload(program)
    assert payload["kind"] == "opentsp.hardware_export.matmul_4x4.v1"
    assert len(payload["tile_programs"]) == 4
    first = payload["tile_programs"][0]
    assert first["name"] == "c00"
    assert first["instruction_words_hex"][0] == "00000101"
    assert first["instructions"][0]["opcode"] == OP_LOAD_A
    assert first["program_len"] == 7
    assert first["expected_c"] == [[26, -16], [-11, 35]]


def test_export_writes_manifest_hex_files_and_expected_outputs(tmp_path) -> None:
    program = generate_default_matmul_4x4_program()
    paths = export_hardware_bundle(program, tmp_path / "bundle")
    assert paths.manifest_json.exists()
    assert paths.expected_c_json.exists()
    assert len(paths.tile_dirs) == 4

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["expected_c"] == program.expected_c

    c00 = paths.root_dir / "c00"
    assert (c00 / "instructions.hex").read_text(encoding="utf-8").splitlines()[0] == "00000101"
    assert len((c00 / "instructions.hex").read_text(encoding="utf-8").splitlines()) == 7
    assert len((c00 / "a_memory.hex").read_text(encoding="utf-8").splitlines()) == 4
    assert len((c00 / "b_memory.hex").read_text(encoding="utf-8").splitlines()) == 5
    assert json.loads((c00 / "expected_c.json").read_text(encoding="utf-8")) == [[26, -16], [-11, 35]]
