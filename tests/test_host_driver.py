from __future__ import annotations

from opentsp.hardware_export import export_default_matmul_4x4_hardware_bundle, pack_i8x4
from opentsp.hardware_loader import load_hardware_bundle
from opentsp.host_driver import (
    ADDR_CONTROL,
    ADDR_DATA_WORD,
    ADDR_INSTR_WORD,
    CONTROL_CLEAR,
    CONTROL_START,
    build_bundle_host_writes,
    build_tile_program_host_writes,
    summarize_host_writes,
)


def _bundle(tmp_path):
    out_dir = tmp_path / "bundle"
    export_default_matmul_4x4_hardware_bundle(out_dir)
    return load_hardware_bundle(out_dir)


def test_tile_program_writes_start_with_clear_and_end_with_start(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    tile = bundle.tile_programs[0]
    writes = build_tile_program_host_writes(tile)

    assert writes[0].addr == ADDR_CONTROL
    assert writes[0].value == CONTROL_CLEAR
    assert writes[-1].addr == ADDR_CONTROL
    assert writes[-1].value == CONTROL_START


def test_instruction_words_are_emitted_for_each_instruction(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    tile = bundle.tile_programs[0]
    writes = build_tile_program_host_writes(tile)

    instr_writes = [write for write in writes if write.addr == ADDR_INSTR_WORD]

    assert len(instr_writes) == tile.program_len
    assert [write.value for write in instr_writes] == [int(instr.word_hex, 16) for instr in tile.instructions]


def test_tile_memory_words_are_packed_deterministically(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    tile = bundle.tile_programs[0]
    writes = build_tile_program_host_writes(tile)

    data_words = [write.value for write in writes if write.addr == ADDR_DATA_WORD]
    expected = [pack_i8x4(tile_value) for tile_value in tile.a_memory] + [
        pack_i8x4(tile_value) for tile_value in tile.b_memory
    ]

    assert data_words == expected


def test_bundle_host_writes_are_built_for_all_output_tiles(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    writes_by_tile = build_bundle_host_writes(bundle)

    assert set(writes_by_tile) == {tile.name for tile in bundle.tile_programs}
    assert len(writes_by_tile) == 4
    assert all(writes for writes in writes_by_tile.values())


def test_summary_counts_register_writes(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    tile = bundle.tile_programs[0]
    summary = summarize_host_writes(build_tile_program_host_writes(tile))

    assert summary["INSTR_WORD"] == tile.program_len
    assert summary["DATA_WORD"] == len(tile.a_memory) + len(tile.b_memory)
    assert summary["CONTROL"] == 2
    assert "PROGRAM_LEN" in summary
