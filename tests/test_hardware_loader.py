from __future__ import annotations

import json

import pytest

from opentsp.hardware_export import export_default_matmul_4x4_hardware_bundle
from opentsp.hardware_loader import load_hardware_bundle, reconstruct_matrix_from_tile_outputs


def test_loader_reads_exported_bundle_files(tmp_path) -> None:
    paths = export_default_matmul_4x4_hardware_bundle(tmp_path / "bundle")
    bundle = load_hardware_bundle(paths.root_dir)
    assert bundle.manifest["kind"] == "opentsp.hardware_export.matmul_4x4.v1"
    assert len(bundle.tile_programs) == 4
    assert bundle.expected_c == ((26, -16, -5, -7), (-11, 35, -4, 28), (-1, -15, 25, -25), (-45, 52, -3, 27))


def test_loader_decodes_instruction_words_and_tile_memory(tmp_path) -> None:
    paths = export_default_matmul_4x4_hardware_bundle(tmp_path / "bundle")
    bundle = load_hardware_bundle(paths.root_dir)
    c00 = bundle.tile_programs[0]
    assert c00.name == "c00"
    assert c00.program_len == 7
    assert [(instr.opcode, instr.cycles) for instr in c00.instructions] == [(1, 1), (2, 1), (3, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    assert c00.a_memory[0] == ((1, -2), (4, 5))
    assert c00.b_memory[1] == ((2, 0), (-3, 5))
    assert c00.expected_c == ((26, -16), (-11, 35))


def test_reconstruct_matrix_from_loaded_tile_outputs(tmp_path) -> None:
    paths = export_default_matmul_4x4_hardware_bundle(tmp_path / "bundle")
    bundle = load_hardware_bundle(paths.root_dir)
    reconstructed = reconstruct_matrix_from_tile_outputs(
        bundle.tile_programs,
        {tile.name: tile.expected_c for tile in bundle.tile_programs},
    )
    assert reconstructed == bundle.expected_c


def test_loader_rejects_missing_manifest(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_hardware_bundle(tmp_path / "missing")


def test_loader_rejects_unknown_kind(tmp_path) -> None:
    root = tmp_path / "bad"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"kind": "unknown"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_hardware_bundle(root)
