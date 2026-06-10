# SPDX-License-Identifier: Apache-2.0
"""Tests for standalone Verilator top build generation."""

from pathlib import Path
import json

from scripts.verilator_top_build import (
    create_verilator_top_build,
    generate_main_cpp,
    read_filelist,
)


def test_read_filelist_contains_axi_top_source():
    sources = read_filelist("fpga/generic_axi/filelist.f")
    source_strings = {source.as_posix() for source in sources}

    assert "fpga/generic_axi/opentsp_axi_top_sim.sv" in source_strings


def test_generate_main_cpp_references_verilated_top_header():
    text = generate_main_cpp("opentsp_axi_top_sim")

    assert '#include "Vopentsp_axi_top_sim.h"' in text
    assert "OpenTSP standalone Verilator top smoke test: PASS" in text
    assert "rst_ni" in text
    assert "clk_i" in text


def test_create_verilator_top_build_writes_artifacts(tmp_path: Path):
    out_dir = tmp_path / "verilator_top"

    build = create_verilator_top_build(output_dir=out_dir)

    assert build.main_cpp.exists()
    assert build.build_script.exists()
    assert build.metadata_json.exists()


def test_verilator_command_contains_expected_flags(tmp_path: Path):
    build = create_verilator_top_build(output_dir=tmp_path / "verilator_top")
    command = build.verilator_command()

    assert command[0] == "verilator"
    assert "--top-module" in command
    assert "opentsp_axi_top_sim" in command
    assert "--trace" in command
    assert build.main_cpp.as_posix() in command


def test_metadata_matches_generated_build(tmp_path: Path):
    build = create_verilator_top_build(output_dir=tmp_path / "verilator_top")
    metadata = json.loads(build.metadata_json.read_text(encoding="utf-8"))

    assert metadata["kind"] == "opentsp.verilator_top_build.v1"
    assert metadata["top_module"] == "opentsp_axi_top_sim"
    assert metadata["binary_name"] == "opentsp_axi_top_sim"
    assert metadata["sources"] == [source.as_posix() for source in build.sources]
