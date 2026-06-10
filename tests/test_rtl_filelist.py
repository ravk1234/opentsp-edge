# SPDX-License-Identifier: Apache-2.0
"""Tests for RTL filelist validation."""

from pathlib import Path

from scripts.rtl_filelist import read_rtl_filelist, validate_rtl_filelist


REQUIRED_SOURCES = (
    "rtl/instruction_controller.sv",
    "rtl/systolic_tile_2x2.sv",
    "rtl/tile_engine_mem_2x2.sv",
    "rtl/host_tile_engine_2x2.sv",
    "rtl/axi_lite_host_regs.sv",
    "rtl/axi_lite_tile_engine_2x2.sv",
    "fpga/generic_axi/opentsp_axi_top_sim.sv",
)


def test_generic_axi_filelist_contains_required_sources():
    filelist = read_rtl_filelist("fpga/generic_axi/filelist.f")
    sources = {source.as_posix() for source in filelist.sources}

    for required in REQUIRED_SOURCES:
        assert required in sources


def test_generic_axi_filelist_has_no_duplicates():
    filelist = read_rtl_filelist("fpga/generic_axi/filelist.f")

    assert len(filelist.sources) == len(filelist.unique_sources)


def test_generic_axi_filelist_sources_exist():
    filelist = read_rtl_filelist("fpga/generic_axi/filelist.f")
    errors = validate_rtl_filelist(
        filelist,
        repo_root=".",
        required_sources=REQUIRED_SOURCES,
    )

    assert errors == []


def test_validate_rtl_filelist_reports_missing_source(tmp_path: Path):
    filelist_path = tmp_path / "bad_filelist.f"
    filelist_path.write_text("rtl/does_not_exist.sv\n", encoding="utf-8")

    filelist = read_rtl_filelist(filelist_path)
    errors = validate_rtl_filelist(filelist, repo_root=".")

    assert any("missing RTL source" in error for error in errors)


def test_validate_rtl_filelist_reports_duplicate_source(tmp_path: Path):
    filelist_path = tmp_path / "dup_filelist.f"
    filelist_path.write_text(
        "rtl/instruction_controller.sv\nrtl/instruction_controller.sv\n",
        encoding="utf-8",
    )

    filelist = read_rtl_filelist(filelist_path)
    errors = validate_rtl_filelist(filelist, repo_root=".")

    assert any("duplicate RTL source" in error for error in errors)