# SPDX-License-Identifier: Apache-2.0
"""Check the OpenTSP generic AXI RTL filelist."""

from __future__ import annotations

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


def main() -> None:
    repo_root = Path(".")
    filelist_path = repo_root / "fpga" / "generic_axi" / "filelist.f"

    filelist = read_rtl_filelist(filelist_path)
    errors = validate_rtl_filelist(
        filelist,
        repo_root=repo_root,
        required_sources=REQUIRED_SOURCES,
    )

    print("OpenTSP RTL source check")
    print("-" * 80)
    print(f"Filelist: {filelist_path}")
    print(f"RTL source count: {len(filelist.sources)}")

    for index, source in enumerate(filelist.sources):
        print(f"{index:02d} {source.as_posix()}")

    if errors:
        print()
        print("Errors")
        print("-" * 80)
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print()
    print("RTL source check: PASSED")


if __name__ == "__main__":
    main()
