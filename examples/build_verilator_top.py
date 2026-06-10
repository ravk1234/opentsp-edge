# SPDX-License-Identifier: Apache-2.0
"""Generate a standalone Verilator top-level build bundle."""

from __future__ import annotations

from scripts.verilator_top_build import create_verilator_top_build


def main() -> None:
    build = create_verilator_top_build()

    print("Standalone Verilator top build demo")
    print("-" * 80)
    print(f"Top module: {build.top_module}")
    print(f"Filelist: {build.filelist_path}")
    print(f"Output dir: {build.output_dir}")
    print(f"Source count: {len(build.sources)}")
    print(f"C++ harness: {build.main_cpp}")
    print(f"Build script: {build.build_script}")
    print(f"Metadata: {build.metadata_json}")
    print()
    print("First RTL sources")
    print("-" * 80)
    for index, source in enumerate(build.sources[:8]):
        print(f"{index:02d} {source.as_posix()}")
    print()
    print("Verilator command preview")
    print("-" * 80)
    print(" ".join(build.verilator_command()[:10]) + " ...")
    print()
    print("Standalone Verilator top build generation check: PASSED")


if __name__ == "__main__":
    main()
