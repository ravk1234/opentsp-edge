from __future__ import annotations

from pathlib import Path

from opentsp.cpp_host_runner import export_default_cpp_host_runner


def main() -> None:
    out_dir = Path("artifacts/cpp_host_runner/matmul_4x4")
    result = export_default_cpp_host_runner(out_dir)

    print("C++ host executable runner export demo")
    print("-" * 80)
    print(f"Output: {out_dir}")
    print(f"C file: {result.c_path}")
    print(f"Header file: {result.h_path}")
    print(f"C++ host sim: {result.cpp_path}")
    print(f"Makefile: {result.makefile_path}")
    print(f"Tile programs: {result.tile_program_count}")
    print(f"Total host writes: {result.total_writes}")

    cpp_text = result.cpp_path.read_text(encoding="utf-8")
    print("\nGenerated C++ preview")
    print("-" * 80)
    for line in cpp_text.splitlines()[:28]:
        print(line)

    assert result.cpp_path.exists()
    assert result.makefile_path.exists()
    assert "opentsp_run_matmul_4x4" in cpp_text
    assert "OpenTSP C++ host simulation: PASS" in cpp_text
    print("\nC++ host runner export check: PASSED")


if __name__ == "__main__":
    main()
