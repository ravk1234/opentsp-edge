from __future__ import annotations

from pathlib import Path

from opentsp.c_host_runner import export_default_c_host_runner


def main() -> None:
    out_dir = Path("artifacts/host_runner/matmul_4x4")
    result = export_default_c_host_runner(out_dir)

    print("C host runner export demo")
    print("-" * 80)
    print(f"Output: {out_dir}")
    print(f"C file: {result.c_path}")
    print(f"Header file: {result.h_path}")
    print(f"Tile programs: {result.tile_program_count}")
    print(f"Total host writes: {result.total_writes}")

    c_text = result.c_path.read_text(encoding="utf-8")
    h_text = result.h_path.read_text(encoding="utf-8")
    print("\nGenerated C preview")
    print("-" * 80)
    for line in c_text.splitlines()[:24]:
        print(line)

    assert "OPENTSP_MATMUL_4X4_WRITES" in c_text
    assert "opentsp_run_matmul_4x4" in c_text
    assert "OPENTSP_REG_CONTROL" in h_text
    assert result.total_writes > 0
    print("\nC host runner export check: PASSED")


if __name__ == "__main__":
    main()
