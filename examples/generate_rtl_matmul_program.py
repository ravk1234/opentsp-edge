from __future__ import annotations

from pathlib import Path

from opentsp.rtl_matmul_program import generate_default_matmul_4x4_program, save_matmul_4x4_program


def main() -> None:
    program = generate_default_matmul_4x4_program()
    out_path = save_matmul_4x4_program(Path("artifacts") / "rtl_programs" / "matmul_4x4_program.json", program)

    print("RTL 4x4 matmul program demo")
    print("-" * 80)
    print("A shape: 4x4")
    print("B shape: 4x4")
    print("Tile size: 2x2")
    print(f"Output: {out_path}")
    print(f"Tile programs: {program.tile_count}")
    print(f"Total instructions: {program.total_instructions}")
    print("Expected C:")
    for row in program.expected_c:
        print(f"  {row}")
    print("Per-output-tile programs")
    print("-" * 80)
    for tile_program in program.tile_programs:
        print(
            f"{tile_program.name:<6} C[{tile_program.c_row}:{tile_program.c_row + 2}, "
            f"{tile_program.c_col}:{tile_program.c_col + 2}] "
            f"instructions={tile_program.program_len:<3} data_tiles={len(tile_program.data_tiles):<3} "
            f"expected={tile_program.expected_c}"
        )
    print("Program generation check: PASSED")


if __name__ == "__main__":
    main()
