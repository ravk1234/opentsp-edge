from __future__ import annotations

from pathlib import Path

from opentsp.hardware_export import export_default_matmul_4x4_hardware_bundle
from opentsp.hardware_loader import load_hardware_bundle, reconstruct_matrix_from_tile_outputs


def main() -> None:
    out_dir = Path("artifacts/hardware_bundle/matmul_4x4")
    export_default_matmul_4x4_hardware_bundle(out_dir)
    bundle = load_hardware_bundle(out_dir)

    print("Hardware bundle loader demo")
    print("-" * 80)
    print(f"Input: {bundle.root_dir}")
    print(f"Kind: {bundle.manifest['kind']}")
    print(f"Tile programs: {len(bundle.tile_programs)}")
    print(f"Expected C: {bundle.expected_c}")

    first = bundle.tile_programs[0]
    print("\nFirst loaded tile program")
    print("-" * 80)
    print(f"name: {first.name}")
    print(f"program_len: {first.program_len}")
    print(f"instructions: {[(i.opcode, i.cycles) for i in first.instructions]}")
    print(f"a_memory entries: {len(first.a_memory)}")
    print(f"b_memory entries: {len(first.b_memory)}")
    print(f"expected_c: {first.expected_c}")

    reconstructed = reconstruct_matrix_from_tile_outputs(
        bundle.tile_programs,
        {tile.name: tile.expected_c for tile in bundle.tile_programs},
    )
    assert reconstructed == bundle.expected_c
    assert len(bundle.tile_programs) == 4
    assert first.program_len == 7
    print("\nHardware bundle loader check: PASSED")


if __name__ == "__main__":
    main()
