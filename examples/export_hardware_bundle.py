from __future__ import annotations

import json
from pathlib import Path

from opentsp.hardware_export import export_default_matmul_4x4_hardware_bundle


def main() -> None:
    out_dir = Path("artifacts/hardware_bundle/matmul_4x4")
    paths = export_default_matmul_4x4_hardware_bundle(out_dir)
    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))

    print("Hardware bundle export demo")
    print("-" * 80)
    print(f"Output: {paths.root_dir}")
    print(f"Manifest: {paths.manifest_json}")
    print(f"Expected C: {paths.expected_c_json}")
    print(f"Tile program count: {len(paths.tile_dirs)}")
    print(f"Kind: {manifest['kind']}")

    first_tile = manifest["tile_programs"][0]
    print("\nFirst tile program")
    print("-" * 80)
    print(f"name: {first_tile['name']}")
    print(f"program_len: {first_tile['program_len']}")
    print(f"instruction_words_hex: {first_tile['instruction_words_hex']}")
    print(f"a_memory_words_hex: {first_tile['a_memory_words_hex']}")
    print(f"b_memory_words_hex: {first_tile['b_memory_words_hex']}")
    print(f"expected_c: {first_tile['expected_c']}")

    assert manifest["kind"] == "opentsp.hardware_export.matmul_4x4.v1"
    assert len(paths.tile_dirs) == 4
    assert first_tile["program_len"] == 7
    assert len(first_tile["instruction_words_hex"]) == 7
    assert (paths.tile_dirs[0] / "instructions.hex").exists()
    assert (paths.tile_dirs[0] / "a_memory.hex").exists()
    assert (paths.tile_dirs[0] / "b_memory.hex").exists()
    print("\nHardware bundle export check: PASSED")


if __name__ == "__main__":
    main()
