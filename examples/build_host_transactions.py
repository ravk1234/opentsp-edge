from __future__ import annotations

from pathlib import Path

from opentsp.hardware_export import export_default_matmul_4x4_hardware_bundle
from opentsp.hardware_loader import load_hardware_bundle
from opentsp.host_driver import build_bundle_host_writes, summarize_host_writes


def main() -> None:
    out_dir = Path("artifacts/hardware_bundle/matmul_4x4")
    export_default_matmul_4x4_hardware_bundle(out_dir)
    bundle = load_hardware_bundle(out_dir)
    transactions_by_tile = build_bundle_host_writes(bundle)

    total_writes = sum(len(writes) for writes in transactions_by_tile.values())
    first_tile = bundle.tile_programs[0]
    first_writes = transactions_by_tile[first_tile.name]
    summary = summarize_host_writes(first_writes)

    print("Host transaction build demo")
    print("-" * 80)
    print(f"Input bundle: {bundle.root_dir}")
    print(f"Tile programs: {len(bundle.tile_programs)}")
    print(f"Total host writes: {total_writes}")
    print(f"First tile: {first_tile.name}")
    print(f"First tile host writes: {len(first_writes)}")
    print(f"First tile summary: {summary}")

    print("\nFirst 20 host writes")
    print("-" * 80)
    for idx, write in enumerate(first_writes[:20]):
        print(
            f"{idx:03d} {write.name:<14} addr=0x{write.addr:02x} "
            f"value=0x{write.value:08x} {write.description}"
        )

    assert len(bundle.tile_programs) == 4
    assert total_writes > 0
    assert first_writes[0].name == "CONTROL"
    assert first_writes[0].value == 0x2
    assert first_writes[-1].name == "CONTROL"
    assert first_writes[-1].value == 0x1
    assert summary["INSTR_WORD"] == first_tile.program_len
    print("\nHost transaction build check: PASSED")


if __name__ == "__main__":
    main()
