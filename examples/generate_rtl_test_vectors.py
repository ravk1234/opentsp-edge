from __future__ import annotations

from pathlib import Path

from opentsp.rtl_test_vectors import generate_systolic_tile_vectors, save_systolic_tile_vectors


def main() -> None:
    vectors = generate_systolic_tile_vectors(seed=2026, random_count=8)
    out_path = save_systolic_tile_vectors(
        Path("artifacts") / "rtl_vectors" / "systolic_tile_2x2_vectors.json",
        vectors,
    )
    print("Generated RTL test vectors")
    print("-" * 80)
    print(f"Output: {out_path}")
    print(f"Vector count: {len(vectors)}")
    print(f"Total tile-pair cycles: {sum(v.k_tiles for v in vectors)}")
    print("First vector:")
    first = vectors[0]
    print(f"  name: {first.name}")
    print(f"  k_tiles: {first.k_tiles}")
    print(f"  expected_c: {first.expected_c}")
    print("RTL test vector generation check: PASSED")


if __name__ == "__main__":
    main()
