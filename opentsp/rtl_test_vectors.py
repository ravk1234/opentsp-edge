from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .rtl_reference import systolic_tile_2x2_sequence


@dataclass(frozen=True)
class SystolicTileVector:
    """Serializable test vector for rtl/systolic_tile_2x2.sv.

    Each vector is a deterministic sequence of 2x2 signed INT8 A/B tile pairs.
    The RTL tile should clear its accumulator before the sequence, consume each
    tile pair for one valid cycle, and then produce `expected_c`.
    """

    name: str
    a_tiles: list[list[list[int]]]
    b_tiles: list[list[list[int]]]
    expected_c: list[list[int]]
    description: str = ""

    @property
    def k_tiles(self) -> int:
        return len(self.a_tiles)


def _to_int_list_tile(tile: Sequence[Sequence[int]] | np.ndarray) -> list[list[int]]:
    arr = np.asarray(tile, dtype=np.int64)
    if arr.shape != (2, 2):
        raise ValueError(f"expected a 2x2 tile, got {arr.shape}")
    if int(arr.min()) < -128 or int(arr.max()) > 127:
        raise ValueError("tile values must fit signed INT8 range")
    return [[int(v) for v in row] for row in arr.tolist()]


def make_systolic_tile_vector(
    name: str,
    a_tiles: Iterable[Sequence[Sequence[int]] | np.ndarray],
    b_tiles: Iterable[Sequence[Sequence[int]] | np.ndarray],
    *,
    description: str = "",
) -> SystolicTileVector:
    """Build one JSON-serializable RTL vector from A/B tile sequences."""

    a_list = [_to_int_list_tile(tile) for tile in a_tiles]
    b_list = [_to_int_list_tile(tile) for tile in b_tiles]
    if len(a_list) != len(b_list):
        raise ValueError(f"a_tiles and b_tiles must have same length, got {len(a_list)} and {len(b_list)}")
    if not a_list:
        raise ValueError("at least one tile pair is required")

    expected = systolic_tile_2x2_sequence(
        [(np.asarray(a, dtype=np.int8), np.asarray(b, dtype=np.int8)) for a, b in zip(a_list, b_list)]
    )
    return SystolicTileVector(
        name=name,
        a_tiles=a_list,
        b_tiles=b_list,
        expected_c=[[int(v) for v in row] for row in expected.tolist()],
        description=description,
    )


def generate_systolic_tile_vectors(*, seed: int = 1234, random_count: int = 8) -> list[SystolicTileVector]:
    """Generate deterministic edge-case and random RTL vectors.

    These vectors are intentionally small so they can be used both by ordinary
    Python tests and cocotb RTL simulations.
    """

    vectors: list[SystolicTileVector] = [
        make_systolic_tile_vector(
            "identity_positive",
            a_tiles=[[[1, 2], [3, 4]]],
            b_tiles=[[[1, 0], [0, 1]]],
            description="One tile multiplied by identity.",
        ),
        make_systolic_tile_vector(
            "signed_mixed_values",
            a_tiles=[[[5, -2], [-7, 3]]],
            b_tiles=[[[-4, 6], [8, -1]]],
            description="Single tile with mixed positive and negative INT8 values.",
        ),
        make_systolic_tile_vector(
            "accumulate_two_k_tiles",
            a_tiles=[[[1, 2], [3, 4]], [[-1, 5], [6, -2]]],
            b_tiles=[[[7, 8], [9, 10]], [[2, -3], [4, 1]]],
            description="Two valid cycles accumulate two K tiles.",
        ),
        make_systolic_tile_vector(
            "edge_int8_values",
            a_tiles=[[[127, -128], [-1, 1]]],
            b_tiles=[[[1, -1], [2, -2]]],
            description="Uses signed INT8 edge values without overflowing INT32.",
        ),
    ]

    rng = np.random.default_rng(seed)
    for idx in range(random_count):
        k_tiles = int(rng.integers(1, 5))
        a_tiles = [rng.integers(-16, 17, size=(2, 2), dtype=np.int16).astype(np.int8) for _ in range(k_tiles)]
        b_tiles = [rng.integers(-16, 17, size=(2, 2), dtype=np.int16).astype(np.int8) for _ in range(k_tiles)]
        vectors.append(
            make_systolic_tile_vector(
                f"random_sequence_{idx:02d}",
                a_tiles=a_tiles,
                b_tiles=b_tiles,
                description=f"Deterministic random sequence with {k_tiles} K tile(s).",
            )
        )
    return vectors


def vectors_to_payload(vectors: Sequence[SystolicTileVector]) -> dict[str, Any]:
    return {
        "format": "opentsp.systolic_tile_2x2.vectors.v1",
        "tile_shape": [2, 2],
        "input_dtype": "int8",
        "accumulator_dtype": "int32",
        "vectors": [asdict(v) for v in vectors],
    }


def save_systolic_tile_vectors(path: str | Path, vectors: Sequence[SystolicTileVector]) -> Path:
    """Write deterministic RTL vectors to JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = vectors_to_payload(vectors)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_systolic_tile_vectors(path: str | Path) -> list[SystolicTileVector]:
    """Load vectors written by `save_systolic_tile_vectors`."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "opentsp.systolic_tile_2x2.vectors.v1":
        raise ValueError("unsupported vector format")
    vectors = []
    for item in payload["vectors"]:
        vectors.append(
            SystolicTileVector(
                name=str(item["name"]),
                a_tiles=item["a_tiles"],
                b_tiles=item["b_tiles"],
                expected_c=item["expected_c"],
                description=str(item.get("description", "")),
            )
        )
    return vectors
