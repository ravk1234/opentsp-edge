from __future__ import annotations

import json

import numpy as np

from opentsp.rtl_reference import systolic_tile_2x2_sequence
from opentsp.rtl_test_vectors import (
    generate_systolic_tile_vectors,
    load_systolic_tile_vectors,
    make_systolic_tile_vector,
    save_systolic_tile_vectors,
    vectors_to_payload,
)


def test_make_vector_computes_expected_c() -> None:
    vector = make_systolic_tile_vector(
        "manual",
        a_tiles=[[[1, 2], [3, 4]], [[-1, 0], [5, -2]]],
        b_tiles=[[[5, 6], [7, 8]], [[2, -3], [4, 1]]],
    )
    expected = systolic_tile_2x2_sequence(
        [
            (np.array([[1, 2], [3, 4]], dtype=np.int8), np.array([[5, 6], [7, 8]], dtype=np.int8)),
            (np.array([[-1, 0], [5, -2]], dtype=np.int8), np.array([[2, -3], [4, 1]], dtype=np.int8)),
        ]
    )
    assert vector.expected_c == expected.tolist()
    assert vector.k_tiles == 2


def test_generated_vectors_are_deterministic() -> None:
    first = generate_systolic_tile_vectors(seed=2026, random_count=5)
    second = generate_systolic_tile_vectors(seed=2026, random_count=5)
    assert vectors_to_payload(first) == vectors_to_payload(second)


def test_generated_vectors_match_python_contract() -> None:
    vectors = generate_systolic_tile_vectors(seed=7, random_count=6)
    for vector in vectors:
        expected = systolic_tile_2x2_sequence(
            [
                (np.asarray(a, dtype=np.int8), np.asarray(b, dtype=np.int8))
                for a, b in zip(vector.a_tiles, vector.b_tiles)
            ]
        )
        assert vector.expected_c == expected.tolist(), vector.name


def test_vector_payload_is_json_serializable() -> None:
    payload = vectors_to_payload(generate_systolic_tile_vectors(seed=1, random_count=2))
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["format"] == "opentsp.systolic_tile_2x2.vectors.v1"
    assert decoded["tile_shape"] == [2, 2]
    assert len(decoded["vectors"]) == 6


def test_vector_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "vectors.json"
    vectors = generate_systolic_tile_vectors(seed=42, random_count=3)
    save_systolic_tile_vectors(path, vectors)
    loaded = load_systolic_tile_vectors(path)
    assert vectors_to_payload(loaded) == vectors_to_payload(vectors)
