from __future__ import annotations

from opentsp.c_host_runner import (
    export_c_host_runner,
    export_default_c_host_runner,
    flatten_bundle_host_writes,
    render_c_header,
    render_c_runner,
)
from opentsp.hardware_export import export_default_matmul_4x4_hardware_bundle


def _bundle_dir(tmp_path):
    bundle_dir = tmp_path / "hardware_bundle" / "matmul_4x4"
    export_default_matmul_4x4_hardware_bundle(bundle_dir)
    return bundle_dir


def test_flatten_bundle_host_writes_has_all_tile_programs(tmp_path) -> None:
    bundle_dir = _bundle_dir(tmp_path)
    writes = flatten_bundle_host_writes(bundle_dir)

    assert len(writes) == 176
    assert writes[0].name == "CONTROL"
    assert writes[-1].name == "CONTROL"


def test_render_c_header_contains_register_defines() -> None:
    text = render_c_header()

    assert "OPENTSP_REG_CONTROL" in text
    assert "OPENTSP_STATUS_DONE" in text
    assert "opentsp_run_matmul_4x4" in text
    assert "#include <stdint.h>" in text


def test_render_c_runner_contains_write_array_and_poll_loop(tmp_path) -> None:
    bundle_dir = _bundle_dir(tmp_path)
    writes = flatten_bundle_host_writes(bundle_dir)
    expected_c = ((26, -16, -5, -7), (-11, 35, -4, 28), (-1, -15, 25, -25), (-45, 52, -3, 27))
    text = render_c_runner(writes, expected_c=expected_c)

    assert "OPENTSP_MATMUL_4X4_WRITES[]" in text
    assert "OPENTSP_REG_STATUS" in text
    assert "OPENTSP_STATUS_DONE" in text
    assert "{26, -16, -5, -7}" in text


def test_export_c_host_runner_writes_c_and_header(tmp_path) -> None:
    bundle_dir = _bundle_dir(tmp_path)
    result = export_c_host_runner(bundle_dir, tmp_path / "host_runner")

    assert result.c_path.exists()
    assert result.h_path.exists()
    assert result.total_writes == 176
    assert result.tile_program_count == 4
    assert "opentsp_run_matmul_4x4" in result.c_path.read_text(encoding="utf-8")


def test_export_default_c_host_runner_creates_default_bundle_and_runner(tmp_path) -> None:
    result = export_default_c_host_runner(tmp_path / "host_runner" / "matmul_4x4")

    assert result.c_path.exists()
    assert result.h_path.exists()
    assert result.total_writes > 0
