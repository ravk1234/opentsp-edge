from __future__ import annotations

from opentsp.cpp_host_runner import (
    export_cpp_host_runner,
    export_default_cpp_host_runner,
    render_cpp_host_makefile,
    render_cpp_host_sim,
)
from opentsp.c_host_runner import export_default_c_host_runner


def test_render_cpp_host_sim_contains_host_callbacks() -> None:
    text = render_cpp_host_sim()

    assert "host_write_reg" in text
    assert "host_read_reg" in text
    assert "opentsp_run_matmul_4x4" in text
    assert "OpenTSP C++ host simulation: PASS" in text


def test_render_makefile_builds_expected_target() -> None:
    text = render_cpp_host_makefile()

    assert "opentsp_matmul_4x4_host_sim" in text
    assert "opentsp_matmul_4x4_runner.c" in text
    assert "g++" in text


def test_export_cpp_host_runner_writes_cpp_and_makefile(tmp_path) -> None:
    c_result = export_default_c_host_runner(tmp_path / "host_runner")
    result = export_cpp_host_runner(tmp_path / "host_runner", c_host_result=c_result)

    assert result.c_path.exists()
    assert result.h_path.exists()
    assert result.cpp_path.exists()
    assert result.makefile_path.exists()
    assert result.total_writes == 176
    assert result.tile_program_count == 4


def test_export_default_cpp_host_runner_creates_full_scaffold(tmp_path) -> None:
    result = export_default_cpp_host_runner(tmp_path / "cpp_host_runner" / "matmul_4x4")

    assert result.c_path.exists()
    assert result.h_path.exists()
    assert result.cpp_path.exists()
    assert result.makefile_path.exists()
    assert "OPENTSP_MATMUL_4X4_WRITES" in result.c_path.read_text(encoding="utf-8")


def test_cpp_host_sim_has_expected_output_check(tmp_path) -> None:
    result = export_default_cpp_host_runner(tmp_path / "cpp_host_runner" / "matmul_4x4")
    text = result.cpp_path.read_text(encoding="utf-8")

    assert "OPENTSP_MATMUL_4X4_EXPECTED_C" in text
    assert "Mismatch at C" in text
    assert "return 1" in text
