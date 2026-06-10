# SPDX-License-Identifier: Apache-2.0
"""Utilities for reading and validating OpenTSP RTL filelists."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALLOWED_RTL_SUFFIXES = {".sv", ".v", ".vh"}


@dataclass(frozen=True)
class RtlFilelist:
    """Parsed RTL filelist."""

    path: Path
    sources: tuple[Path, ...]

    @property
    def unique_sources(self) -> tuple[Path, ...]:
        seen: set[Path] = set()
        out: list[Path] = []
        for source in self.sources:
            if source not in seen:
                out.append(source)
                seen.add(source)
        return tuple(out)


def read_rtl_filelist(path: str | Path) -> RtlFilelist:
    """Read a simple Verilog/SystemVerilog filelist.

    Blank lines and lines beginning with ``#`` are ignored.
    Paths are returned relative to the repository root, matching the filelist.
    """

    filelist_path = Path(path)
    sources: list[Path] = []

    for raw_line in filelist_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sources.append(Path(line))

    return RtlFilelist(path=filelist_path, sources=tuple(sources))


def validate_rtl_filelist(
    filelist: RtlFilelist,
    repo_root: str | Path = ".",
    required_sources: tuple[str, ...] = (),
) -> list[str]:
    """Return validation errors for an RTL filelist."""

    root = Path(repo_root)
    errors: list[str] = []

    if not filelist.path.exists():
        errors.append(f"filelist does not exist: {filelist.path}")

    if not filelist.sources:
        errors.append("filelist contains no RTL sources")

    seen: set[Path] = set()
    for source in filelist.sources:
        if source in seen:
            errors.append(f"duplicate RTL source: {source}")
        seen.add(source)

        if source.suffix not in ALLOWED_RTL_SUFFIXES:
            errors.append(f"unsupported RTL source extension: {source}")

        if not (root / source).exists():
            errors.append(f"missing RTL source: {source}")

    source_strings = {source.as_posix() for source in filelist.sources}
    for required in required_sources:
        if required not in source_strings:
            errors.append(f"required RTL source missing from filelist: {required}")

    return errors


def default_generic_axi_filelist(repo_root: str | Path = ".") -> Path:
    """Return the default generic AXI filelist path."""

    return Path(repo_root) / "fpga" / "generic_axi" / "filelist.f"
