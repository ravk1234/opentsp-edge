from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


DTYPE_BYTES: Dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
}


@dataclass(frozen=True)
class SRAMBank:
    bank_id: int
    size_bytes: int
    width_bits: int = 32


@dataclass(frozen=True)
class AcceleratorConfig:
    """Simple target model for deterministic local scheduling.

    This is an abstract accelerator model, not a real chip description yet.
    """

    name: str = "opentsp-local-v0"
    num_banks: int = 4
    bank_size_bytes: int = 256 * 1024
    mac_lanes: int = 64
    elem_lanes: int = 16
    clock_mhz: int = 100
    matmul_overhead_cycles: int = 8
    elementwise_overhead_cycles: int = 4
    softmax_overhead_cycles: int = 12

    @property
    def banks(self) -> List[SRAMBank]:
        return [SRAMBank(i, self.bank_size_bytes) for i in range(self.num_banks)]


def dtype_nbytes(dtype: str) -> int:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype {dtype}")
    return DTYPE_BYTES[dtype]
