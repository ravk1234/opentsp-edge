"""Demo for the OpenTSP memory-bank conflict checker."""

from __future__ import annotations

from opentsp.bank_conflict_checker import check_instruction_memory_conflicts
from opentsp.memory_bank import MemoryBankConfig


def main() -> None:
    clean_instructions = [
        {"opcode": "LOAD_A", "op_name": "q_proj", "micro_op": "load_a_tile", "start_cycle": 0, "cycles": 3, "bank": 0, "offset": 0, "bytes": 8},
        {"opcode": "LOAD_B", "op_name": "q_proj", "micro_op": "load_b_tile", "start_cycle": 3, "cycles": 4, "bank": 1, "offset": 0, "bytes": 32},
        {"opcode": "MAC_TILE", "op_name": "q_proj", "micro_op": "mac_tile", "start_cycle": 7, "cycles": 3},
        {"opcode": "STORE_C", "op_name": "q_proj", "micro_op": "store_c_tile", "start_cycle": 10, "cycles": 3, "bank": 2, "offset": 0, "bytes": 16},
    ]

    conflicting_instructions = [
        {"opcode": "LOAD_A", "op_name": "bad", "micro_op": "load_a_tile", "start_cycle": 0, "cycles": 2, "bank": 0, "offset": 0, "bytes": 8},
        {"opcode": "LOAD_B", "op_name": "bad", "micro_op": "load_b_tile", "start_cycle": 1, "cycles": 2, "bank": 0, "offset": 64, "bytes": 32},
    ]

    config = MemoryBankConfig(num_banks=4, bank_size_bytes=256, read_ports_per_bank=1, write_ports_per_bank=1)
    clean_result = check_instruction_memory_conflicts(clean_instructions, config)
    conflict_result = check_instruction_memory_conflicts(conflicting_instructions, config)

    print("Memory-bank conflict checker demo")
    print("-" * 80)
    print(f"Clean program accesses: {clean_result.total_accesses}")
    print(f"Clean program conflicts: {len(clean_result.conflicts)}")
    print(f"Conflict demo accesses: {conflict_result.total_accesses}")
    print(f"Conflict demo conflicts: {len(conflict_result.conflicts)}")

    print("\nConflict details")
    print("-" * 80)
    for conflict in conflict_result.conflicts:
        print(f"{conflict.conflict_kind:22s} cycle={conflict.cycle} bank={conflict.bank} {conflict.message}")

    assert clean_result.ok
    assert not conflict_result.ok
    print("\nMemory-bank conflict checker demo: PASSED")


if __name__ == "__main__":
    main()
