from opentsp.bank_conflict_checker import check_instruction_memory_conflicts, instructions_to_memory_accesses
from opentsp.memory_bank import MemoryAccess, MemoryBankConfig, check_memory_bank_conflicts


def test_non_overlapping_accesses_have_no_conflicts():
    accesses = [
        MemoryAccess(start_cycle=0, cycles=2, bank=0, offset=0, size_bytes=8, kind="read"),
        MemoryAccess(start_cycle=2, cycles=2, bank=0, offset=8, size_bytes=8, kind="read"),
        MemoryAccess(start_cycle=4, cycles=2, bank=2, offset=0, size_bytes=16, kind="write"),
    ]

    result = check_memory_bank_conflicts(accesses, MemoryBankConfig(num_banks=4, bank_size_bytes=64))
    assert result.ok
    assert result.total_accesses == 3


def test_two_reads_same_bank_same_cycle_conflict():
    accesses = [
        MemoryAccess(start_cycle=0, cycles=3, bank=1, offset=0, size_bytes=8, kind="read"),
        MemoryAccess(start_cycle=1, cycles=2, bank=1, offset=8, size_bytes=8, kind="read"),
    ]

    result = check_memory_bank_conflicts(accesses, MemoryBankConfig(num_banks=4, bank_size_bytes=64))
    assert not result.ok
    assert any(c.conflict_kind == "read_port_conflict" for c in result.conflicts)


def test_read_write_same_bank_same_cycle_conflict():
    accesses = [
        MemoryAccess(start_cycle=5, cycles=1, bank=2, offset=0, size_bytes=8, kind="read"),
        MemoryAccess(start_cycle=5, cycles=1, bank=2, offset=16, size_bytes=8, kind="write"),
    ]

    result = check_memory_bank_conflicts(accesses, MemoryBankConfig(num_banks=4, bank_size_bytes=64))
    assert not result.ok
    assert any(c.conflict_kind == "read_write_conflict" for c in result.conflicts)


def test_out_of_bounds_access_is_reported():
    accesses = [MemoryAccess(start_cycle=0, cycles=1, bank=0, offset=60, size_bytes=8, kind="read")]
    result = check_memory_bank_conflicts(accesses, MemoryBankConfig(num_banks=4, bank_size_bytes=64))
    assert not result.ok
    assert any(c.conflict_kind == "out_of_bounds" for c in result.conflicts)


def test_instruction_dicts_are_converted_to_accesses():
    instructions = [
        {"opcode": "LOAD_A", "start_cycle": 0, "cycles": 2, "bank": 0, "offset": 0, "bytes": 8},
        {"opcode": "MAC_TILE", "start_cycle": 2, "cycles": 3},
        {"opcode": "STORE_C", "start_cycle": 5, "cycles": 2, "bank": 2, "offset": 0, "bytes": 16},
    ]

    accesses = instructions_to_memory_accesses(instructions)
    assert len(accesses) == 2
    assert accesses[0].kind == "read"
    assert accesses[1].kind == "write"


def test_clean_instruction_stream_passes_bank_checker():
    instructions = [
        {"opcode": "LOAD_A", "start_cycle": 0, "cycles": 2, "bank": 0, "offset": 0, "bytes": 8},
        {"opcode": "LOAD_B", "start_cycle": 2, "cycles": 2, "bank": 1, "offset": 0, "bytes": 32},
        {"opcode": "MAC_TILE", "start_cycle": 4, "cycles": 3},
        {"opcode": "STORE_C", "start_cycle": 7, "cycles": 2, "bank": 2, "offset": 0, "bytes": 16},
    ]

    result = check_instruction_memory_conflicts(instructions, MemoryBankConfig(num_banks=4, bank_size_bytes=256))
    assert result.ok
    assert result.total_accesses == 3


def test_conflicting_instruction_stream_fails_bank_checker():
    instructions = [
        {"opcode": "LOAD_A", "start_cycle": 0, "cycles": 3, "bank": 0, "offset": 0, "bytes": 8},
        {"opcode": "LOAD_B", "start_cycle": 1, "cycles": 2, "bank": 0, "offset": 32, "bytes": 32},
    ]

    result = check_instruction_memory_conflicts(instructions, MemoryBankConfig(num_banks=4, bank_size_bytes=256))
    assert not result.ok
    assert any(c.conflict_kind == "read_port_conflict" for c in result.conflicts)
