from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional

from .hardware import AcceleratorConfig, dtype_nbytes
from .ir import Graph, OpSpec, TensorSpec


@dataclass(frozen=True)
class BankAllocation:
    tensor: str
    bank: int
    offset: int
    size_bytes: int


@dataclass(frozen=True)
class ScheduledOp:
    index: int
    name: str
    kind: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict
    start_cycle: int
    end_cycle: int
    cycles: int


@dataclass(frozen=True)
class CompiledProgram:
    graph_name: str
    hardware: AcceleratorConfig
    schedule: List[ScheduledOp]
    allocations: Dict[str, BankAllocation]
    total_cycles: int


def tensor_size_bytes(t: TensorSpec) -> int:
    return t.numel * dtype_nbytes(t.dtype)


def _estimate_cycles(graph: Graph, op: OpSpec, hw: AcceleratorConfig) -> int:
    kind = op.kind

    if kind == "matmul":
        a = graph.tensor(op.inputs[0]).shape
        b = graph.tensor(op.inputs[1]).shape
        if len(a) != 2 or len(b) != 2:
            raise ValueError(f"matmul expects 2D tensors for {op.name}")
        m, k = a
        k2, n = b
        if k != k2:
            raise ValueError(f"matmul shape mismatch for {op.name}: {a} x {b}")
        macs = m * n * k
        return hw.matmul_overhead_cycles + ceil(macs / hw.mac_lanes)

    if kind in {"add", "silu", "rmsnorm", "argmax", "append_cache"}:
        out = graph.tensor(op.outputs[0])
        return hw.elementwise_overhead_cycles + ceil(out.numel / hw.elem_lanes)

    if kind == "attention_decode":
        # q: [1, d], k_cache/v_cache: [T, d]
        q = graph.tensor(op.inputs[0]).shape
        k = graph.tensor(op.inputs[1]).shape
        if len(q) != 2 or len(k) != 2:
            raise ValueError(f"attention_decode expects q [1,d], cache [T,d] for {op.name}")
        _, d = q
        t, d2 = k
        if d != d2:
            raise ValueError(f"attention_decode shape mismatch for {op.name}")
        score_macs = t * d
        value_macs = t * d
        return hw.softmax_overhead_cycles + ceil((score_macs + value_macs) / hw.mac_lanes) + t

    if kind == "softmax":
        out = graph.tensor(op.outputs[0])
        return hw.softmax_overhead_cycles + ceil(out.numel / hw.elem_lanes)

    raise ValueError(f"Unsupported op kind: {kind}")


def _allocate_banks(graph: Graph, hw: AcceleratorConfig) -> Dict[str, BankAllocation]:
    """Deterministic greedy round-robin allocation.

    This is deliberately simple. Later versions should do lifetime-aware allocation.
    """

    bank_offsets = [0 for _ in range(hw.num_banks)]
    allocations: Dict[str, BankAllocation] = {}

    # Stable order: graph insertion order from model builder.
    for i, (name, spec) in enumerate(graph.tensors.items()):
        size = tensor_size_bytes(spec)
        bank = spec.bank if spec.bank is not None else i % hw.num_banks
        offset = bank_offsets[bank]

        if offset + size > hw.bank_size_bytes:
            raise MemoryError(
                f"Tensor {name} needs {size} bytes but bank {bank} would exceed "
                f"{hw.bank_size_bytes} bytes. Increase bank size or add tiling."
            )

        allocations[name] = BankAllocation(name, bank, offset, size)
        bank_offsets[bank] += size

    return allocations


def compile_graph(graph: Graph, hw: Optional[AcceleratorConfig] = None) -> CompiledProgram:
    hw = hw or AcceleratorConfig()
    graph.validate()

    allocations = _allocate_banks(graph, hw)
    schedule: List[ScheduledOp] = []
    cycle = 0

    for idx, op in enumerate(graph.ops):
        cycles = _estimate_cycles(graph, op, hw)
        scheduled = ScheduledOp(
            index=idx,
            name=op.name,
            kind=op.kind,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            attrs=dict(op.attrs),
            start_cycle=cycle,
            end_cycle=cycle + cycles,
            cycles=cycles,
        )
        schedule.append(scheduled)
        cycle += cycles

    return CompiledProgram(
        graph_name=graph.name,
        hardware=hw,
        schedule=schedule,
        allocations=allocations,
        total_cycles=cycle,
    )
