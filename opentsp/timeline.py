from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Any


@dataclass(frozen=True)
class TimelineEvent:
    """One event in the global deterministic token-step timeline."""

    index: int
    op_name: str
    source: str
    micro_op: str
    start_cycle: int
    end_cycle: int
    cycles: int
    bytes_moved: int = 0
    macs: int = 0
    detail: str = ""


@dataclass(frozen=True)
class TimelineResult:
    """Unified timeline produced from matmul, attention, and fallback ops."""

    events: tuple[TimelineEvent, ...]
    total_cycles: int

    @property
    def event_count(self) -> int:
        return len(self.events)

    def op_summary(self) -> tuple[tuple[str, str, int, int], ...]:
        """Return (op_name, source, event_count, cycles) in timeline order."""

        grouped: "OrderedDict[tuple[str, str], list[int]]" = OrderedDict()
        for event in self.events:
            key = (event.op_name, event.source)
            if key not in grouped:
                grouped[key] = [0, 0]
            grouped[key][0] += 1
            grouped[key][1] += event.cycles
        return tuple((op, source, count_cycles[0], count_cycles[1]) for (op, source), count_cycles in grouped.items())


def _stat_map(stats: Iterable[Any]) -> dict[str, Any]:
    return {stat.op_name: stat for stat in stats}


def _append_event(
    events: list[TimelineEvent],
    *,
    op_name: str,
    source: str,
    micro_op: str,
    global_cycle: int,
    local_start: int,
    local_end: int,
    cycles: int,
    bytes_moved: int = 0,
    macs: int = 0,
    detail: str = "",
) -> None:
    events.append(
        TimelineEvent(
            index=len(events),
            op_name=op_name,
            source=source,
            micro_op=micro_op,
            start_cycle=global_cycle + local_start,
            end_cycle=global_cycle + local_end,
            cycles=cycles,
            bytes_moved=bytes_moved,
            macs=macs,
            detail=detail,
        )
    )


def build_unified_timeline(
    program: Any,
    matmul_stats: Sequence[Any],
    attention_stats: Sequence[Any],
    *,
    include_fallback_ops: bool = True,
) -> TimelineResult:
    """Build one global cycle timeline for a decoder step.

    Tiled INT8 matmul and KV-cache attention already expose detailed local
    micro-op schedules. This function places those local schedules into one
    global token-step timeline, preserving the compiled graph operation order.

    Non-accelerated ops are represented as one coarse `baseline` event using
    the cycle estimate from the original compiled program.
    """

    matmuls = _stat_map(matmul_stats)
    attentions = _stat_map(attention_stats)

    events: list[TimelineEvent] = []
    cycle = 0

    for op in program.schedule:
        if op.name in matmuls:
            stat = matmuls[op.name]
            for event in stat.events:
                detail = f"M{event.m_range} N{event.n_range} K{event.k_range}"
                _append_event(
                    events,
                    op_name=op.name,
                    source="tiled_matmul",
                    micro_op=event.kind,
                    global_cycle=cycle,
                    local_start=event.start_cycle,
                    local_end=event.end_cycle,
                    cycles=event.cycles,
                    bytes_moved=event.bytes_moved,
                    macs=event.macs,
                    detail=detail,
                )
            cycle += int(stat.cycles)
            continue

        if op.name in attentions:
            stat = attentions[op.name]
            for event in stat.events:
                detail = f"T{event.token_range} D{event.dim_range}"
                _append_event(
                    events,
                    op_name=op.name,
                    source="kv_attention",
                    micro_op=event.kind,
                    global_cycle=cycle,
                    local_start=event.start_cycle,
                    local_end=event.end_cycle,
                    cycles=event.cycles,
                    bytes_moved=event.bytes_moved,
                    macs=event.macs,
                    detail=detail,
                )
            cycle += int(stat.cycles)
            continue

        if include_fallback_ops:
            _append_event(
                events,
                op_name=op.name,
                source="baseline",
                micro_op=op.kind,
                global_cycle=cycle,
                local_start=0,
                local_end=int(op.cycles),
                cycles=int(op.cycles),
                detail=f"inputs={tuple(op.inputs)} outputs={tuple(op.outputs)}",
            )
        cycle += int(op.cycles)

    return TimelineResult(events=tuple(events), total_cycles=cycle)
