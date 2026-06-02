from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


Shape = Tuple[int, ...]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Shape
    dtype: str = "float32"
    is_weight: bool = False
    bank: Optional[int] = None

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str
    inputs: Sequence[str]
    outputs: Sequence[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    name: str
    tensors: Dict[str, TensorSpec]
    ops: List[OpSpec]
    inputs: List[str]
    outputs: List[str]

    def tensor(self, name: str) -> TensorSpec:
        if name not in self.tensors:
            raise KeyError(f"Unknown tensor: {name}")
        return self.tensors[name]

    def validate(self) -> None:
        produced = set(self.inputs)
        produced.update(name for name, spec in self.tensors.items() if spec.is_weight)

        for op in self.ops:
            missing = [x for x in op.inputs if x not in produced]
            if missing:
                raise ValueError(f"Op {op.name} has missing inputs: {missing}")
            for out in op.outputs:
                if out not in self.tensors:
                    raise ValueError(f"Op {op.name} output {out} has no TensorSpec")
                produced.add(out)

        missing_outputs = [x for x in self.outputs if x not in produced]
        if missing_outputs:
            raise ValueError(f"Graph outputs not produced: {missing_outputs}")
