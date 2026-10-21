from dataclasses import dataclass, field
from typing import Literal

FeatureKind = Literal["numeric", "binary", "categorical", "datetime"]

@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: FeatureKind
    availability: str = "observed"

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    op: str
    inputs: tuple[str, ...]
    params: tuple[tuple[str, object], ...] = field(default_factory=tuple)
    kind: FeatureKind = "numeric"
    lookback: int = 0
    complexity: int = 1

    @property
    def key(self):
        return (self.op, self.inputs, self.params, self.kind, self.lookback)

    @property
    def sources(self) -> frozenset[str]:
        """All raw source columns used by this expression."""
        return frozenset(self.inputs)
