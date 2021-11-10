from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic, Optional

T = TypeVar("T")


@dataclass
class PlotOptions:
    title: Optional[str]


@dataclass
class PlotData(Generic[T]):
    data: T
    label: str
    color: str | list[str] | None = None
