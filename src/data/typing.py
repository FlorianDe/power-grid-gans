from dataclasses import dataclass
from typing import Optional


@dataclass
class Feature:
    label: str
    display_name: Optional[str] = None
