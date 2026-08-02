"""EDA contracts and scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEDAAnalyzer(ABC):
    """Contract for EDA components."""

    @abstractmethod
    def run(self, config: dict[str, Any]) -> None:
        raise NotImplementedError
