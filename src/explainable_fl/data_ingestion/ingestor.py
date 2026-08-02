"""Data ingestion contracts and scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDataIngestor(ABC):
    """Contract for data ingestion components."""

    @abstractmethod
    def run(self, config: dict[str, Any]) -> None:
        raise NotImplementedError
