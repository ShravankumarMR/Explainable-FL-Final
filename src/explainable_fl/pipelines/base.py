"""Base pipeline abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePipeline(ABC):
    """Contract for executable pipelines."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline."""
        raise NotImplementedError
