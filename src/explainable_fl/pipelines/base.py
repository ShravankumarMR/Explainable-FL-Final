"""Base pipeline abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from explainable_fl.config.loader import AppConfig


class BasePipeline(ABC):
    """Contract for executable pipelines."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline."""
        raise NotImplementedError
