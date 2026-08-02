"""Feature engineering contracts and scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseFeatureEngineer(ABC):
    """Contract for feature engineering components."""

    @abstractmethod
    def run(self, config: dict[str, Any]) -> None:
        raise NotImplementedError
