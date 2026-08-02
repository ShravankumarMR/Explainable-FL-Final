"""Inference contracts and scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePredictor(ABC):
    """Contract for inference/prediction components."""

    @abstractmethod
    def run(self, config: dict[str, Any]) -> None:
        raise NotImplementedError
