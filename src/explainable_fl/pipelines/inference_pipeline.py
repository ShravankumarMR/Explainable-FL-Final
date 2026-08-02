"""Inference pipeline scaffold."""

from __future__ import annotations

from explainable_fl.pipelines.base import BasePipeline


class InferencePipeline(BasePipeline):
    """Coordinates inference stages.

    Business logic intentionally not implemented in scaffold stage.
    """

    def run(self) -> None:
        raise NotImplementedError("Inference pipeline logic is not implemented yet")
