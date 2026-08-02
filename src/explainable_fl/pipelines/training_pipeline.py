"""Training pipeline scaffold."""

from __future__ import annotations

from explainable_fl.pipelines.base import BasePipeline


class TrainingPipeline(BasePipeline):
    """Coordinates training stages.

    Business logic intentionally not implemented in scaffold stage.
    """

    def run(self) -> None:
        raise NotImplementedError("Training pipeline logic is not implemented yet")
