"""Training pipeline scaffold."""

from __future__ import annotations

import logging

from explainable_fl.data_ingestion.ingestor import IEEECISDataIngestor
from explainable_fl.pipelines.base import BasePipeline

LOGGER = logging.getLogger("explainable_fl.pipelines.training")


class TrainingPipeline(BasePipeline):
    """Coordinates training stages.

    Business logic intentionally not implemented in scaffold stage.
    """

    def run(self) -> None:
        ingestor = IEEECISDataIngestor()
        report = ingestor.run(self.config)
        LOGGER.info("Training pipeline ingestion stage completed: %s", report.output_files)
