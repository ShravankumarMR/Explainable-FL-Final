"""Data ingestion pipeline for raw dataset ingestion and merge outputs."""

from __future__ import annotations

import logging

from explainable_fl.data_ingestion.ingestor import IEEECISDataIngestor
from explainable_fl.pipelines.base import BasePipeline

LOGGER = logging.getLogger("explainable_fl.pipelines.ingestion")


class IngestionPipeline(BasePipeline):
    """Coordinates data ingestion from configured raw dataset sources."""

    def run(self) -> None:
        ingestor = IEEECISDataIngestor()
        report = ingestor.run(self.config)

        LOGGER.info("Ingestion completed for dataset=%s", report.dataset_name)
        LOGGER.info("Row counts: %s", report.row_counts)
        LOGGER.info("Missing identity rows: %s", report.missing_identity_rows)
        LOGGER.info("Duplicate TransactionID counts: %s", report.duplicate_transaction_ids)
        LOGGER.info("Memory usage (MB): %s", report.memory_usage_mb)
        LOGGER.info("Saved outputs: %s", report.output_files)
