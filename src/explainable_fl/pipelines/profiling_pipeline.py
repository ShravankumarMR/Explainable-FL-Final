"""Profiling pipeline to produce automated EDA artifacts for train/test splits."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from explainable_fl.config.loader import ConfigError
from explainable_fl.data_ingestion.ingestor import IEEECISDataIngestor
from explainable_fl.eda.profiler import DatasetProfiler
from explainable_fl.pipelines.base import BasePipeline

LOGGER = logging.getLogger("explainable_fl.pipelines.profiling")


class ProfilingPipeline(BasePipeline):
    """Coordinates ingestion outputs and profiling report generation."""

    def run(self) -> None:
        if not self.config.profiling.enabled:
            LOGGER.info("Profiling is disabled by config.profiling.enabled=false")
            return

        if not self.config.pipelines.profile.enabled:
            LOGGER.info("Profile pipeline toggle is disabled at pipelines.profile.enabled=false")
            return

        ingestion_report = IEEECISDataIngestor().run(self.config)
        dataset_name = ingestion_report.dataset_name

        profiler = DatasetProfiler()
        reports_root = Path(self.config.paths.reports_dir) / "profiling" / dataset_name
        reports_root.mkdir(parents=True, exist_ok=True)

        split_targets = self._get_split_targets(
            ingestion_report.output_files,
            include_train=self.config.profiling.include_train,
            include_test=self.config.profiling.include_test,
        )

        if not split_targets:
            raise ConfigError("No profiling targets are enabled. Set include_train or include_test to true.")

        for split_name, parquet_path in split_targets.items():
            dataframe = pd.read_parquet(parquet_path)
            split_dir = reports_root / split_name
            artifacts = profiler.run(
                dataframe=dataframe,
                split_name=split_name,
                output_dir=split_dir,
                high_cardinality_ratio=self.config.profiling.high_cardinality_ratio,
            )
            LOGGER.info("Profiled split=%s with outputs in %s", artifacts.split_name, artifacts.output_dir)
            LOGGER.info("CSV files: %s", artifacts.csv_files)
            LOGGER.info("HTML report: %s", artifacts.html_file)

    @staticmethod
    def _get_split_targets(
        output_files: dict[str, str],
        include_train: bool,
        include_test: bool,
    ) -> dict[str, str]:
        split_targets: dict[str, str] = {}

        if include_train:
            train_path = output_files.get("train_merged_parquet")
            if train_path is None:
                raise ConfigError("Ingestion output file not found: train_merged_parquet")
            split_targets["train"] = train_path

        if include_test:
            test_path = output_files.get("test_merged_parquet")
            if test_path is None:
                raise ConfigError("Ingestion output file not found: test_merged_parquet")
            split_targets["test"] = test_path

        return split_targets
