"""EDA pipeline to generate reusable visual artifacts per dataset split."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from explainable_fl.config.loader import ConfigError
from explainable_fl.data_ingestion.ingestor import IEEECISDataIngestor
from explainable_fl.eda.analyzer import DatasetEDAAnalyzer
from explainable_fl.pipelines.base import BasePipeline

LOGGER = logging.getLogger("explainable_fl.pipelines.eda")


class EDAPipeline(BasePipeline):
    """Coordinates ingestion outputs and visualization generation."""

    def run(self) -> None:
        if not self.config.eda.enabled:
            LOGGER.info("EDA is disabled by config.eda.enabled=false")
            return

        if not self.config.pipelines.eda.enabled:
            LOGGER.info("EDA pipeline toggle is disabled at pipelines.eda.enabled=false")
            return

        ingestion_report = IEEECISDataIngestor().run(self.config)
        dataset_name = ingestion_report.dataset_name
        dataset_cfg = self.config.ingestion.datasets.get(dataset_name)
        if dataset_cfg is None:
            raise ConfigError(f"Dataset config not found for active dataset: {dataset_name}")

        analyzer = DatasetEDAAnalyzer()
        reports_root = Path(self.config.paths.reports_dir) / "eda" / dataset_cfg.base_dir
        reports_root.mkdir(parents=True, exist_ok=True)

        split_targets = self._get_split_targets(
            ingestion_report.output_files,
            include_train=self.config.eda.include_train,
            include_test=self.config.eda.include_test,
        )

        if not split_targets:
            raise ConfigError("No EDA targets are enabled. Set include_train or include_test to true.")

        for split_name, parquet_path in split_targets.items():
            dataframe = pd.read_parquet(parquet_path)
            split_dir = reports_root / split_name
            artifacts = analyzer.run(
                dataframe=dataframe,
                split_name=split_name,
                output_dir=split_dir,
                config=self.config.eda,
            )
            LOGGER.info("EDA generated for split=%s at %s", artifacts.split_name, artifacts.output_dir)
            LOGGER.info("Figures: %s", artifacts.figure_files)
            LOGGER.info("Summary: %s", artifacts.summary_file)

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
