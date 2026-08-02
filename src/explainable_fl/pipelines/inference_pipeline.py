"""Inference pipeline scaffold."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from explainable_fl.config.loader import ConfigError
from explainable_fl.data_ingestion.ingestor import IEEECISDataIngestor
from explainable_fl.pipelines.base import BasePipeline
from explainable_fl.preprocessing.preprocessor import TabularPreprocessor

LOGGER = logging.getLogger("explainable_fl.pipelines.inference")


class InferencePipeline(BasePipeline):
    """Coordinates inference stages.

    Business logic intentionally not implemented in scaffold stage.
    """

    def run(self) -> None:
        if not self.config.pipelines.infer.enabled:
            LOGGER.info("Infer pipeline toggle is disabled at pipelines.infer.enabled=false")
            return

        report = IEEECISDataIngestor().run(self.config)
        test_df = pd.read_parquet(report.output_files["test_merged_parquet"])

        processed_root = Path(self.config.paths.data_processed) / report.dataset_name
        artifacts_dir = processed_root / self.config.preprocessing.artifacts_subdir
        artifacts_path = artifacts_dir / "preprocessing_artifacts.pkl"
        if not artifacts_path.exists():
            raise ConfigError(
                "Preprocessing artifacts not found for inference. "
                f"Expected file: {artifacts_path}. Run training pipeline first."
            )

        preprocessor = TabularPreprocessor(
            config=self.config,
            max_onehot_cardinality=self.config.preprocessing.max_onehot_cardinality,
        ).load_artifacts(artifacts_path)

        inference_ready = preprocessor.transform(test_df)
        processed_root.mkdir(parents=True, exist_ok=True)
        output_path = processed_root / "inference_preprocessed.parquet"
        inference_ready.to_parquet(output_path, index=False)

        LOGGER.info("Inference preprocessing output: %s", output_path)
