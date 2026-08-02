"""Training pipeline scaffold."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from explainable_fl.data_ingestion.ingestor import IEEECISDataIngestor
from explainable_fl.feature_engineering.engineer import TabularFeatureEngineer
from explainable_fl.pipelines.base import BasePipeline
from explainable_fl.preprocessing.preprocessor import TabularPreprocessor

LOGGER = logging.getLogger("explainable_fl.pipelines.training")


class TrainingPipeline(BasePipeline):
    """Coordinates training stages.

    Business logic intentionally not implemented in scaffold stage.
    """

    def run(self) -> None:
        if not self.config.pipelines.train.enabled:
            LOGGER.info("Train pipeline toggle is disabled at pipelines.train.enabled=false")
            return

        ingestor = IEEECISDataIngestor()
        report = ingestor.run(self.config)

        train_parquet = report.output_files["train_merged_parquet"]
        test_parquet = report.output_files["test_merged_parquet"]
        train_df = pd.read_parquet(train_parquet)
        test_df = pd.read_parquet(test_parquet)

        feature_engineer = TabularFeatureEngineer(config=self.config)
        train_featured = feature_engineer.fit_transform(train_df)
        test_featured = feature_engineer.transform(test_df)

        preprocessor = TabularPreprocessor(
            config=self.config,
            max_onehot_cardinality=self.config.preprocessing.max_onehot_cardinality,
        )
        train_preprocessed = preprocessor.fit_transform(train_featured)
        test_preprocessed = preprocessor.transform(test_featured)

        processed_root = Path(self.config.paths.data_processed) / report.dataset_name
        preprocessing_artifacts_dir = processed_root / self.config.preprocessing.artifacts_subdir
        feature_artifacts_dir = processed_root / self.config.feature_engineering.artifacts_subdir
        processed_root.mkdir(parents=True, exist_ok=True)

        train_out = processed_root / "train_preprocessed.parquet"
        test_out = processed_root / "test_preprocessed.parquet"
        train_preprocessed.to_parquet(train_out, index=False)
        test_preprocessed.to_parquet(test_out, index=False)

        feature_artifacts = feature_engineer.save_artifacts(feature_artifacts_dir)
        preprocessing_artifacts = preprocessor.save_artifacts(preprocessing_artifacts_dir)

        LOGGER.info("Training pipeline ingestion stage completed: %s", report.output_files)
        LOGGER.info("Preprocessed train output: %s", train_out)
        LOGGER.info("Preprocessed test output: %s", test_out)
        LOGGER.info("Feature engineering artifacts: %s", feature_artifacts)
        LOGGER.info("Preprocessing artifacts: %s", preprocessing_artifacts)
