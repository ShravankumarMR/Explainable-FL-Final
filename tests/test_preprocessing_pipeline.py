"""Tests for preprocessing fit/transform behavior and pipeline integration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from explainable_fl.config.loader import load_app_config
from explainable_fl.pipelines.inference_pipeline import InferencePipeline
from explainable_fl.pipelines.training_pipeline import TrainingPipeline
from explainable_fl.preprocessing.preprocessor import TabularPreprocessor


def _write_ieee_csvs(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionAmt": [100.0, None, 220.0, 315.0],
            "ProductCD": ["W", "C", None, "W"],
            "merchant": ["m1", "m2", "m3", "m4"],
            "dropme": [10, 11, 12, 13],
            "isFraud": [0, 1, 0, 1],
        }
    ).to_csv(dataset_root / "train_transaction.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [1, 2, 4],
            "DeviceType": ["desktop", None, "mobile"],
        }
    ).to_csv(dataset_root / "train_identity.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [8, 9],
            "TransactionAmt": [None, 500.0],
            "ProductCD": ["R", "C"],
            "merchant": ["m100", "m2"],
            "dropme": [20, 21],
            "isFraud": [0, 0],
        }
    ).to_csv(dataset_root / "test_transaction.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [8, 9],
            "DeviceType": ["tablet", "desktop"],
        }
    ).to_csv(dataset_root / "test_identity.csv", index=False)


def _write_config(config_path: Path, raw_path: Path, interim_path: Path, processed_path: Path, reports_path: Path) -> None:
    config_path.write_text(
        f"""
project:
  name: explainable-fl
  environment: test
  random_seed: 42

paths:
  data_raw: {raw_path.as_posix()}
  data_interim: {interim_path.as_posix()}
  data_processed: {processed_path.as_posix()}
  models_dir: models
  reports_dir: {reports_path.as_posix()}

ingestion:
  active_dataset: ieee_cis_fraud
  validate_schema: true
  datasets:
    ieee_cis_fraud:
      base_dir: IEEE-CIS-FRAUD
      train_transaction_file: train_transaction.csv
      train_identity_file: train_identity.csv
      test_transaction_file: test_transaction.csv
      test_identity_file: test_identity.csv
      merge_key: TransactionID
      required_transaction_columns: [TransactionID]
      required_identity_columns: [TransactionID]
      output_train_parquet: train_merged.parquet
      output_test_parquet: test_merged.parquet
      report_file: ingestion_report.yaml

preprocessing:
  target_column: isFraud
  drop_columns: [dropme]
  missing_value_strategy: median
  scale_numeric: true
  categorical_encoding: onehot
  max_onehot_cardinality: 2
  artifacts_subdir: preprocessing

feature_engineering:
  selected_features: []
  create_interactions: false
  polynomial_degree: 1

model_parameters:
  algorithm: random_forest
  hyperparameters: {{}}

evaluation:
  metrics: [accuracy]
  test_size: 0.2
  cross_validation_folds: 5

profiling:
  enabled: false
  include_train: false
  include_test: false
  high_cardinality_ratio: 0.8

eda:
  enabled: false
  include_train: false
  include_test: false
  target_column: isFraud
  sample_size: 100
  max_numerical_features: 10
  max_categorical_features: 10
  max_category_levels: 8
  correlation_top_n: 20
  pca_components: 2
  umap_n_neighbors: 5
  umap_min_dist: 0.1
  feature_importance_top_n: 10
  random_state: 42

run:
  mode: train
  experiment_name: test

logging:
  config_path:
  level: INFO

pipelines:
  ingest:
    enabled: true
  train:
    enabled: true
  infer:
    enabled: true
  profile:
    enabled: false
  eda:
    enabled: false
""".strip(),
        encoding="utf-8",
    )


def test_tabular_preprocessor_fit_transform_consistency(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    interim_root = tmp_path / "interim"
    processed_root = tmp_path / "processed"
    reports_root = tmp_path / "reports"
    _write_ieee_csvs(raw_root / "IEEE-CIS-FRAUD")

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_root, interim_root, processed_root, reports_root)
    config = load_app_config(cfg_path)

    train_df = pd.read_csv(raw_root / "IEEE-CIS-FRAUD" / "train_transaction.csv")
    test_df = pd.read_csv(raw_root / "IEEE-CIS-FRAUD" / "test_transaction.csv")

    preprocessor = TabularPreprocessor(config=config, max_onehot_cardinality=2)
    train_transformed = preprocessor.fit_transform(train_df)
    test_transformed = preprocessor.transform(test_df)

    train_features = [column for column in train_transformed.columns if column != "isFraud"]
    test_features = [column for column in test_transformed.columns if column != "isFraud"]
    assert train_features == test_features

    assert "dropme" not in train_transformed.columns
    assert "merchant__freq" in train_transformed.columns
    assert any(column.startswith("ProductCD__") for column in train_transformed.columns)

    assert train_transformed[train_features].isna().sum().sum() == 0
    assert test_transformed[test_features].isna().sum().sum() == 0

    artifacts_paths = preprocessor.save_artifacts(processed_root / "artifacts")
    loaded = TabularPreprocessor(config=config).load_artifacts(artifacts_paths["pickle"])
    reloaded_test = loaded.transform(test_df)

    assert_frame_equal(test_transformed, reloaded_test)


def test_training_and_inference_pipelines_generate_preprocessed_outputs(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    interim_root = tmp_path / "interim"
    processed_root = tmp_path / "processed"
    reports_root = tmp_path / "reports"
    _write_ieee_csvs(raw_root / "IEEE-CIS-FRAUD")

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_root, interim_root, processed_root, reports_root)
    config = load_app_config(cfg_path)

    TrainingPipeline(config=config).run()

    dataset_processed_root = processed_root / "ieee_cis_fraud"
    artifacts_dir = dataset_processed_root / "preprocessing"

    train_preprocessed = dataset_processed_root / "train_preprocessed.parquet"
    test_preprocessed = dataset_processed_root / "test_preprocessed.parquet"
    artifacts_pickle = artifacts_dir / "preprocessing_artifacts.pkl"
    artifacts_yaml = artifacts_dir / "preprocessing_artifacts.yaml"

    assert train_preprocessed.exists()
    assert test_preprocessed.exists()
    assert artifacts_pickle.exists()
    assert artifacts_yaml.exists()

    InferencePipeline(config=config).run()
    inference_preprocessed = dataset_processed_root / "inference_preprocessed.parquet"
    assert inference_preprocessed.exists()

    train_df = pd.read_parquet(train_preprocessed)
    inference_df = pd.read_parquet(inference_preprocessed)

    train_features = [column for column in train_df.columns if column != "isFraud"]
    inference_features = [column for column in inference_df.columns if column != "isFraud"]
    assert train_features == inference_features
