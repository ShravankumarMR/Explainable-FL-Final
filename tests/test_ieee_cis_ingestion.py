"""Tests for IEEE-CIS data ingestion pipeline behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from explainable_fl.config.loader import ConfigError, load_app_config
from explainable_fl.data_ingestion.ingestor import IEEECISDataIngestor


def _write_ieee_csvs(dataset_root: Path, include_identity_key: bool = True) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionAmt": [10.0, 20.0, 30.0],
        }
    ).to_csv(dataset_root / "train_transaction.csv", index=False)

    identity_train = pd.DataFrame(
        {
            "TransactionID": [1, 3, 3],
            "DeviceType": ["desktop", "mobile", "mobile"],
        }
    )
    if not include_identity_key:
        identity_train = identity_train.drop(columns=["TransactionID"])
    identity_train.to_csv(dataset_root / "train_identity.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [4, 5],
            "TransactionAmt": [40.0, 50.0],
        }
    ).to_csv(dataset_root / "test_transaction.csv", index=False)

    identity_test = pd.DataFrame(
        {
            "TransactionID": [4],
            "DeviceType": ["tablet"],
        }
    )
    if not include_identity_key:
        identity_test = identity_test.drop(columns=["TransactionID"])
    identity_test.to_csv(dataset_root / "test_identity.csv", index=False)


def _write_config(config_path: Path, raw_path: Path, interim_path: Path, reports_path: Path) -> None:
    config_path.write_text(
        f"""
project:
  name: explainable-fl
  environment: test
  random_seed: 42

paths:
  data_raw: {raw_path.as_posix()}
  data_interim: {interim_path.as_posix()}
  data_processed: data/processed
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
  drop_columns: []
  missing_value_strategy: median
  scale_numeric: true
  categorical_encoding: onehot

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

run:
  mode: ingest
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
    enabled: false
""".strip(),
        encoding="utf-8",
    )


def test_ieee_ingestor_merges_and_writes_parquet(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    interim_root = tmp_path / "interim"
    reports_root = tmp_path / "reports"
    _write_ieee_csvs(raw_root / "IEEE-CIS-FRAUD")

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_root, interim_root, reports_root)
    config = load_app_config(cfg_path)

    report = IEEECISDataIngestor().run(config)

    train_out = interim_root / "ieee_cis_fraud" / "train_merged.parquet"
    test_out = interim_root / "ieee_cis_fraud" / "test_merged.parquet"
    report_out = reports_root / "ingestion" / "ieee_cis_fraud" / "ingestion_report.yaml"

    assert train_out.exists()
    assert test_out.exists()
    assert report_out.exists()

    train_df = pd.read_parquet(train_out)
    test_df = pd.read_parquet(test_out)
    assert len(train_df) == report.row_counts["train_merged"]
    assert len(test_df) == report.row_counts["test_merged"]

    assert report.missing_identity_rows["train"] == 1
    assert report.missing_identity_rows["test"] == 1
    assert report.duplicate_transaction_ids["train_identity"] == 1
    assert "train_merged" in report.memory_usage_mb


def test_ieee_ingestor_schema_validation_raises(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    interim_root = tmp_path / "interim"
    reports_root = tmp_path / "reports"
    _write_ieee_csvs(raw_root / "IEEE-CIS-FRAUD", include_identity_key=False)

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_root, interim_root, reports_root)
    config = load_app_config(cfg_path)

    with pytest.raises(ConfigError, match="Schema validation failed"):
        IEEECISDataIngestor().run(config)
