"""Tests for automated profiling pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from explainable_fl.config.loader import load_app_config
from explainable_fl.pipelines.profiling_pipeline import ProfilingPipeline


def _write_ieee_csvs(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 3],
            "TransactionAmt": [10.0, 20.0, 30.0, 30.0],
            "ProductCD": ["W", "H", "W", "W"],
        }
    ).to_csv(dataset_root / "train_transaction.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [1, 3],
            "DeviceType": ["desktop", "mobile"],
        }
    ).to_csv(dataset_root / "train_identity.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [4, 5, 6],
            "TransactionAmt": [40.0, 50.0, 60.0],
            "ProductCD": ["C", "C", "R"],
        }
    ).to_csv(dataset_root / "test_transaction.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [4, 6],
            "DeviceType": ["tablet", "desktop"],
        }
    ).to_csv(dataset_root / "test_identity.csv", index=False)


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

profiling:
  enabled: true
  include_train: true
  include_test: true
  high_cardinality_ratio: 0.5

run:
  mode: profile
  experiment_name: test

logging:
  config_path:
  level: INFO

pipelines:
  ingest:
    enabled: true
  train:
    enabled: false
  infer:
    enabled: false
  profile:
    enabled: true
""".strip(),
        encoding="utf-8",
    )


def test_profiling_pipeline_exports_csv_and_html_reports(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    interim_root = tmp_path / "interim"
    reports_root = tmp_path / "reports"
    _write_ieee_csvs(raw_root / "IEEE-CIS-FRAUD")

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_root, interim_root, reports_root)
    config = load_app_config(cfg_path)

    ProfilingPipeline(config=config).run()

    base_report_dir = reports_root / "profiling" / "ieee_cis_fraud"
    expected_csv_files = [
        "data_types.csv",
        "missing_values.csv",
        "unique_counts.csv",
        "summary_statistics.csv",
        "duplicate_analysis.csv",
        "memory_usage.csv",
        "feature_groups.csv",
    ]

    for split in ["train", "test"]:
        split_dir = base_report_dir / split
        assert split_dir.exists()

        html_report = split_dir / "profile_report.html"
        assert html_report.exists()
        assert "Automated Profile Report" in html_report.read_text(encoding="utf-8")

        for csv_name in expected_csv_files:
            csv_path = split_dir / csv_name
            assert csv_path.exists()

    train_dupes = pd.read_csv(base_report_dir / "train" / "duplicate_analysis.csv")
    assert int(train_dupes.loc[0, "duplicate_rows"]) >= 0

    feature_groups = pd.read_csv(base_report_dir / "train" / "feature_groups.csv")
    assert "group" in feature_groups.columns
    assert "feature" in feature_groups.columns
