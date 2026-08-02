"""Tests for EDA visualization pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from explainable_fl.config.loader import load_app_config
from explainable_fl.pipelines.eda_pipeline import EDAPipeline


def _write_ieee_csvs(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "TransactionID": list(range(1, 13)),
            "TransactionAmt": [10.0, 12.0, 9.0, 80.0, 70.0, 65.0, 14.0, 16.0, 18.0, 95.0, 120.0, 130.0],
            "TransactionDT": [
                1000,
                2000,
                3800,
                7200,
                8200,
                9000,
                3600 * 25,
                3600 * 26,
                3600 * 27,
                3600 * 48,
                3600 * 49,
                3600 * 50,
            ],
            "ProductCD": ["W", "W", "C", "H", "H", "S", "R", "W", "C", "R", "H", "S"],
            "card1": [1000, 1000, 1001, 1001, 1002, 1002, 1003, 1003, 1004, 1004, 1005, 1005],
            "card2": [111, 111, 120, 120, 130, 130, 140, 140, 150, 150, 160, 160],
            "P_emaildomain": [
                "gmail.com",
                "yahoo.com",
                "gmail.com",
                "hotmail.com",
                "gmail.com",
                "yahoo.com",
                "gmail.com",
                "aol.com",
                "aol.com",
                "gmail.com",
                "hotmail.com",
                "yahoo.com",
            ],
            "R_emaildomain": [
                "gmail.com",
                "gmail.com",
                "hotmail.com",
                "hotmail.com",
                "gmail.com",
                "gmail.com",
                "aol.com",
                "aol.com",
                "aol.com",
                "gmail.com",
                "hotmail.com",
                "hotmail.com",
            ],
            "isFraud": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        }
    ).to_csv(dataset_root / "train_transaction.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": list(range(1, 13)),
            "DeviceType": ["desktop", "mobile", "desktop", "mobile"] * 3,
            "DeviceInfo": ["Windows", "iOS", "Linux", "Android"] * 3,
            "id_01": [0.1, 0.2, None, 0.4, 0.2, 0.3, 0.1, None, 0.5, 0.3, 0.4, 0.2],
            "id_12": ["Found", "Found", "NotFound", "Found"] * 3,
        }
    ).to_csv(dataset_root / "train_identity.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [21, 22, 23, 24, 25, 26],
            "TransactionAmt": [15.0, 18.0, 20.0, 85.0, 88.0, 90.0],
            "TransactionDT": [4000, 5000, 6000, 3600 * 30, 3600 * 31, 3600 * 32],
            "ProductCD": ["W", "W", "C", "H", "H", "S"],
            "card1": [1006, 1006, 1007, 1007, 1008, 1008],
            "card2": [170, 170, 180, 180, 190, 190],
            "P_emaildomain": ["gmail.com", "gmail.com", "aol.com", "hotmail.com", "gmail.com", "aol.com"],
            "R_emaildomain": ["gmail.com", "aol.com", "aol.com", "hotmail.com", "gmail.com", "hotmail.com"],
            "isFraud": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(dataset_root / "test_transaction.csv", index=False)

    pd.DataFrame(
        {
            "TransactionID": [21, 22, 23, 24, 25, 26],
            "DeviceType": ["desktop", "mobile", "desktop", "mobile", "desktop", "mobile"],
            "DeviceInfo": ["Windows", "Android", "Linux", "iOS", "Windows", "Android"],
            "id_01": [0.1, None, 0.3, 0.4, None, 0.6],
            "id_12": ["Found", "NotFound", "Found", "Found", "NotFound", "Found"],
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
  enabled: false
  include_train: false
  include_test: false
  high_cardinality_ratio: 0.5

eda:
  enabled: true
  include_train: true
  include_test: true
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
  mode: eda
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
    enabled: false
  eda:
    enabled: true
""".strip(),
        encoding="utf-8",
    )


def test_eda_pipeline_exports_all_requested_figures(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    interim_root = tmp_path / "interim"
    reports_root = tmp_path / "reports"
    _write_ieee_csvs(raw_root / "IEEE-CIS-FRAUD")

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_root, interim_root, reports_root)
    config = load_app_config(cfg_path)

    EDAPipeline(config=config).run()

    base_report_dir = reports_root / "eda" / "IEEE-CIS-FRAUD"
    expected_figures = [
        "fraud_distribution.png",
        "numerical_features.png",
        "categorical_features.png",
        "transaction_amount.png",
        "product_categories.png",
        "card_features.png",
        "email_domains.png",
        "identity_features.png",
        "missing_values.png",
        "correlations.png",
        "transactiondt_analysis.png",
        "fraud_by_hour.png",
        "fraud_by_day.png",
        "pca.png",
        "umap.png",
        "extra_trees_feature_importance.png",
    ]

    for split in ["train", "test"]:
        split_dir = base_report_dir / split
        figures_dir = split_dir / "figures"
        assert split_dir.exists()
        assert figures_dir.exists()

        summary_file = split_dir / "eda_artifacts.yaml"
        assert summary_file.exists()
        summary_payload = yaml.safe_load(summary_file.read_text(encoding="utf-8"))
        assert summary_payload["split"] == split

        for fig_name in expected_figures:
            fig_path = figures_dir / fig_name
            assert fig_path.exists()
            assert fig_path.stat().st_size > 0
