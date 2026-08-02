"""Tests for modular feature engineering generators and artifact consistency."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from explainable_fl.config.loader import load_app_config
from explainable_fl.feature_engineering.engineer import TabularFeatureEngineer


def _write_config(path: Path) -> None:
    path.write_text(
        """
project:
  name: explainable-fl
  environment: test
  random_seed: 42

paths:
  data_raw: data/raw
  data_interim: data/interim
  data_processed: data/processed
  models_dir: models
  reports_dir: reports

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
  enabled: true
  selected_features: []
  create_interactions: false
  polynomial_degree: 1
  artifacts_subdir: feature_engineering
  transaction_dt:
    enabled: true
    source_column: TransactionDT
    hour_feature: transaction_hour
    weekday_feature: transaction_weekday
    weekend_feature: is_weekend
    weekend_days: [5, 6]
  log_transaction_amount:
    enabled: true
    source_column: TransactionAmt
    output_column: log_transaction_amt
  missing_indicators:
    enabled: true
    columns: [TransactionAmt, DeviceType]
    prefix: is_missing__
  count_encoding:
    enabled: true
    columns: [card1, DeviceType]
    suffix: __count
    fill_value: 0.0
  frequency_encoding:
    enabled: true
    columns: [card1, DeviceType]
    suffix: __freq
    fill_value: 0.0
  aggregations:
    device:
      enabled: true
      group_columns: [DeviceType]
      value_columns: [TransactionAmt]
      stats: [mean, std]
    email:
      enabled: true
      group_columns: [P_emaildomain]
      value_columns: [TransactionAmt]
      stats: [mean]
    card:
      enabled: true
      group_columns: [card1]
      value_columns: [TransactionAmt]
      stats: [mean]
    address:
      enabled: true
      group_columns: [addr1]
      value_columns: [TransactionAmt]
      stats: [mean]
  interactions:
    enabled: true
    pairs:
      - left: card1
        right: addr1
        op: concat
        name: card1_x_addr1
      - left: TransactionAmt
        right: transaction_hour
        op: divide
        name: amt_per_hour

model_parameters:
  algorithm: random_forest
  hyperparameters: {}

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
    enabled: false
  train:
    enabled: false
  infer:
    enabled: false
  profile:
    enabled: false
  eda:
    enabled: false
""".strip(),
        encoding="utf-8",
    )


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [0, 3600, 86400 * 5, 86400 * 6 + 7200],
            "TransactionAmt": [100.0, 200.0, None, 400.0],
            "DeviceType": ["desktop", "mobile", None, "desktop"],
            "DeviceInfo": ["d1", "m1", "m2", None],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", None],
            "R_emaildomain": ["gmail.com", None, "yahoo.com", "gmail.com"],
            "card1": [1111, 1111, 2222, 3333],
            "addr1": [100, 100, 200, None],
            "isFraud": [0, 1, 0, 1],
        }
    )


def test_feature_engineering_generates_requested_feature_families(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    config = load_app_config(config_path)

    dataframe = _sample_dataframe()
    transformed = TabularFeatureEngineer(config).fit_transform(dataframe)

    expected_columns = {
        "transaction_hour",
        "transaction_weekday",
        "is_weekend",
        "log_transaction_amt",
        "is_missing__TransactionAmt",
        "is_missing__DeviceType",
        "card1__count",
        "card1__freq",
        "DeviceType__count",
        "DeviceType__freq",
        "device_stats__DeviceType__TransactionAmt__mean",
        "email_stats__P_emaildomain__TransactionAmt__mean",
        "card_stats__card1__TransactionAmt__mean",
        "address_stats__addr1__TransactionAmt__mean",
        "card1_x_addr1",
        "amt_per_hour",
    }
    assert expected_columns.issubset(set(transformed.columns))

    assert transformed["transaction_hour"].tolist()[:2] == [0, 1]
    assert transformed["transaction_weekday"].tolist()[2] == 5
    assert transformed["is_weekend"].tolist()[2:] == [1, 1]
    assert transformed["is_missing__TransactionAmt"].tolist() == [0, 0, 1, 0]
    assert transformed["is_missing__DeviceType"].tolist() == [0, 0, 1, 0]


def test_feature_engineering_artifacts_keep_transform_consistent(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    config = load_app_config(config_path)

    train_df = _sample_dataframe()
    test_df = train_df.copy()
    test_df.loc[0, "card1"] = 9999
    test_df.loc[1, "DeviceType"] = "tablet"

    engineer = TabularFeatureEngineer(config)
    train_transformed = engineer.fit_transform(train_df)
    test_transformed = engineer.transform(test_df)

    artifacts = engineer.save_artifacts(tmp_path / "artifacts")
    loaded = TabularFeatureEngineer(config).load_artifacts(artifacts["pickle"])
    loaded_test = loaded.transform(test_df)

    assert_frame_equal(test_transformed, loaded_test)

    feature_columns = [column for column in train_transformed.columns if column != "isFraud"]
    transformed_columns = [column for column in test_transformed.columns if column != "isFraud"]
    assert feature_columns == transformed_columns
