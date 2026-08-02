"""Data ingestion contracts and IEEE-CIS implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from explainable_fl.config.loader import AppConfig, ConfigError, DatasetIngestionConfig


@dataclass(slots=True)
class IngestionReport:
    dataset_name: str
    merge_key: str
    row_counts: dict[str, int]
    missing_identity_rows: dict[str, int]
    duplicate_transaction_ids: dict[str, int]
    memory_usage_mb: dict[str, float]
    output_files: dict[str, str]


class BaseDataIngestor(ABC):
    """Contract for data ingestion components."""

    @abstractmethod
    def run(self, config: AppConfig) -> IngestionReport:
        raise NotImplementedError


class IEEECISDataIngestor(BaseDataIngestor):
    """Reads IEEE-CIS CSV files, validates, merges, and stores parquet outputs."""

    def run(self, config: AppConfig) -> IngestionReport:
        dataset_name = config.ingestion.active_dataset
        dataset_cfg = config.ingestion.datasets.get(dataset_name)
        if dataset_cfg is None:
            raise ConfigError(
                f"Active ingestion dataset '{dataset_name}' not found in config.ingestion.datasets"
            )
        if not dataset_cfg.base_dir:
            raise ConfigError(
                f"Dataset '{dataset_name}' is missing required 'base_dir' in ingestion config"
            )

        raw_dir = Path(config.paths.data_raw) / dataset_cfg.base_dir
        interim_dir = Path(config.paths.data_interim) / dataset_name
        reports_dir = Path(config.paths.reports_dir) / "ingestion" / dataset_name
        interim_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        train_transaction = self._read_csv(raw_dir / dataset_cfg.train_transaction_file)
        train_identity = self._read_csv(raw_dir / dataset_cfg.train_identity_file)
        test_transaction = self._read_csv(raw_dir / dataset_cfg.test_transaction_file)
        test_identity = self._read_csv(raw_dir / dataset_cfg.test_identity_file)

        if config.ingestion.validate_schema:
            self._validate_schema(train_transaction, dataset_cfg.required_transaction_columns, "train_transaction")
            self._validate_schema(train_identity, dataset_cfg.required_identity_columns, "train_identity")
            self._validate_schema(test_transaction, dataset_cfg.required_transaction_columns, "test_transaction")
            self._validate_schema(test_identity, dataset_cfg.required_identity_columns, "test_identity")

        merge_key = dataset_cfg.merge_key
        self._validate_merge_key(train_transaction, train_identity, merge_key, "train")
        self._validate_merge_key(test_transaction, test_identity, merge_key, "test")

        train_merged, missing_train_identity = self._merge_with_missing_count(
            train_transaction, train_identity, merge_key
        )
        test_merged, missing_test_identity = self._merge_with_missing_count(
            test_transaction, test_identity, merge_key
        )

        train_parquet_path = interim_dir / dataset_cfg.output_train_parquet
        test_parquet_path = interim_dir / dataset_cfg.output_test_parquet
        train_merged.to_parquet(train_parquet_path, index=False)
        test_merged.to_parquet(test_parquet_path, index=False)

        report = IngestionReport(
            dataset_name=dataset_name,
            merge_key=merge_key,
            row_counts={
                "train_transaction": int(len(train_transaction)),
                "train_identity": int(len(train_identity)),
                "train_merged": int(len(train_merged)),
                "test_transaction": int(len(test_transaction)),
                "test_identity": int(len(test_identity)),
                "test_merged": int(len(test_merged)),
            },
            missing_identity_rows={
                "train": missing_train_identity,
                "test": missing_test_identity,
            },
            duplicate_transaction_ids={
                "train_transaction": self._count_duplicate_ids(train_transaction, merge_key),
                "train_identity": self._count_duplicate_ids(train_identity, merge_key),
                "test_transaction": self._count_duplicate_ids(test_transaction, merge_key),
                "test_identity": self._count_duplicate_ids(test_identity, merge_key),
            },
            memory_usage_mb={
                "train_transaction": self._memory_mb(train_transaction),
                "train_identity": self._memory_mb(train_identity),
                "train_merged": self._memory_mb(train_merged),
                "test_transaction": self._memory_mb(test_transaction),
                "test_identity": self._memory_mb(test_identity),
                "test_merged": self._memory_mb(test_merged),
            },
            output_files={
                "train_merged_parquet": str(train_parquet_path),
                "test_merged_parquet": str(test_parquet_path),
            },
        )

        report_path = reports_dir / dataset_cfg.report_file
        with report_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(report), handle, sort_keys=False)

        return report

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise ConfigError(f"Required CSV file not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _validate_schema(dataframe: pd.DataFrame, required_columns: list[str], table_name: str) -> None:
        missing_columns = [column for column in required_columns if column not in dataframe.columns]
        if missing_columns:
            raise ConfigError(
                f"Schema validation failed for {table_name}. Missing columns: {missing_columns}"
            )

    @staticmethod
    def _validate_merge_key(
        transaction_df: pd.DataFrame,
        identity_df: pd.DataFrame,
        merge_key: str,
        split_name: str,
    ) -> None:
        if merge_key not in transaction_df.columns:
            raise ConfigError(
                f"Merge key '{merge_key}' not found in {split_name}_transaction dataframe"
            )
        if merge_key not in identity_df.columns:
            raise ConfigError(
                f"Merge key '{merge_key}' not found in {split_name}_identity dataframe"
            )

    @staticmethod
    def _merge_with_missing_count(
        transaction_df: pd.DataFrame,
        identity_df: pd.DataFrame,
        merge_key: str,
    ) -> tuple[pd.DataFrame, int]:
        missing_identity = int((~transaction_df[merge_key].isin(identity_df[merge_key])).sum())
        merged = transaction_df.merge(identity_df, on=merge_key, how="left")
        return merged, missing_identity

    @staticmethod
    def _count_duplicate_ids(dataframe: pd.DataFrame, merge_key: str) -> int:
        return int(dataframe.duplicated(subset=[merge_key]).sum())

    @staticmethod
    def _memory_mb(dataframe: pd.DataFrame) -> float:
        return round(float(dataframe.memory_usage(deep=True).sum()) / (1024 * 1024), 3)
