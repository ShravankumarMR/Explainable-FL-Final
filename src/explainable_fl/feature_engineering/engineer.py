"""Configurable tabular feature engineering with train/test-consistent artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pickle

import numpy as np
import pandas as pd
import yaml

from explainable_fl.config.loader import AppConfig, ConfigError


@dataclass(slots=True)
class FeatureEngineeringArtifacts:
    """Serializable state used to keep train/inference feature logic aligned."""

    target_column: str
    feature_columns: list[str]
    count_maps: dict[str, dict[str, float]]
    frequency_maps: dict[str, dict[str, float]]
    aggregation_maps: dict[str, dict[str, dict[str, float]]]
    aggregation_defaults: dict[str, float]
    feature_dtypes: dict[str, str]


@dataclass(slots=True)
class TransactionDTConfig:
    enabled: bool = True
    source_column: str = "TransactionDT"
    hour_feature: str = "transaction_hour"
    weekday_feature: str = "transaction_weekday"
    weekend_feature: str = "is_weekend"
    weekend_days: tuple[int, ...] = (5, 6)


@dataclass(slots=True)
class LogAmountConfig:
    enabled: bool = True
    source_column: str = "TransactionAmt"
    output_column: str = "log_transaction_amt"


@dataclass(slots=True)
class MissingIndicatorConfig:
    enabled: bool = True
    columns: list[str] | None = None
    prefix: str = "is_missing__"


@dataclass(slots=True)
class CountEncodingConfig:
    enabled: bool = True
    columns: list[str] | None = None
    suffix: str = "__count"
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        if self.columns is None:
            self.columns = [
                "card1",
                "card2",
                "addr1",
                "addr2",
                "P_emaildomain",
                "R_emaildomain",
                "DeviceType",
                "DeviceInfo",
            ]


@dataclass(slots=True)
class FrequencyEncodingConfig:
    enabled: bool = True
    columns: list[str] | None = None
    suffix: str = "__freq"
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        if self.columns is None:
            self.columns = [
                "card1",
                "card2",
                "addr1",
                "addr2",
                "P_emaildomain",
                "R_emaildomain",
                "DeviceType",
                "DeviceInfo",
            ]


@dataclass(slots=True)
class AggregationConfig:
    enabled: bool = True
    group_columns: list[str] | None = None
    value_columns: list[str] | None = None
    stats: list[str] | None = None
    prefix: str = "agg"

    def __post_init__(self) -> None:
        if self.group_columns is None:
            self.group_columns = []
        if self.value_columns is None:
            self.value_columns = ["TransactionAmt"]
        if self.stats is None:
            self.stats = ["mean", "std", "min", "max"]


@dataclass(slots=True)
class InteractionFeatureConfig:
    enabled: bool = True
    interactions: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.interactions is None:
            self.interactions = [
                {"left": "card1", "right": "addr1", "op": "concat", "name": "card1_x_addr1"},
                {
                    "left": "TransactionAmt",
                    "right": "transaction_hour",
                    "op": "divide",
                    "name": "amt_per_hour",
                },
            ]


class BaseFeatureGenerator(ABC):
    """Feature generator contract with optional fitted state."""

    @abstractmethod
    def fit(self, dataframe: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TransactionDTFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, config: TransactionDTConfig) -> None:
        self.config = config

    def fit(self, dataframe: pd.DataFrame) -> None:
        return None

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled or self.config.source_column not in dataframe.columns:
            return dataframe

        dt_seconds = pd.to_numeric(dataframe[self.config.source_column], errors="coerce")
        dataframe[self.config.hour_feature] = ((dt_seconds // 3600) % 24).fillna(-1).astype("int16")
        dataframe[self.config.weekday_feature] = ((dt_seconds // 86400) % 7).fillna(-1).astype("int16")
        dataframe[self.config.weekend_feature] = (
            dataframe[self.config.weekday_feature].isin(self.config.weekend_days).astype("uint8")
        )
        return dataframe


class LogTransactionAmountGenerator(BaseFeatureGenerator):
    def __init__(self, config: LogAmountConfig) -> None:
        self.config = config

    def fit(self, dataframe: pd.DataFrame) -> None:
        return None

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled or self.config.source_column not in dataframe.columns:
            return dataframe

        amount = pd.to_numeric(dataframe[self.config.source_column], errors="coerce").fillna(0.0)
        dataframe[self.config.output_column] = np.log1p(amount.abs()).astype("float32")
        return dataframe


class MissingIndicatorGenerator(BaseFeatureGenerator):
    def __init__(self, config: MissingIndicatorConfig) -> None:
        self.config = config

    def fit(self, dataframe: pd.DataFrame) -> None:
        return None

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled:
            return dataframe

        columns = self.config.columns or list(dataframe.columns)
        for column in columns:
            if column in dataframe.columns:
                dataframe[f"{self.config.prefix}{column}"] = dataframe[column].isna().astype("uint8")
        return dataframe


class CountEncodingGenerator(BaseFeatureGenerator):
    def __init__(self, config: CountEncodingConfig) -> None:
        self.config = config
        self.maps: dict[str, dict[str, float]] = {}

    def fit(self, dataframe: pd.DataFrame) -> None:
        if not self.config.enabled:
            return
        self.maps = {}
        for column in self.config.columns:
            if column in dataframe.columns:
                counts = dataframe[column].astype("string").value_counts(dropna=False)
                self.maps[column] = {str(key): float(value) for key, value in counts.items()}

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled:
            return dataframe
        for column in self.config.columns:
            if column in dataframe.columns and column in self.maps:
                dataframe[f"{column}{self.config.suffix}"] = (
                    dataframe[column]
                    .astype("string")
                    .map(self.maps[column])
                    .fillna(self.config.fill_value)
                    .astype("float32")
                )
        return dataframe


class FrequencyEncodingGenerator(BaseFeatureGenerator):
    def __init__(self, config: FrequencyEncodingConfig) -> None:
        self.config = config
        self.maps: dict[str, dict[str, float]] = {}

    def fit(self, dataframe: pd.DataFrame) -> None:
        if not self.config.enabled:
            return
        self.maps = {}
        for column in self.config.columns:
            if column in dataframe.columns:
                freqs = dataframe[column].astype("string").value_counts(normalize=True, dropna=False)
                self.maps[column] = {str(key): float(value) for key, value in freqs.items()}

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled:
            return dataframe
        for column in self.config.columns:
            if column in dataframe.columns and column in self.maps:
                dataframe[f"{column}{self.config.suffix}"] = (
                    dataframe[column]
                    .astype("string")
                    .map(self.maps[column])
                    .fillna(self.config.fill_value)
                    .astype("float32")
                )
        return dataframe


class AggregationFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, config: AggregationConfig) -> None:
        self.config = config
        self.maps: dict[str, dict[str, dict[str, float]]] = {}
        self.defaults: dict[str, float] = {}

    def fit(self, dataframe: pd.DataFrame) -> None:
        if not self.config.enabled:
            return

        valid_stats = {"mean", "std", "min", "max", "median"}
        invalid_stats = [stat for stat in self.config.stats if stat not in valid_stats]
        if invalid_stats:
            raise ConfigError(
                "Unsupported aggregation stats: "
                f"{invalid_stats}. Supported values: {sorted(valid_stats)}"
            )

        self.maps = {}
        self.defaults = {}
        for group_column in self.config.group_columns:
            if group_column not in dataframe.columns:
                continue

            for value_column in self.config.value_columns:
                if value_column not in dataframe.columns:
                    continue

                numeric_series = pd.to_numeric(dataframe[value_column], errors="coerce")
                key = f"{group_column}::{value_column}"
                grouped = (
                    pd.DataFrame({group_column: dataframe[group_column], value_column: numeric_series})
                    .groupby(group_column)[value_column]
                    .agg(self.config.stats)
                )

                self.maps[key] = {}
                for stat in self.config.stats:
                    values = grouped[stat].astype("float64").fillna(0.0)
                    self.maps[key][stat] = {str(idx): float(val) for idx, val in values.items()}
                    if numeric_series.dropna().empty:
                        global_val = 0.0
                    else:
                        stat_value = float(getattr(numeric_series, stat)())
                        global_val = 0.0 if pd.isna(stat_value) else stat_value
                    self.defaults[f"{key}::{stat}"] = global_val

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled:
            return dataframe

        for key, stat_maps in self.maps.items():
            group_column, value_column = key.split("::", maxsplit=1)
            if group_column not in dataframe.columns:
                continue

            group_series = dataframe[group_column].astype("string")
            for stat, mapping in stat_maps.items():
                feature_name = f"{self.config.prefix}__{group_column}__{value_column}__{stat}"
                default_key = f"{key}::{stat}"
                dataframe[feature_name] = (
                    group_series.map(mapping)
                    .fillna(self.defaults.get(default_key, 0.0))
                    .astype("float32")
                )
        return dataframe


class InteractionFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, config: InteractionFeatureConfig) -> None:
        self.config = config

    def fit(self, dataframe: pd.DataFrame) -> None:
        return None

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled:
            return dataframe

        for interaction in self.config.interactions:
            left = str(interaction.get("left", "")).strip()
            right = str(interaction.get("right", "")).strip()
            operation = str(interaction.get("op", "concat")).strip().lower()
            output = str(interaction.get("name", f"{left}_x_{right}_{operation}")).strip()

            if not left or not right or left not in dataframe.columns or right not in dataframe.columns:
                continue

            if operation == "concat":
                dataframe[output] = (
                    dataframe[left].astype("string").fillna("__NA__")
                    + "_"
                    + dataframe[right].astype("string").fillna("__NA__")
                )
            elif operation == "multiply":
                left_num = pd.to_numeric(dataframe[left], errors="coerce").fillna(0.0)
                right_num = pd.to_numeric(dataframe[right], errors="coerce").fillna(0.0)
                dataframe[output] = (left_num * right_num).astype("float32")
            elif operation == "divide":
                left_num = pd.to_numeric(dataframe[left], errors="coerce").fillna(0.0)
                right_num = pd.to_numeric(dataframe[right], errors="coerce").replace(0, pd.NA)
                dataframe[output] = left_num.divide(right_num).fillna(0.0).astype("float32")
            elif operation == "difference":
                left_num = pd.to_numeric(dataframe[left], errors="coerce").fillna(0.0)
                right_num = pd.to_numeric(dataframe[right], errors="coerce").fillna(0.0)
                dataframe[output] = (left_num - right_num).astype("float32")

        return dataframe


class BaseFeatureEngineer(ABC):
    """Contract for feature engineering components."""

    @abstractmethod
    def run(self, config: dict[str, Any]) -> None:
        raise NotImplementedError


class TabularFeatureEngineer(BaseFeatureEngineer):
    """Orchestrates independent feature generators with train/test-consistent state."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.artifacts: FeatureEngineeringArtifacts | None = None

        feature_cfg = self.config.feature_engineering
        if not feature_cfg.enabled:
            self._generators = []
            return

        aggregations_cfg = feature_cfg.aggregations
        interactions_enabled = bool(
            feature_cfg.interactions.get("enabled", feature_cfg.create_interactions)
        )
        self._generators: list[BaseFeatureGenerator] = [
            TransactionDTFeatureGenerator(
                TransactionDTConfig(
                    enabled=bool(feature_cfg.transaction_dt.get("enabled", True)),
                    source_column=str(feature_cfg.transaction_dt.get("source_column", "TransactionDT")),
                    hour_feature=str(feature_cfg.transaction_dt.get("hour_feature", "transaction_hour")),
                    weekday_feature=str(
                        feature_cfg.transaction_dt.get("weekday_feature", "transaction_weekday")
                    ),
                    weekend_feature=str(feature_cfg.transaction_dt.get("weekend_feature", "is_weekend")),
                    weekend_days=tuple(feature_cfg.transaction_dt.get("weekend_days", [5, 6])),
                )
            ),
            LogTransactionAmountGenerator(
                LogAmountConfig(
                    enabled=bool(feature_cfg.log_transaction_amount.get("enabled", True)),
                    source_column=str(
                        feature_cfg.log_transaction_amount.get("source_column", "TransactionAmt")
                    ),
                    output_column=str(
                        feature_cfg.log_transaction_amount.get(
                            "output_column", "log_transaction_amt"
                        )
                    ),
                )
            ),
            MissingIndicatorGenerator(
                MissingIndicatorConfig(
                    enabled=bool(feature_cfg.missing_indicators.get("enabled", True)),
                    columns=feature_cfg.missing_indicators.get("columns"),
                    prefix=str(feature_cfg.missing_indicators.get("prefix", "is_missing__")),
                )
            ),
            CountEncodingGenerator(
                CountEncodingConfig(
                    enabled=bool(feature_cfg.count_encoding.get("enabled", True)),
                    columns=list(feature_cfg.count_encoding.get("columns", [])) or None,
                    suffix=str(feature_cfg.count_encoding.get("suffix", "__count")),
                    fill_value=float(feature_cfg.count_encoding.get("fill_value", 0.0)),
                )
            ),
            FrequencyEncodingGenerator(
                FrequencyEncodingConfig(
                    enabled=bool(feature_cfg.frequency_encoding.get("enabled", True)),
                    columns=list(feature_cfg.frequency_encoding.get("columns", [])) or None,
                    suffix=str(feature_cfg.frequency_encoding.get("suffix", "__freq")),
                    fill_value=float(feature_cfg.frequency_encoding.get("fill_value", 0.0)),
                )
            ),
            AggregationFeatureGenerator(
                AggregationConfig(
                    enabled=bool(aggregations_cfg.get("device", {}).get("enabled", True)),
                    group_columns=list(
                        aggregations_cfg.get("device", {}).get(
                            "group_columns", ["DeviceType", "DeviceInfo"]
                        )
                    ),
                    value_columns=list(
                        aggregations_cfg.get("device", {}).get("value_columns", ["TransactionAmt"])
                    ),
                    stats=list(aggregations_cfg.get("device", {}).get("stats", ["mean", "std"])),
                    prefix="device_stats",
                )
            ),
            AggregationFeatureGenerator(
                AggregationConfig(
                    enabled=bool(aggregations_cfg.get("email", {}).get("enabled", True)),
                    group_columns=list(
                        aggregations_cfg.get("email", {}).get(
                            "group_columns", ["P_emaildomain", "R_emaildomain"]
                        )
                    ),
                    value_columns=list(
                        aggregations_cfg.get("email", {}).get("value_columns", ["TransactionAmt"])
                    ),
                    stats=list(aggregations_cfg.get("email", {}).get("stats", ["mean", "std"])),
                    prefix="email_stats",
                )
            ),
            AggregationFeatureGenerator(
                AggregationConfig(
                    enabled=bool(aggregations_cfg.get("card", {}).get("enabled", True)),
                    group_columns=list(
                        aggregations_cfg.get("card", {}).get(
                            "group_columns", ["card1", "card2", "card3", "card5"]
                        )
                    ),
                    value_columns=list(
                        aggregations_cfg.get("card", {}).get("value_columns", ["TransactionAmt"])
                    ),
                    stats=list(aggregations_cfg.get("card", {}).get("stats", ["mean", "std"])),
                    prefix="card_stats",
                )
            ),
            AggregationFeatureGenerator(
                AggregationConfig(
                    enabled=bool(aggregations_cfg.get("address", {}).get("enabled", True)),
                    group_columns=list(
                        aggregations_cfg.get("address", {}).get("group_columns", ["addr1", "addr2"])
                    ),
                    value_columns=list(
                        aggregations_cfg.get("address", {}).get("value_columns", ["TransactionAmt"])
                    ),
                    stats=list(aggregations_cfg.get("address", {}).get("stats", ["mean", "std"])),
                    prefix="address_stats",
                )
            ),
            InteractionFeatureGenerator(
                InteractionFeatureConfig(
                    enabled=interactions_enabled,
                    interactions=list(feature_cfg.interactions.get("pairs", [])) or None,
                )
            ),
        ]

    def run(self, config: dict[str, Any]) -> None:
        raise NotImplementedError("Use fit/transform methods for tabular feature engineering.")

    def fit(self, dataframe: pd.DataFrame) -> "TabularFeatureEngineer":
        features, target = self._split_target(dataframe)
        transformed = features.copy()

        for generator in self._generators:
            generator.fit(transformed)
            transformed = generator.transform(transformed)

        self.artifacts = FeatureEngineeringArtifacts(
            target_column=self.config.preprocessing.target_column,
            feature_columns=list(transformed.columns),
            count_maps=self._collect_count_maps(),
            frequency_maps=self._collect_frequency_maps(),
            aggregation_maps=self._collect_aggregation_maps(),
            aggregation_defaults=self._collect_aggregation_defaults(),
            feature_dtypes={column: str(dtype) for column, dtype in transformed.dtypes.items()},
        )

        if target is not None:
            transformed[self.artifacts.target_column] = target
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        artifacts = self._require_artifacts()
        features, target = self._split_target(dataframe)
        transformed = features.copy()

        self._restore_fitted_state(artifacts)
        for generator in self._generators:
            transformed = generator.transform(transformed)

        for column in artifacts.feature_columns:
            if column not in transformed.columns:
                transformed[column] = 0
        transformed = transformed[artifacts.feature_columns]

        if target is not None:
            transformed[artifacts.target_column] = target

        return transformed

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.fit(dataframe)
        return self.transform(dataframe)

    def save_artifacts(self, output_dir: Path | str) -> dict[str, str]:
        artifacts = self._require_artifacts()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        pickle_path = directory / "feature_engineering_artifacts.pkl"
        yaml_path = directory / "feature_engineering_artifacts.yaml"
        summary_csv_path = directory / "feature_summary.csv"

        with pickle_path.open("wb") as handle:
            pickle.dump({"artifacts": artifacts}, handle)

        summary = {
            "target_column": artifacts.target_column,
            "feature_columns": artifacts.feature_columns,
            "count_encoded_columns": sorted(artifacts.count_maps.keys()),
            "frequency_encoded_columns": sorted(artifacts.frequency_maps.keys()),
            "aggregation_groups": sorted(artifacts.aggregation_maps.keys()),
        }
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle, sort_keys=False)

        feature_summary = pd.DataFrame(
            {
                "feature": artifacts.feature_columns,
                "dtype": [artifacts.feature_dtypes[column] for column in artifacts.feature_columns],
                "is_count_encoded": [
                    any(column == f"{base}__count" for base in artifacts.count_maps)
                    for column in artifacts.feature_columns
                ],
                "is_frequency_encoded": [
                    any(column == f"{base}__freq" for base in artifacts.frequency_maps)
                    for column in artifacts.feature_columns
                ],
            }
        )
        feature_summary.to_csv(summary_csv_path, index=False)

        return {
            "pickle": str(pickle_path),
            "yaml": str(yaml_path),
            "feature_summary_csv": str(summary_csv_path),
        }

    def load_artifacts(self, artifacts_file: Path | str) -> "TabularFeatureEngineer":
        path = Path(artifacts_file)
        if not path.exists():
            raise ConfigError(f"Feature engineering artifacts file not found: {path}")

        with path.open("rb") as handle:
            payload = pickle.load(handle)

        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, FeatureEngineeringArtifacts):
            raise ConfigError(f"Invalid feature engineering payload in file: {path}")

        self.artifacts = artifacts
        self._restore_fitted_state(artifacts)
        return self

    def _require_artifacts(self) -> FeatureEngineeringArtifacts:
        if self.artifacts is None:
            raise ConfigError("Feature engineer is not fitted. Call fit() or load_artifacts() first.")
        return self.artifacts

    def _split_target(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        target_column = self.config.preprocessing.target_column
        if target_column in dataframe.columns:
            return dataframe.drop(columns=[target_column]), dataframe[target_column].copy()
        return dataframe.copy(), None

    def _collect_count_maps(self) -> dict[str, dict[str, float]]:
        for generator in self._generators:
            if isinstance(generator, CountEncodingGenerator):
                return generator.maps
        return {}

    def _collect_frequency_maps(self) -> dict[str, dict[str, float]]:
        for generator in self._generators:
            if isinstance(generator, FrequencyEncodingGenerator):
                return generator.maps
        return {}

    def _collect_aggregation_maps(self) -> dict[str, dict[str, dict[str, float]]]:
        all_maps: dict[str, dict[str, dict[str, float]]] = {}
        for generator in self._generators:
            if isinstance(generator, AggregationFeatureGenerator):
                all_maps.update(generator.maps)
        return all_maps

    def _collect_aggregation_defaults(self) -> dict[str, float]:
        all_defaults: dict[str, float] = {}
        for generator in self._generators:
            if isinstance(generator, AggregationFeatureGenerator):
                all_defaults.update(generator.defaults)
        return all_defaults

    def _restore_fitted_state(self, artifacts: FeatureEngineeringArtifacts) -> None:
        for generator in self._generators:
            if isinstance(generator, CountEncodingGenerator):
                generator.maps = artifacts.count_maps
            elif isinstance(generator, FrequencyEncodingGenerator):
                generator.maps = artifacts.frequency_maps
            elif isinstance(generator, AggregationFeatureGenerator):
                scoped_maps = {
                    key: value
                    for key, value in artifacts.aggregation_maps.items()
                    if any(key.startswith(f"{col}::") for col in generator.config.group_columns)
                }
                scoped_defaults = {
                    key: value
                    for key, value in artifacts.aggregation_defaults.items()
                    if any(key.startswith(f"{col}::") for col in generator.config.group_columns)
                }
                generator.maps = scoped_maps
                generator.defaults = scoped_defaults
