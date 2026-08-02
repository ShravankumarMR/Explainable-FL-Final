"""Dataset preprocessing with fit/transform, encoding, and artifact persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pickle

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

from explainable_fl.config.loader import AppConfig, ConfigError


@dataclass(slots=True)
class PreprocessingArtifacts:
    """Serializable preprocessing state used for train/test-consistent transforms."""

    target_column: str
    drop_columns: list[str]
    missing_value_strategy: str
    input_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    onehot_columns: list[str]
    frequency_columns: list[str]
    numeric_fill_values: dict[str, float]
    categorical_fill_values: dict[str, str]
    onehot_levels: dict[str, list[str]]
    frequency_maps: dict[str, dict[str, float]]
    feature_columns: list[str]
    feature_dtypes: dict[str, str]


class BasePreprocessor(ABC):
    """Contract for preprocessing components."""

    @abstractmethod
    def run(self, config: dict[str, Any]) -> None:
        raise NotImplementedError


class TabularPreprocessor(BasePreprocessor):
    """Preprocessor that supports consistent fit/transform for tabular datasets."""

    MISSING_TOKEN = "__MISSING__"

    def __init__(self, config: AppConfig, max_onehot_cardinality: int = 16) -> None:
        self.config = config
        self.max_onehot_cardinality = max_onehot_cardinality
        self.artifacts: PreprocessingArtifacts | None = None

    def run(self, config: dict[str, Any]) -> None:
        raise NotImplementedError(
            "Use fit/transform methods for tabular preprocessing instead of run(config)."
        )

    def fit(self, dataframe: pd.DataFrame) -> "TabularPreprocessor":
        features, _ = self._split_target(dataframe)
        features = self._drop_configured_columns(features)

        numeric_columns, categorical_columns = self._identify_column_types(features)
        numeric_fill_values = self._compute_numeric_fill_values(features, numeric_columns)
        categorical_fill_values = {
            column: self.MISSING_TOKEN for column in categorical_columns
        }

        prepared = features.copy()
        prepared = self._apply_missing_value_handling(
            prepared,
            numeric_fill_values=numeric_fill_values,
            categorical_fill_values=categorical_fill_values,
        )

        onehot_columns: list[str] = []
        frequency_columns: list[str] = []
        onehot_levels: dict[str, list[str]] = {}
        frequency_maps: dict[str, dict[str, float]] = {}

        for column in categorical_columns:
            cardinality = int(prepared[column].nunique(dropna=False))
            if cardinality > self.max_onehot_cardinality:
                frequency_columns.append(column)
                normalized = prepared[column].astype(str).value_counts(normalize=True)
                frequency_maps[column] = {
                    str(category): float(freq) for category, freq in normalized.items()
                }
            else:
                onehot_columns.append(column)
                categories = sorted(prepared[column].astype(str).unique().tolist())
                onehot_levels[column] = categories

        transformed_features = self._transform_features(
            dataframe=features,
            input_columns=list(features.columns),
            numeric_columns=numeric_columns,
            onehot_columns=onehot_columns,
            frequency_columns=frequency_columns,
            numeric_fill_values=numeric_fill_values,
            categorical_fill_values=categorical_fill_values,
            onehot_levels=onehot_levels,
            frequency_maps=frequency_maps,
        )

        self.artifacts = PreprocessingArtifacts(
            target_column=self.config.preprocessing.target_column,
            drop_columns=list(self.config.preprocessing.drop_columns),
            missing_value_strategy=self.config.preprocessing.missing_value_strategy,
            input_columns=list(features.columns),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            onehot_columns=onehot_columns,
            frequency_columns=frequency_columns,
            numeric_fill_values=numeric_fill_values,
            categorical_fill_values=categorical_fill_values,
            onehot_levels=onehot_levels,
            frequency_maps=frequency_maps,
            feature_columns=list(transformed_features.columns),
            feature_dtypes={
                column: str(dtype) for column, dtype in transformed_features.dtypes.items()
            },
        )
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        artifacts = self._require_artifacts()
        features, target = self._split_target(dataframe)
        features = self._drop_configured_columns(features)

        transformed_features = self._transform_features(
            dataframe=features,
            input_columns=artifacts.input_columns,
            numeric_columns=artifacts.numeric_columns,
            onehot_columns=artifacts.onehot_columns,
            frequency_columns=artifacts.frequency_columns,
            numeric_fill_values=artifacts.numeric_fill_values,
            categorical_fill_values=artifacts.categorical_fill_values,
            onehot_levels=artifacts.onehot_levels,
            frequency_maps=artifacts.frequency_maps,
        )

        for column in artifacts.feature_columns:
            if column not in transformed_features.columns:
                transformed_features[column] = 0

        transformed_features = transformed_features[artifacts.feature_columns]
        transformed_features = self._cast_feature_dtypes(
            transformed_features,
            artifacts.feature_dtypes,
        )

        if target is not None:
            transformed_features[artifacts.target_column] = target

        return transformed_features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.fit(dataframe)
        return self.transform(dataframe)

    def save_artifacts(self, output_dir: Path | str) -> dict[str, str]:
        artifacts = self._require_artifacts()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        pickle_path = directory / "preprocessing_artifacts.pkl"
        yaml_path = directory / "preprocessing_artifacts.yaml"

        payload = {
            "max_onehot_cardinality": self.max_onehot_cardinality,
            "artifacts": artifacts,
        }
        with pickle_path.open("wb") as handle:
            pickle.dump(payload, handle)

        summary = {
            "target_column": artifacts.target_column,
            "missing_value_strategy": artifacts.missing_value_strategy,
            "input_columns": artifacts.input_columns,
            "numeric_columns": artifacts.numeric_columns,
            "categorical_columns": artifacts.categorical_columns,
            "onehot_columns": artifacts.onehot_columns,
            "frequency_columns": artifacts.frequency_columns,
            "feature_columns": artifacts.feature_columns,
        }
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle, sort_keys=False)

        return {
            "pickle": str(pickle_path),
            "yaml": str(yaml_path),
        }

    def load_artifacts(self, artifacts_file: Path | str) -> "TabularPreprocessor":
        path = Path(artifacts_file)
        if not path.exists():
            raise ConfigError(f"Preprocessing artifacts file not found: {path}")

        with path.open("rb") as handle:
            payload = pickle.load(handle)

        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, PreprocessingArtifacts):
            raise ConfigError(f"Invalid preprocessing artifacts payload in file: {path}")

        self.max_onehot_cardinality = int(payload.get("max_onehot_cardinality", 16))
        self.artifacts = artifacts
        return self

    def _require_artifacts(self) -> PreprocessingArtifacts:
        if self.artifacts is None:
            raise ConfigError("Preprocessor is not fitted. Call fit() or load_artifacts() first.")
        return self.artifacts

    def _split_target(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        target_column = self.config.preprocessing.target_column
        if target_column in dataframe.columns:
            return dataframe.drop(columns=[target_column]), dataframe[target_column].copy()
        return dataframe.copy(), None

    def _drop_configured_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        drop_columns = [
            column for column in self.config.preprocessing.drop_columns if column in dataframe.columns
        ]
        return dataframe.drop(columns=drop_columns)

    @staticmethod
    def _identify_column_types(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []

        for column in dataframe.columns:
            if is_numeric_dtype(dataframe[column]):
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)

        return numeric_columns, categorical_columns

    def _compute_numeric_fill_values(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
    ) -> dict[str, float]:
        strategy = self.config.preprocessing.missing_value_strategy.strip().lower()
        valid_strategies = {"median", "mean", "zero"}
        if strategy not in valid_strategies:
            raise ConfigError(
                "Unsupported preprocessing.missing_value_strategy: "
                f"{self.config.preprocessing.missing_value_strategy}. "
                f"Supported values: {sorted(valid_strategies)}"
            )

        fill_values: dict[str, float] = {}
        for column in numeric_columns:
            numeric_series = pd.to_numeric(dataframe[column], errors="coerce")
            if strategy == "median":
                value = float(numeric_series.median()) if not numeric_series.dropna().empty else 0.0
            elif strategy == "mean":
                value = float(numeric_series.mean()) if not numeric_series.dropna().empty else 0.0
            else:
                value = 0.0
            fill_values[column] = value
        return fill_values

    def _apply_missing_value_handling(
        self,
        dataframe: pd.DataFrame,
        numeric_fill_values: dict[str, float],
        categorical_fill_values: dict[str, str],
    ) -> pd.DataFrame:
        handled = dataframe.copy()

        for column, value in numeric_fill_values.items():
            if column not in handled.columns:
                handled[column] = value
            handled[column] = pd.to_numeric(handled[column], errors="coerce").fillna(value)

        for column, value in categorical_fill_values.items():
            if column not in handled.columns:
                handled[column] = value
            handled[column] = handled[column].astype("string").fillna(value)

        return handled

    def _transform_features(
        self,
        dataframe: pd.DataFrame,
        input_columns: list[str],
        numeric_columns: list[str],
        onehot_columns: list[str],
        frequency_columns: list[str],
        numeric_fill_values: dict[str, float],
        categorical_fill_values: dict[str, str],
        onehot_levels: dict[str, list[str]],
        frequency_maps: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        prepared = dataframe.copy()

        for column in input_columns:
            if column not in prepared.columns:
                prepared[column] = pd.NA
        prepared = prepared[input_columns]

        prepared = self._apply_missing_value_handling(
            prepared,
            numeric_fill_values=numeric_fill_values,
            categorical_fill_values=categorical_fill_values,
        )

        transformed = pd.DataFrame(index=prepared.index)

        for column in numeric_columns:
            transformed[column] = prepared[column]

        for column in frequency_columns:
            mapping = frequency_maps[column]
            transformed[f"{column}__freq"] = (
                prepared[column]
                .astype(str)
                .map(mapping)
                .fillna(0.0)
                .astype("float32")
            )

        for column in onehot_columns:
            source = prepared[column].astype(str)
            for level in onehot_levels[column]:
                encoded_name = f"{column}__{level}"
                transformed[encoded_name] = (source == level).astype("uint8")

        return self._optimize_data_types(transformed)

    @staticmethod
    def _optimize_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
        optimized = dataframe.copy()
        for column in optimized.columns:
            series = optimized[column]
            if pd.api.types.is_integer_dtype(series):
                optimized[column] = pd.to_numeric(series, downcast="integer")
            elif pd.api.types.is_float_dtype(series):
                optimized[column] = pd.to_numeric(series, downcast="float")
        return optimized

    @staticmethod
    def _cast_feature_dtypes(dataframe: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
        casted = dataframe.copy()
        for column, dtype in dtypes.items():
            casted[column] = casted[column].astype(dtype)
        return casted
