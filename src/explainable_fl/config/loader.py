"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration loading fails."""


@dataclass(slots=True)
class ProjectConfig:
    name: str = "explainable-fl"
    environment: str = "dev"
    random_seed: int = 42


@dataclass(slots=True)
class PathsConfig:
    data_raw: str
    data_interim: str
    data_processed: str
    models_dir: str
    reports_dir: str


@dataclass(slots=True)
class PreprocessingConfig:
    target_column: str = "target"
    drop_columns: list[str] = field(default_factory=list)
    missing_value_strategy: str = "median"
    scale_numeric: bool = True
    categorical_encoding: str = "onehot"


@dataclass(slots=True)
class FeatureEngineeringConfig:
    selected_features: list[str] = field(default_factory=list)
    create_interactions: bool = False
    polynomial_degree: int = 1


@dataclass(slots=True)
class ModelParametersConfig:
    algorithm: str = "random_forest"
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])
    test_size: float = 0.2
    cross_validation_folds: int = 5


@dataclass(slots=True)
class ProfilingConfig:
    enabled: bool = True
    include_train: bool = True
    include_test: bool = True
    high_cardinality_ratio: float = 0.8


@dataclass(slots=True)
class EDAConfig:
    enabled: bool = True
    include_train: bool = True
    include_test: bool = True
    target_column: str = "isFraud"
    sample_size: int = 20000
    max_numerical_features: int = 20
    max_categorical_features: int = 16
    max_category_levels: int = 12
    correlation_top_n: int = 40
    pca_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    feature_importance_top_n: int = 20
    random_state: int = 42


@dataclass(slots=True)
class LoggingConfig:
    config_path: str | None = None
    level: str = "INFO"


@dataclass(slots=True)
class DatasetIngestionConfig:
    base_dir: str
    train_transaction_file: str = "train_transaction.csv"
    train_identity_file: str = "train_identity.csv"
    test_transaction_file: str = "test_transaction.csv"
    test_identity_file: str = "test_identity.csv"
    merge_key: str = "TransactionID"
    required_transaction_columns: list[str] = field(default_factory=lambda: ["TransactionID"])
    required_identity_columns: list[str] = field(default_factory=lambda: ["TransactionID"])
    output_train_parquet: str = "train_merged.parquet"
    output_test_parquet: str = "test_merged.parquet"
    report_file: str = "ingestion_report.yaml"


@dataclass(slots=True)
class IngestionConfig:
    active_dataset: str
    datasets: dict[str, DatasetIngestionConfig]
    validate_schema: bool = True


@dataclass(slots=True)
class PipelineToggleConfig:
    enabled: bool = True


@dataclass(slots=True)
class PipelinesConfig:
    ingest: PipelineToggleConfig = field(default_factory=PipelineToggleConfig)
    train: PipelineToggleConfig = field(default_factory=PipelineToggleConfig)
    infer: PipelineToggleConfig = field(default_factory=PipelineToggleConfig)
    profile: PipelineToggleConfig = field(default_factory=PipelineToggleConfig)
    eda: PipelineToggleConfig = field(default_factory=PipelineToggleConfig)


@dataclass(slots=True)
class RunConfig:
    mode: str = "train"
    experiment_name: str = "baseline"


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig
    paths: PathsConfig
    ingestion: IngestionConfig
    preprocessing: PreprocessingConfig
    feature_engineering: FeatureEngineeringConfig
    model_parameters: ModelParametersConfig
    evaluation: EvaluationConfig
    profiling: ProfilingConfig
    eda: EDAConfig
    logging: LoggingConfig
    run: RunConfig = field(default_factory=RunConfig)
    pipelines: PipelinesConfig = field(default_factory=PipelinesConfig)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML config must be a mapping/object")

    return data


def _read_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name)
    if section is None:
        raise ConfigError(f"Missing required config section: '{section_name}'")
    if not isinstance(section, dict):
        raise ConfigError(f"Config section '{section_name}' must be a mapping/object")
    return section


def load_app_config(path: str | Path) -> AppConfig:
    raw = load_yaml_config(path)

    project_raw = raw.get("project", {})
    if not isinstance(project_raw, dict):
        raise ConfigError("Config section 'project' must be a mapping/object")

    paths_raw = _read_section(raw, "paths")
    ingestion_raw = _read_section(raw, "ingestion")
    preprocessing_raw = _read_section(raw, "preprocessing")
    feature_engineering_raw = _read_section(raw, "feature_engineering")
    model_parameters_raw = _read_section(raw, "model_parameters")
    evaluation_raw = _read_section(raw, "evaluation")
    profiling_raw = raw.get("profiling", {})
    if not isinstance(profiling_raw, dict):
        raise ConfigError("Config section 'profiling' must be a mapping/object")
    eda_raw = raw.get("eda", {})
    if not isinstance(eda_raw, dict):
        raise ConfigError("Config section 'eda' must be a mapping/object")
    logging_raw = _read_section(raw, "logging")

    run_raw = raw.get("run", {})
    if not isinstance(run_raw, dict):
        raise ConfigError("Config section 'run' must be a mapping/object")

    pipelines_raw = raw.get("pipelines", {})
    if not isinstance(pipelines_raw, dict):
        raise ConfigError("Config section 'pipelines' must be a mapping/object")

    if not isinstance(ingestion_raw.get("datasets"), dict):
        raise ConfigError("Config section 'ingestion.datasets' must be a mapping/object")

    dataset_map: dict[str, DatasetIngestionConfig] = {}
    for dataset_name, dataset_raw in ingestion_raw["datasets"].items():
        if not isinstance(dataset_raw, dict):
            raise ConfigError(
                f"Ingestion dataset '{dataset_name}' must be a mapping/object"
            )
        dataset_map[str(dataset_name)] = DatasetIngestionConfig(
            base_dir=str(dataset_raw.get("base_dir", "")),
            train_transaction_file=str(
                dataset_raw.get("train_transaction_file", "train_transaction.csv")
            ),
            train_identity_file=str(dataset_raw.get("train_identity_file", "train_identity.csv")),
            test_transaction_file=str(
                dataset_raw.get("test_transaction_file", "test_transaction.csv")
            ),
            test_identity_file=str(dataset_raw.get("test_identity_file", "test_identity.csv")),
            merge_key=str(dataset_raw.get("merge_key", "TransactionID")),
            required_transaction_columns=list(
                dataset_raw.get("required_transaction_columns", ["TransactionID"])
            ),
            required_identity_columns=list(
                dataset_raw.get("required_identity_columns", ["TransactionID"])
            ),
            output_train_parquet=str(dataset_raw.get("output_train_parquet", "train_merged.parquet")),
            output_test_parquet=str(dataset_raw.get("output_test_parquet", "test_merged.parquet")),
            report_file=str(dataset_raw.get("report_file", "ingestion_report.yaml")),
        )

    train_raw = pipelines_raw.get("train", {})
    infer_raw = pipelines_raw.get("infer", {})
    ingest_raw = pipelines_raw.get("ingest", {})
    profile_raw = pipelines_raw.get("profile", {})
    eda_pipeline_raw = pipelines_raw.get("eda", {})
    if (
        not isinstance(train_raw, dict)
        or not isinstance(infer_raw, dict)
        or not isinstance(ingest_raw, dict)
        or not isinstance(profile_raw, dict)
        or not isinstance(eda_pipeline_raw, dict)
    ):
        raise ConfigError(
            "Pipeline sections 'ingest', 'train', 'infer', 'profile', and 'eda' must be mappings/objects"
        )

    try:
        return AppConfig(
            project=ProjectConfig(
                name=str(project_raw.get("name", "explainable-fl")),
                environment=str(project_raw.get("environment", "dev")),
                random_seed=int(project_raw.get("random_seed", 42)),
            ),
            paths=PathsConfig(
                data_raw=str(paths_raw["data_raw"]),
                data_interim=str(paths_raw["data_interim"]),
                data_processed=str(paths_raw["data_processed"]),
                models_dir=str(paths_raw["models_dir"]),
                reports_dir=str(paths_raw["reports_dir"]),
            ),
            ingestion=IngestionConfig(
                active_dataset=str(ingestion_raw.get("active_dataset", "")),
                datasets=dataset_map,
                validate_schema=bool(ingestion_raw.get("validate_schema", True)),
            ),
            preprocessing=PreprocessingConfig(
                target_column=str(preprocessing_raw.get("target_column", "target")),
                drop_columns=list(preprocessing_raw.get("drop_columns", [])),
                missing_value_strategy=str(
                    preprocessing_raw.get("missing_value_strategy", "median")
                ),
                scale_numeric=bool(preprocessing_raw.get("scale_numeric", True)),
                categorical_encoding=str(
                    preprocessing_raw.get("categorical_encoding", "onehot")
                ),
            ),
            feature_engineering=FeatureEngineeringConfig(
                selected_features=list(feature_engineering_raw.get("selected_features", [])),
                create_interactions=bool(
                    feature_engineering_raw.get("create_interactions", False)
                ),
                polynomial_degree=int(feature_engineering_raw.get("polynomial_degree", 1)),
            ),
            model_parameters=ModelParametersConfig(
                algorithm=str(model_parameters_raw.get("algorithm", "random_forest")),
                hyperparameters=dict(model_parameters_raw.get("hyperparameters", {})),
            ),
            evaluation=EvaluationConfig(
                metrics=list(evaluation_raw.get("metrics", ["accuracy"])),
                test_size=float(evaluation_raw.get("test_size", 0.2)),
                cross_validation_folds=int(evaluation_raw.get("cross_validation_folds", 5)),
            ),
            profiling=ProfilingConfig(
                enabled=bool(profiling_raw.get("enabled", True)),
                include_train=bool(profiling_raw.get("include_train", True)),
                include_test=bool(profiling_raw.get("include_test", True)),
                high_cardinality_ratio=float(
                    profiling_raw.get("high_cardinality_ratio", 0.8)
                ),
            ),
            eda=EDAConfig(
                enabled=bool(eda_raw.get("enabled", True)),
                include_train=bool(eda_raw.get("include_train", True)),
                include_test=bool(eda_raw.get("include_test", True)),
                target_column=str(
                    eda_raw.get("target_column", preprocessing_raw.get("target_column", "isFraud"))
                ),
                sample_size=int(eda_raw.get("sample_size", 20000)),
                max_numerical_features=int(eda_raw.get("max_numerical_features", 20)),
                max_categorical_features=int(eda_raw.get("max_categorical_features", 16)),
                max_category_levels=int(eda_raw.get("max_category_levels", 12)),
                correlation_top_n=int(eda_raw.get("correlation_top_n", 40)),
                pca_components=int(eda_raw.get("pca_components", 2)),
                umap_n_neighbors=int(eda_raw.get("umap_n_neighbors", 15)),
                umap_min_dist=float(eda_raw.get("umap_min_dist", 0.1)),
                feature_importance_top_n=int(eda_raw.get("feature_importance_top_n", 20)),
                random_state=int(eda_raw.get("random_state", project_raw.get("random_seed", 42))),
            ),
            logging=LoggingConfig(
                config_path=(
                    str(logging_raw["config_path"])
                    if logging_raw.get("config_path") is not None
                    else None
                ),
                level=str(logging_raw.get("level", "INFO")),
            ),
            run=RunConfig(
                mode=str(run_raw.get("mode", "train")),
                experiment_name=str(run_raw.get("experiment_name", "baseline")),
            ),
            pipelines=PipelinesConfig(
                ingest=PipelineToggleConfig(enabled=bool(ingest_raw.get("enabled", True))),
                train=PipelineToggleConfig(enabled=bool(train_raw.get("enabled", True))),
                infer=PipelineToggleConfig(enabled=bool(infer_raw.get("enabled", True))),
                profile=PipelineToggleConfig(enabled=bool(profile_raw.get("enabled", True))),
                eda=PipelineToggleConfig(enabled=bool(eda_pipeline_raw.get("enabled", True))),
            ),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required config key: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value in YAML config: {exc}") from exc
