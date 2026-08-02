"""Scaffold smoke tests."""

from pathlib import Path

import pytest

from explainable_fl.config.loader import ConfigError, load_app_config


def test_config_file_exists() -> None:
    assert Path("configs/config.yaml").exists()


def test_logging_config_exists() -> None:
    assert Path("configs/logging.yaml").exists()


def test_typed_config_loader_reads_all_sections() -> None:
    config = load_app_config("configs/config.yaml")

    assert config.paths.data_raw == "data/raw"
    assert config.ingestion.active_dataset == "ieee_cis_fraud"
    assert "ieee_cis_fraud" in config.ingestion.datasets
    assert config.preprocessing.target_column == "target"
    assert config.feature_engineering.polynomial_degree == 1
    assert config.model_parameters.algorithm == "random_forest"
    assert "accuracy" in config.evaluation.metrics
    assert config.profiling.enabled is True
    assert config.profiling.high_cardinality_ratio > 0
    assert config.eda.enabled is True
    assert config.eda.sample_size > 0
    assert config.logging.config_path == "configs/logging.yaml"
    assert config.pipelines.profile.enabled is True
    assert config.pipelines.eda.enabled is True


def test_typed_config_loader_requires_core_sections(tmp_path: Path) -> None:
    bad_cfg = tmp_path / "config.yaml"
    bad_cfg.write_text(
        """
project:
    name: explainable-fl
paths:
    data_raw: data/raw
    data_interim: data/interim
    data_processed: data/processed
    models_dir: models
    reports_dir: reports
logging:
    config_path: configs/logging.yaml
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="ingestion"):
        load_app_config(bad_cfg)
