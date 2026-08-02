"""Scaffold smoke tests."""

from pathlib import Path


def test_config_file_exists() -> None:
    assert Path("configs/config.yaml").exists()


def test_logging_config_exists() -> None:
    assert Path("configs/logging.yaml").exists()
