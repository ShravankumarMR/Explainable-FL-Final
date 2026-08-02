"""Logging setup helpers."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def configure_logging(config_path: str | None) -> None:
    if not config_path:
        logging.basicConfig(level=logging.INFO)
        return

    path = Path(config_path)
    if not path.exists():
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            "Logging config file not found at %s. Falling back to basicConfig.",
            path,
        )
        return

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if isinstance(config, dict):
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            "Invalid logging config format in %s. Falling back to basicConfig.",
            path,
        )
