"""Application entry point for configuration-driven pipeline execution."""

from __future__ import annotations

import argparse
import logging

from explainable_fl.config.loader import load_app_config
from explainable_fl.pipelines.inference_pipeline import InferencePipeline
from explainable_fl.pipelines.training_pipeline import TrainingPipeline
from explainable_fl.utilities.logging_utils import configure_logging

LOGGER = logging.getLogger("explainable_fl.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explainable-FL runner")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        required=True,
        help="Pipeline mode to execute",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)

    configure_logging(config.logging.config_path)

    if args.mode == "train":
        pipeline = TrainingPipeline(config=config)
    else:
        pipeline = InferencePipeline(config=config)

    LOGGER.info("Executing %s pipeline", args.mode)
    pipeline.run()


if __name__ == "__main__":
    main()
