# Copilot Project Context (Token-Optimized)

Use this file as the first reference before scanning the repository.
Goal: avoid repeated full-tree/file scans and load only targeted files.

## Project Identity

- Name: explainable-fl
- Language: Python 3.12
- Package root: src/explainable_fl
- Main entrypoint: python -m explainable_fl.main --config configs/config.yaml --mode <ingest|train|infer>

## Core Repository Layout

- configs/
  - config.yaml
  - logging.yaml
- src/explainable_fl/
  - main.py
  - config/loader.py
  - data_ingestion/ingestor.py
  - pipelines/
    - base.py
    - ingestion_pipeline.py
    - training_pipeline.py
    - inference_pipeline.py
- tests/
  - test_scaffold.py
  - test_ieee_cis_ingestion.py

## Current Working Configuration Contract

Top-level sections in configs/config.yaml:

- project
- paths
- ingestion
- preprocessing
- feature_engineering
- model_parameters
- evaluation
- run
- logging
- pipelines

Typed loading:

- load_app_config in src/explainable_fl/config/loader.py
- AppConfig dataclass is the canonical runtime configuration object

## Ingestion System (IEEE-CIS)

Active dataset key:

- ingestion.active_dataset: ieee_cis_fraud

Dataset map entry:

- ingestion.datasets.ieee_cis_fraud.base_dir: IEEE-CIS-FRAUD
- expected CSV files:
  - train_transaction.csv
  - train_identity.csv
  - test_transaction.csv
  - test_identity.csv
- merge key: TransactionID

Pipeline behavior:

- Ingestion mode runs IngestionPipeline -> IEEECISDataIngestor
- Train mode currently runs ingestion stage first (scaffold training)

Outputs:

- data/interim/ieee_cis_fraud/train_merged.parquet
- data/interim/ieee_cis_fraud/test_merged.parquet
- reports/ingestion/ieee_cis_fraud/ingestion_report.yaml

Report includes:

- row_counts
- missing_identity_rows
- duplicate_transaction_ids
- memory_usage_mb

## Data and Git Policy

- Entire data/ directory is gitignored
- Do not commit raw/interim dataset files

## Fast Retrieval Guide for Future Sessions

1. Read this file first.
2. If config question: read configs/config.yaml and src/explainable_fl/config/loader.py only.
3. If ingestion question: read src/explainable_fl/data_ingestion/ingestor.py and src/explainable_fl/pipelines/ingestion_pipeline.py.
4. If runtime/CLI question: read src/explainable_fl/main.py.
5. If validation question: read tests/test_ieee_cis_ingestion.py and tests/test_scaffold.py.

## Standard Commands

- Run ingestion:
  - python -m explainable_fl.main --config configs/config.yaml --mode ingest
- Run tests:
  - python -m pytest -q

## Extension Pattern (Future Datasets)

To add a new raw dataset folder:

1. Add a new key under ingestion.datasets in configs/config.yaml.
2. Set base_dir and file names for that dataset.
3. Switch ingestion.active_dataset to the new key.
4. Reuse existing ingestion pipeline and report flow.
