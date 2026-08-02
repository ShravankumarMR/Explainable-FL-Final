# Explainable-FL

Production-ready machine learning scaffold with a src layout, configuration-driven architecture, reusable pipelines, structured logging, and modular components.

## Python Version

- Python 3.12

## Project Goals

- Configuration-first execution via YAML
- Reusable pipeline abstraction for training and inference
- Modular package boundaries for ML lifecycle stages
- Experiment-friendly structure for notebooks, reports, and model artifacts
- SOLID-oriented interfaces and separation of concerns

## Structure

```text
.
|-- configs/
|   |-- config.yaml
|   `-- logging.yaml
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|   `-- explainable_fl/
|       |-- config/
|       |-- data_ingestion/
|       |-- preprocessing/
|       |-- feature_engineering/
|       |-- eda/
|       |-- model_training/
|       |-- evaluation/
|       |-- inference/
|       |-- pipelines/
|       `-- utilities/
`-- tests/
```

## Quick Start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run Entry Point

```bash
python -m explainable_fl.main --config configs/config.yaml --mode ingest
python -m explainable_fl.main --config configs/config.yaml --mode train
```

## IEEE-CIS Ingestion

- Raw source folder is configured under the ingestion dataset mapping.
- The default dataset key is `ieee_cis_fraud`, backed by `data/raw/IEEE-CIS-FRAUD`.
- Ingestion merges `train_transaction` with `train_identity`, and `test_transaction` with `test_identity` using `TransactionID`.
- Merged parquet files are saved under `data/interim/ieee_cis_fraud/`.
- A YAML ingestion report is saved under `reports/ingestion/ieee_cis_fraud/`.

## Notes

- This scaffold intentionally contains no business logic yet.
- Implement stage-specific logic inside the corresponding modules under `src/explainable_fl/`.
