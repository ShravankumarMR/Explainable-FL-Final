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
python -m explainable_fl.main --config configs/config.yaml --mode train
```

## Notes

- This scaffold intentionally contains no business logic yet.
- Implement stage-specific logic inside the corresponding modules under `src/explainable_fl/`.
