# Working Instructions

## Scope

This repository supports configurable ingestion, profiling, and EDA for multiple datasets. Keep solutions reusable across dataset keys under ingestion.datasets.

## Execution Modes

Use the single entrypoint with mode switches:

- ingest
- train
- infer
- profile
- eda

Command format:

python -m explainable_fl.main --config configs/config.yaml --mode <mode>

## Dataset-Aware Output Convention

- Ingestion output: data/interim/<dataset_key>/
- Ingestion report: reports/ingestion/<dataset_key>/
- Profiling output: reports/profiling/<dataset_key>/<split>/
- EDA figures output: reports/eda/<dataset_base_dir>/<split>/figures/

For IEEE-CIS-FRAUD, EDA goes under reports/eda/IEEE-CIS-FRAUD/.

## EDA Coverage Contract

The EDA pipeline should generate and save at least these visual artifacts per split:

- fraud_distribution
- numerical_features
- categorical_features
- transaction_amount
- product_categories
- card_features
- email_domains
- identity_features
- missing_values
- correlations
- transactiondt_analysis
- fraud_by_hour
- fraud_by_day
- pca
- umap
- extra_trees_feature_importance

## Config Contract

When adding new capabilities:

1. Add typed config dataclass in src/explainable_fl/config/loader.py.
2. Parse defaults safely in load_app_config.
3. Add corresponding section to configs/config.yaml.
4. Add or extend a pipeline toggle under pipelines.
5. Add tests for new runtime behavior and artifact outputs.

## Dependencies

EDA mode requires:

- matplotlib
- seaborn
- scikit-learn
- umap-learn

Keep pyproject.toml and requirements.txt aligned.
