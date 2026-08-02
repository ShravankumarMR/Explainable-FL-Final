# Project Memory

## Confirmed Learnings

- Main runner now supports mode eda in addition to ingest, train, infer, and profile.
- EDA pipeline is implemented in src/explainable_fl/pipelines/eda_pipeline.py.
- EDA visualization logic is centralized in src/explainable_fl/eda/analyzer.py.
- EDA output is dataset-aware and currently writes IEEE-CIS artifacts to reports/eda/IEEE-CIS-FRAUD/.
- EDA split outputs include figures plus eda_artifacts.yaml metadata.
- EDA reads merged parquet outputs created by ingestion, so ingestion remains the single source of merged train/test data.
- Profiling and EDA can be independently toggled via config sections and pipelines toggles.

## Figure Set Produced By EDA

- fraud_distribution.png
- numerical_features.png
- categorical_features.png
- transaction_amount.png
- product_categories.png
- card_features.png
- email_domains.png
- identity_features.png
- missing_values.png
- correlations.png
- transactiondt_analysis.png
- fraud_by_hour.png
- fraud_by_day.png
- pca.png
- umap.png
- extra_trees_feature_importance.png

## Testing Learnings

- tests/test_eda_pipeline.py validates that all expected figures are exported for both train and test splits.
- tests/test_scaffold.py now validates EDA config and pipeline toggle fields.

## Operational Notes

- If terminal output capture is blank, use static diagnostics and file-level checks to validate changes.
- For PowerShell, execute quoted interpreters using call operator:
  & "path/to/python.exe" -m pytest

## How To Add A New Dataset

1. Add a new key under ingestion.datasets in configs/config.yaml.
2. Set base_dir and all source file names for that dataset.
3. Set merge_key and required schema columns for validation.
4. Keep output_train_parquet, output_test_parquet, and report_file configured.
5. Switch ingestion.active_dataset to the new dataset key.
6. Run ingest mode to produce interim parquet and ingestion report.
7. Run profile mode to generate tabular profiling artifacts for train/test.
8. Run eda mode to generate figures under reports/eda/<dataset_base_dir>/.
9. If dataset field names differ from IEEE-CIS conventions, update config.eda.target_column and adjust analyzer selectors where needed.
10. Add a dedicated pytest covering ingestion and EDA artifact expectations for the new dataset.
