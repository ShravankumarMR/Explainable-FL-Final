"""Dataset profiling utilities for automated EDA reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class ProfilingArtifacts:
    split_name: str
    output_dir: str
    csv_files: dict[str, str]
    html_file: str


class DatasetProfiler:
    """Computes per-split tabular profiling artifacts and writes reports."""

    def run(
        self,
        dataframe: pd.DataFrame,
        split_name: str,
        output_dir: Path,
        high_cardinality_ratio: float,
    ) -> ProfilingArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

        dtypes_df = self._build_dtypes(dataframe)
        missing_df = self._build_missing_values(dataframe)
        unique_df = self._build_unique_counts(dataframe)
        summary_df = self._build_summary_statistics(dataframe)
        duplicate_df = self._build_duplicate_analysis(dataframe)
        memory_df = self._build_memory_usage(dataframe)
        groups_df = self._build_feature_groups(dataframe, high_cardinality_ratio)

        csv_files = {
            "data_types": self._write_csv(dtypes_df, output_dir / "data_types.csv"),
            "missing_values": self._write_csv(missing_df, output_dir / "missing_values.csv"),
            "unique_counts": self._write_csv(unique_df, output_dir / "unique_counts.csv"),
            "summary_statistics": self._write_csv(
                summary_df, output_dir / "summary_statistics.csv"
            ),
            "duplicate_analysis": self._write_csv(
                duplicate_df, output_dir / "duplicate_analysis.csv"
            ),
            "memory_usage": self._write_csv(memory_df, output_dir / "memory_usage.csv"),
            "feature_groups": self._write_csv(groups_df, output_dir / "feature_groups.csv"),
        }

        html_path = output_dir / "profile_report.html"
        html_path.write_text(
            self._build_html_report(
                split_name=split_name,
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                dtypes_df=dtypes_df,
                missing_df=missing_df,
                unique_df=unique_df,
                summary_df=summary_df,
                duplicate_df=duplicate_df,
                memory_df=memory_df,
                groups_df=groups_df,
            ),
            encoding="utf-8",
        )

        return ProfilingArtifacts(
            split_name=split_name,
            output_dir=str(output_dir),
            csv_files=csv_files,
            html_file=str(html_path),
        )

    @staticmethod
    def _build_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
        dtypes_df = dataframe.dtypes.astype(str).rename("data_type").to_frame().reset_index()
        dtypes_df = dtypes_df.rename(columns={"index": "feature"})
        return dtypes_df.sort_values("feature").reset_index(drop=True)

    @staticmethod
    def _build_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        total_rows = max(len(dataframe), 1)
        missing_count = dataframe.isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        missing_df = pd.DataFrame(
            {
                "feature": missing_count.index,
                "missing_count": missing_count.values.astype(int),
                "missing_pct": missing_pct.values.round(4),
            }
        )
        return missing_df.sort_values(["missing_count", "feature"], ascending=[False, True]).reset_index(
            drop=True
        )

    @staticmethod
    def _build_unique_counts(dataframe: pd.DataFrame) -> pd.DataFrame:
        total_rows = max(len(dataframe), 1)
        unique_count = dataframe.nunique(dropna=True)
        unique_pct = (unique_count / total_rows) * 100
        unique_df = pd.DataFrame(
            {
                "feature": unique_count.index,
                "unique_count": unique_count.values.astype(int),
                "unique_pct": unique_pct.values.round(4),
            }
        )
        return unique_df.sort_values(["unique_count", "feature"], ascending=[False, True]).reset_index(
            drop=True
        )

    @staticmethod
    def _build_summary_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
        summary_df = dataframe.describe(include="all", percentiles=[0.25, 0.5, 0.75]).transpose()
        summary_df = summary_df.reset_index().rename(columns={"index": "feature"})
        return summary_df

    @staticmethod
    def _build_duplicate_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
        total_rows = len(dataframe)
        duplicate_rows = int(dataframe.duplicated().sum())
        duplicate_pct = round((duplicate_rows / total_rows) * 100, 4) if total_rows else 0.0
        return pd.DataFrame(
            [
                {
                    "total_rows": total_rows,
                    "duplicate_rows": duplicate_rows,
                    "duplicate_pct": duplicate_pct,
                    "unique_rows": int(total_rows - duplicate_rows),
                }
            ]
        )

    @staticmethod
    def _build_memory_usage(dataframe: pd.DataFrame) -> pd.DataFrame:
        usage_bytes = dataframe.memory_usage(deep=True)
        memory_df = usage_bytes.rename("memory_bytes").to_frame().reset_index()
        memory_df = memory_df.rename(columns={"index": "feature"})
        memory_df["memory_mb"] = (memory_df["memory_bytes"] / (1024 * 1024)).round(6)
        return memory_df

    @staticmethod
    def _build_feature_groups(
        dataframe: pd.DataFrame,
        high_cardinality_ratio: float,
    ) -> pd.DataFrame:
        if dataframe.empty:
            return pd.DataFrame(columns=["feature", "group"])

        total_rows = len(dataframe)
        unique_ratio = dataframe.nunique(dropna=True) / max(total_rows, 1)

        groups: list[dict[str, str]] = []
        for feature in dataframe.columns:
            dtype = dataframe[feature].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                group = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                group = "datetime"
            elif pd.api.types.is_bool_dtype(dtype):
                group = "boolean"
            else:
                group = "categorical"

            if unique_ratio[feature] >= high_cardinality_ratio:
                group = f"{group}_high_cardinality"

            groups.append({"feature": str(feature), "group": group})

        groups_df = pd.DataFrame(groups)
        return groups_df.sort_values(["group", "feature"]).reset_index(drop=True)

    @staticmethod
    def _write_csv(dataframe: pd.DataFrame, path: Path) -> str:
        dataframe.to_csv(path, index=False)
        return str(path)

    @staticmethod
    def _build_html_report(
        split_name: str,
        row_count: int,
        column_count: int,
        dtypes_df: pd.DataFrame,
        missing_df: pd.DataFrame,
        unique_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        duplicate_df: pd.DataFrame,
        memory_df: pd.DataFrame,
        groups_df: pd.DataFrame,
    ) -> str:
        sections = [
            ("Data Types", dtypes_df),
            ("Missing Values", missing_df),
            ("Unique Counts", unique_df),
            ("Summary Statistics", summary_df),
            ("Duplicate Analysis", duplicate_df),
            ("Memory Usage", memory_df),
            ("Feature Groups", groups_df),
        ]

        rendered_sections = "\n".join(
            (
                f"<h2>{title}</h2>"
                f"{table.to_html(index=False, border=0, classes='profile-table')}"
            )
            for title, table in sections
        )

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Profile Report - {split_name}</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #4b5563; margin-bottom: 20px; }}
    h2 {{ margin-top: 24px; }}
    .profile-table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    .profile-table th, .profile-table td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    .profile-table th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Automated Profile Report: {split_name}</h1>
  <p class=\"meta\">Rows: {row_count} | Columns: {column_count}</p>
  {rendered_sections}
</body>
</html>
"""
