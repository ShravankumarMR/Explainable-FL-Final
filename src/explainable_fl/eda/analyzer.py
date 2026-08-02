"""Reusable EDA analyzer that exports dataset-specific visual artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from explainable_fl.config.loader import EDAConfig

LOGGER = logging.getLogger("explainable_fl.eda.analyzer")


@dataclass(slots=True)
class EDAArtifacts:
    split_name: str
    output_dir: str
    figure_files: dict[str, str]
    summary_file: str


class DatasetEDAAnalyzer:
    """Generates reusable EDA plots and saves them to disk."""

    def run(
        self,
        dataframe: pd.DataFrame,
        split_name: str,
        output_dir: Path,
        config: EDAConfig,
    ) -> EDAArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        plot_df = self._sample_dataframe(
            dataframe=dataframe,
            sample_size=config.sample_size,
            random_state=config.random_state,
        )

        sns.set_theme(style="whitegrid")

        figure_paths: dict[str, str] = {}
        figure_paths["fraud_distribution"] = self._plot_fraud_distribution(
            plot_df, figures_dir / "fraud_distribution.png", config.target_column
        )
        figure_paths["numerical_features"] = self._plot_numerical_features(
            plot_df, figures_dir / "numerical_features.png", config
        )
        figure_paths["categorical_features"] = self._plot_categorical_features(
            plot_df, figures_dir / "categorical_features.png", config
        )
        figure_paths["transaction_amount"] = self._plot_transaction_amount(
            plot_df, figures_dir / "transaction_amount.png"
        )
        figure_paths["product_categories"] = self._plot_prefixed_features(
            plot_df,
            figures_dir / "product_categories.png",
            ["ProductCD"],
            "Product Categories",
            config.max_category_levels,
        )
        figure_paths["card_features"] = self._plot_prefixed_features(
            plot_df,
            figures_dir / "card_features.png",
            ["card"],
            "Card Features",
            config.max_category_levels,
        )
        figure_paths["email_domains"] = self._plot_prefixed_features(
            plot_df,
            figures_dir / "email_domains.png",
            ["P_emaildomain", "R_emaildomain", "emaildomain"],
            "Email Domains",
            config.max_category_levels,
        )
        figure_paths["identity_features"] = self._plot_prefixed_features(
            plot_df,
            figures_dir / "identity_features.png",
            ["id_", "DeviceType", "DeviceInfo"],
            "Identity Features",
            config.max_category_levels,
        )
        figure_paths["missing_values"] = self._plot_missing_values(
            plot_df, figures_dir / "missing_values.png", config.max_numerical_features
        )
        figure_paths["correlations"] = self._plot_correlations(
            plot_df, figures_dir / "correlations.png", config.correlation_top_n
        )
        figure_paths["transactiondt_analysis"] = self._plot_transaction_dt_distribution(
            plot_df, figures_dir / "transactiondt_analysis.png"
        )
        figure_paths["fraud_by_hour"] = self._plot_fraud_by_time(
            plot_df,
            figures_dir / "fraud_by_hour.png",
            config.target_column,
            granularity="hour",
        )
        figure_paths["fraud_by_day"] = self._plot_fraud_by_time(
            plot_df,
            figures_dir / "fraud_by_day.png",
            config.target_column,
            granularity="day",
        )
        figure_paths["pca"] = self._plot_pca(
            plot_df, figures_dir / "pca.png", config.target_column, config
        )
        figure_paths["umap"] = self._plot_umap(
            plot_df, figures_dir / "umap.png", config.target_column, config
        )
        figure_paths["extra_trees_feature_importance"] = self._plot_feature_importance(
            plot_df,
            figures_dir / "extra_trees_feature_importance.png",
            config.target_column,
            config.feature_importance_top_n,
            config.random_state,
        )

        summary_path = output_dir / "eda_artifacts.yaml"
        summary_payload = {
            "split": split_name,
            "rows_full": int(len(dataframe)),
            "rows_sampled": int(len(plot_df)),
            "figures": figure_paths,
        }
        summary_path.write_text(yaml.safe_dump(summary_payload, sort_keys=False), encoding="utf-8")

        return EDAArtifacts(
            split_name=split_name,
            output_dir=str(output_dir),
            figure_files=figure_paths,
            summary_file=str(summary_path),
        )

    @staticmethod
    def _sample_dataframe(dataframe: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
        if sample_size <= 0 or len(dataframe) <= sample_size:
            return dataframe.copy()
        return dataframe.sample(n=sample_size, random_state=random_state).copy()

    @staticmethod
    def _save(path: Path) -> str:
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return str(path)

    def _plot_note(self, path: Path, title: str, message: str) -> str:
        plt.figure(figsize=(8, 4))
        plt.title(title)
        plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        plt.axis("off")
        return self._save(path)

    def _plot_fraud_distribution(self, df: pd.DataFrame, path: Path, target_col: str) -> str:
        if target_col not in df.columns:
            return self._plot_note(path, "Fraud Distribution", f"Column '{target_col}' not found.")

        plt.figure(figsize=(7, 5))
        counts = df[target_col].value_counts(dropna=False)
        sns.barplot(x=counts.index.astype(str), y=counts.values, color="#2F6DB3")
        plt.title("Fraud Distribution")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        return self._save(path)

    def _plot_numerical_features(self, df: pd.DataFrame, path: Path, config: EDAConfig) -> str:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return self._plot_note(path, "Numerical Features", "No numeric columns available.")

        cols = numeric_cols[: config.max_numerical_features]
        plot_df = df[cols].copy().replace([np.inf, -np.inf], np.nan)

        n = len(cols)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
        axes_flat = np.array(axes).reshape(-1)

        for idx, col in enumerate(cols):
            sns.histplot(plot_df[col].dropna(), bins=30, ax=axes_flat[idx], color="#3C8D40")
            axes_flat[idx].set_title(col)
            axes_flat[idx].set_xlabel("")
            axes_flat[idx].set_ylabel("Count")

        for idx in range(len(cols), len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle("Numerical Features", y=1.02)
        return self._save(path)

    def _plot_categorical_features(self, df: pd.DataFrame, path: Path, config: EDAConfig) -> str:
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if not cat_cols:
            return self._plot_note(path, "Categorical Features", "No categorical columns available.")

        cols = cat_cols[: config.max_categorical_features]
        n = len(cols)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
        axes_flat = np.array(axes).reshape(-1)

        for idx, col in enumerate(cols):
            top_vals = (
                df[col]
                .astype(str)
                .fillna("missing")
                .value_counts()
                .head(config.max_category_levels)
                .sort_values(ascending=True)
            )
            sns.barplot(x=top_vals.values, y=top_vals.index, ax=axes_flat[idx], color="#C97B29")
            axes_flat[idx].set_title(col)
            axes_flat[idx].set_xlabel("Count")
            axes_flat[idx].set_ylabel("")

        for idx in range(len(cols), len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle("Categorical Features", y=1.02)
        return self._save(path)

    def _plot_transaction_amount(self, df: pd.DataFrame, path: Path) -> str:
        if "TransactionAmt" not in df.columns:
            return self._plot_note(path, "Transaction Amount", "Column 'TransactionAmt' not found.")

        plt.figure(figsize=(8, 5))
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").dropna()
        sns.histplot(amt, bins=50, kde=True, color="#7E5BBE")
        plt.title("Transaction Amount Distribution")
        plt.xlabel("TransactionAmt")
        plt.ylabel("Count")
        return self._save(path)

    def _plot_prefixed_features(
        self,
        df: pd.DataFrame,
        path: Path,
        prefixes: list[str],
        title: str,
        max_levels: int,
    ) -> str:
        cols = [
            col
            for col in df.columns
            if any(col == prefix or col.startswith(prefix) or prefix in col for prefix in prefixes)
        ]
        if not cols:
            return self._plot_note(path, title, "No matching columns found.")

        max_cols = min(6, len(cols))
        cols = cols[:max_cols]

        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3.4 * n))
        axes_flat = np.array(axes).reshape(-1)

        for idx, col in enumerate(cols):
            vc = (
                df[col]
                .astype(str)
                .fillna("missing")
                .value_counts()
                .head(max_levels)
                .sort_values(ascending=True)
            )
            sns.barplot(x=vc.values, y=vc.index, ax=axes_flat[idx], color="#4F7F8E")
            axes_flat[idx].set_title(col)
            axes_flat[idx].set_xlabel("Count")
            axes_flat[idx].set_ylabel("")

        fig.suptitle(title, y=1.0)
        return self._save(path)

    def _plot_missing_values(self, df: pd.DataFrame, path: Path, top_n: int) -> str:
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0].head(top_n)

        if missing_pct.empty:
            return self._plot_note(path, "Missing Values", "No missing values detected.")

        plt.figure(figsize=(10, max(4, 0.35 * len(missing_pct))))
        sns.barplot(x=missing_pct.values, y=missing_pct.index, color="#B64949")
        plt.title("Top Missing Values (%)")
        plt.xlabel("Missing %")
        plt.ylabel("Feature")
        return self._save(path)

    def _plot_correlations(self, df: pd.DataFrame, path: Path, top_n: int) -> str:
        num_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        if num_df.shape[1] < 2:
            return self._plot_note(path, "Correlations", "At least two numeric columns are required.")

        variances = num_df.var(numeric_only=True).sort_values(ascending=False)
        corr_cols = variances.head(top_n).index.tolist()
        corr_matrix = num_df[corr_cols].corr().fillna(0)

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
        plt.title("Numeric Correlation Heatmap")
        return self._save(path)

    def _plot_transaction_dt_distribution(self, df: pd.DataFrame, path: Path) -> str:
        if "TransactionDT" not in df.columns:
            return self._plot_note(path, "TransactionDT Analysis", "Column 'TransactionDT' not found.")

        transaction_dt = pd.to_numeric(df["TransactionDT"], errors="coerce").dropna()
        if transaction_dt.empty:
            return self._plot_note(path, "TransactionDT Analysis", "TransactionDT has no numeric values.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.histplot(transaction_dt, bins=50, ax=axes[0], color="#2E9B8A")
        axes[0].set_title("TransactionDT Distribution")
        axes[0].set_xlabel("TransactionDT")
        axes[0].set_ylabel("Count")

        sns.boxplot(x=transaction_dt, ax=axes[1], color="#2E9B8A")
        axes[1].set_title("TransactionDT Boxplot")
        axes[1].set_xlabel("TransactionDT")

        return self._save(path)

    def _plot_fraud_by_time(
        self,
        df: pd.DataFrame,
        path: Path,
        target_col: str,
        granularity: str,
    ) -> str:
        if target_col not in df.columns or "TransactionDT" not in df.columns:
            return self._plot_note(path, f"Fraud by {granularity.title()}", "Required columns are missing.")

        temp = df[[target_col, "TransactionDT"]].copy()
        temp["TransactionDT"] = pd.to_numeric(temp["TransactionDT"], errors="coerce")
        temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
        temp = temp.dropna(subset=["TransactionDT", target_col])
        if temp.empty:
            return self._plot_note(path, f"Fraud by {granularity.title()}", "No valid rows for plot.")

        if granularity == "hour":
            temp["bucket"] = ((temp["TransactionDT"] // 3600) % 24).astype(int)
            title = "Fraud Rate by Hour"
            x_label = "Hour"
        else:
            temp["bucket"] = (temp["TransactionDT"] // (24 * 3600)).astype(int)
            title = "Fraud Rate by Day"
            x_label = "Day"

        fraud_rate = temp.groupby("bucket", as_index=False)[target_col].mean().sort_values("bucket")

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=fraud_rate, x="bucket", y=target_col, marker="o", color="#C94F5D")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Fraud Rate")
        return self._save(path)

    def _build_embedding_matrix(
        self,
        df: pd.DataFrame,
        target_col: str,
        random_state: int,
    ) -> tuple[pd.DataFrame | None, pd.Series | None, str | None]:
        if target_col not in df.columns:
            return None, None, f"Column '{target_col}' not found."

        y = pd.to_numeric(df[target_col], errors="coerce")
        work_df = df.drop(columns=[target_col], errors="ignore").copy()

        selected_numeric = work_df.select_dtypes(include=[np.number]).columns.tolist()[:40]
        selected_categorical = (
            work_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()[:10]
        )
        selected_cols = selected_numeric + selected_categorical
        if not selected_cols:
            return None, None, "No usable features available for embedding."

        x = work_df[selected_cols].copy()
        for col in selected_numeric:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].fillna(x[col].median())

        for col in selected_categorical:
            x[col] = x[col].astype(str).fillna("missing")

        x = pd.get_dummies(x, dummy_na=True)
        if x.empty:
            return None, None, "Feature matrix is empty after encoding."

        # Keep embedding dimensionality bounded for stable runtime on large datasets.
        if x.shape[1] > 250:
            x = x.iloc[:, :250]

        valid_mask = y.notna()
        x = x.loc[valid_mask]
        y = y.loc[valid_mask]
        if x.empty:
            return None, None, "No valid target rows available for embedding."

        if y.nunique(dropna=True) < 2:
            return None, None, "Target column has fewer than 2 classes."

        scaled = StandardScaler().fit_transform(x)
        scaled_df = pd.DataFrame(scaled, columns=x.columns, index=x.index)
        y = y.astype(int)
        np.random.seed(random_state)
        return scaled_df, y, None

    def _plot_pca(self, df: pd.DataFrame, path: Path, target_col: str, config: EDAConfig) -> str:
        x, y, error = self._build_embedding_matrix(df, target_col, config.random_state)
        if error is not None or x is None or y is None:
            return self._plot_note(path, "PCA Projection", error or "Unable to build embedding.")

        pca = PCA(n_components=min(config.pca_components, x.shape[1]), random_state=config.random_state)
        emb = pca.fit_transform(x)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=y.astype(str), s=12, alpha=0.6, linewidth=0)
        plt.title("PCA Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(title=target_col)
        return self._save(path)

    def _plot_umap(self, df: pd.DataFrame, path: Path, target_col: str, config: EDAConfig) -> str:
        x, y, error = self._build_embedding_matrix(df, target_col, config.random_state)
        if error is not None or x is None or y is None:
            return self._plot_note(path, "UMAP Projection", error or "Unable to build embedding.")

        try:
            import umap
        except ImportError:
            LOGGER.warning("umap-learn is not installed; creating note plot instead.")
            return self._plot_note(path, "UMAP Projection", "umap-learn is not installed.")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            random_state=config.random_state,
        )
        emb = reducer.fit_transform(x)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=y.astype(str), s=12, alpha=0.6, linewidth=0)
        plt.title("UMAP Projection")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(title=target_col)
        return self._save(path)

    def _plot_feature_importance(
        self,
        df: pd.DataFrame,
        path: Path,
        target_col: str,
        top_n: int,
        random_state: int,
    ) -> str:
        if target_col not in df.columns:
            return self._plot_note(path, "ExtraTrees Feature Importance", f"Column '{target_col}' not found.")

        y = pd.to_numeric(df[target_col], errors="coerce")
        work_df = df.drop(columns=[target_col], errors="ignore").copy()

        numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()[:60]
        cat_cols = work_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()[:12]
        use_cols = numeric_cols + cat_cols
        if not use_cols:
            return self._plot_note(path, "ExtraTrees Feature Importance", "No usable feature columns found.")

        x = work_df[use_cols].copy()
        for col in numeric_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].fillna(x[col].median())
        for col in cat_cols:
            x[col] = x[col].astype(str).fillna("missing")

        x = pd.get_dummies(x, dummy_na=True)
        valid_mask = y.notna()
        x = x.loc[valid_mask]
        y = y.loc[valid_mask].astype(int)

        if x.empty or y.nunique(dropna=True) < 2:
            return self._plot_note(
                path,
                "ExtraTrees Feature Importance",
                "Need non-empty features and at least 2 target classes.",
            )

        if x.shape[1] > 300:
            x = x.iloc[:, :300]

        model = ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        model.fit(x, y)

        importances = pd.Series(model.feature_importances_, index=x.columns)
        top_features = importances.nlargest(top_n).sort_values(ascending=True)

        plt.figure(figsize=(10, max(4, 0.3 * len(top_features))))
        sns.barplot(x=top_features.values, y=top_features.index, color="#6A8D3D")
        plt.title("ExtraTrees Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        return self._save(path)
