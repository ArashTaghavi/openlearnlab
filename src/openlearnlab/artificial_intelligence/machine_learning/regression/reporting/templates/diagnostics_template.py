import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class DiagnosticsTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        residuals: np.ndarray,
        r2: float,
        rmse: float,
        mae: float,
        mse: float,
        insights: list[tuple[str, str]],
        n_samples: int,
        n_features: int,
        n_folds: int,
        critical_errors: pd.DataFrame,
        outlier_records: pd.DataFrame,
        high_error_records: pd.DataFrame,
        overlapping_records: pd.DataFrame,
        error_records: pd.DataFrame,
        target_error_analysis: pd.DataFrame,
        feature_error_analysis: pd.DataFrame,
        error_threshold: float,
        high_error_percentage: float,
        high_error_mae_share: float,
        high_error_mse_share: float,
        outlier_mae_share: float,
        outlier_mse_share: float,
    ):
        super().__init__(dataset_name=dataset_name, title="Regression Diagnostics")

        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)
        self.residuals = np.asarray(residuals, dtype=float)
        self.r2, self.rmse, self.mae, self.mse = r2, rmse, mae, mse
        self.insights = insights
        self.n_samples, self.n_features, self.n_folds = n_samples, n_features, n_folds
        self.critical_errors = critical_errors
        self.outlier_records = outlier_records
        self.high_error_records = high_error_records
        self.overlapping_records = overlapping_records
        self.error_records = error_records
        self.target_error_analysis = target_error_analysis
        self.feature_error_analysis = feature_error_analysis
        self.error_threshold = error_threshold
        self.high_error_percentage = high_error_percentage
        self.high_error_mae_share = high_error_mae_share
        self.high_error_mse_share = high_error_mse_share
        self.outlier_mae_share = outlier_mae_share
        self.outlier_mse_share = outlier_mse_share
        self.full_width = True

    @staticmethod
    def _figure_to_base64(fig) -> str:
        buffer = BytesIO()
        try:
            fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        finally:
            plt.close(fig)
            buffer.close()

    @staticmethod
    def _fmt(value) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.4f}"
        return escape(str(value))

    @staticmethod
    def _table(
        df: pd.DataFrame,
        *,
        index: bool = True,
        empty_message: str = "No records detected.",
    ) -> str:
        if df.empty:
            return f'<div class="card"><p>{escape(empty_message)}</p></div>'
        view = df.copy()
        # Keep original row indices so users can locate the observations.
        if index:
            view.index.name = "Row Index"
        for column in view.select_dtypes(include=["bool"]).columns:
            view[column] = view[column].map({True: "Yes", False: "No"})
        html = view.to_html(
            classes="data-table",
            index=index,
            border=0,
            escape=True,
            na_rep="—",
            float_format=lambda value: f"{value:,.4f}",
        )
        return f'<div style="overflow-x:auto;">{html}</div>'

    def _records_table(self, df: pd.DataFrame, *, empty_message: str) -> str:
        """Collapse input features into an expandable cell for each record."""
        feature_columns = list(self.error_records.columns)
        if "Actual" not in feature_columns:
            return self._table(df, empty_message=empty_message)
        feature_columns = [
            column
            for column in feature_columns[: feature_columns.index("Actual")]
            if column in df.columns
        ]
        if not feature_columns:
            return self._table(df, empty_message=empty_message)

        view = df.copy()
        replacements = {}
        tokens = []
        for position, (_, record) in enumerate(view.iterrows()):
            token = f"FEATURE_DETAILS_TOKEN_{position}_END"
            items = "".join(
                f"<div><strong>{escape(str(column))}</strong>: "
                f"{self._fmt(record[column])}</div>"
                for column in feature_columns
            )
            replacements[token] = (
                '<details class="record-features">'
                "<summary>View Features</summary>"
                f'<div class="record-features-content">{items}</div>'
                "</details>"
            )
            tokens.append(token)

        first_position = view.columns.get_loc(feature_columns[0])
        view = view.drop(columns=feature_columns)
        view.insert(first_position, "Features", tokens)
        html = self._table(view, empty_message=empty_message)
        for token, markup in replacements.items():
            html = html.replace(token, markup)
        return html

    @staticmethod
    def _section(title: str, description: str, body: str) -> str:
        return (
            '<section class="table-section">'
            f"<h2>{escape(title)}</h2><p>{escape(description)}</p>{body}"
            "</section>"
        )

    def _actual_vs_predicted_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(self.y_true, self.y_pred, alpha=0.65, s=25)
        minimum = float(min(np.min(self.y_true), np.min(self.y_pred)))
        maximum = float(max(np.max(self.y_true), np.max(self.y_pred)))
        ax.plot([minimum, maximum], [minimum, maximum], "r--", linewidth=1.5)
        ax.set(
            xlabel="Actual Values",
            ylabel="Predicted Values",
            title="Actual vs Predicted",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _residuals_vs_predicted_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(self.y_pred, self.residuals, alpha=0.65, s=25)
        ax.axhline(0, linestyle="--", color="red", linewidth=1.5)
        ax.set(
            xlabel="Predicted Values",
            ylabel="Residuals",
            title="Residuals vs Predicted",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _residual_distribution_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(self.residuals, bins="auto", edgecolor="white", alpha=0.85)
        ax.axvline(0, linestyle="--", color="red", linewidth=1.5)
        ax.set(xlabel="Residual", ylabel="Frequency", title="Residual Distribution")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _error_by_target_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(self.y_true, np.abs(self.residuals), alpha=0.6, s=25)
        ax.axhline(
            self.error_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"High-error threshold: {self.error_threshold:.3g}",
        )
        ax.set(
            xlabel="Actual Target",
            ylabel="Absolute Error",
            title="Absolute Error vs Actual Target",
        )
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _summary_section(self) -> str:
        metrics = [
            ("R² Score", f"{self.r2:.4f}"),
            ("MAE", f"{self.mae:.4f}"),
            ("MSE", f"{self.mse:.4f}"),
            ("RMSE", f"{self.rmse:.4f}"),
            ("Observations", f"{self.n_samples:,}"),
            ("Target Outliers", f"{len(self.outlier_records):,}"),
            ("High-Error Records", f"{len(self.high_error_records):,}"),
            ("Overlapping Records", f"{len(self.overlapping_records):,}"),
            ("Evaluation", f"{self.n_folds}-Fold CV"),
        ]
        cards = "".join(
            f'<div class="card"><div class="card-title">{escape(label)}</div><pre>{escape(value)}</pre></div>'
            for label, value in metrics
        )
        return self._section(
            "Model Performance Summary",
            f"Out-of-fold evaluation on {self.n_samples:,} observations and {self.n_features:,} features. "
            "These are diagnostic estimates, not a separate held-out test score.",
            f'<div class="grid">{cards}</div>',
        )

    def _outlier_section(self) -> str:
        return self._section(
            "Target Outlier Records",
            "Records whose actual target values fall outside the 1.5 × IQR bounds. "
            "These values are not necessarily errors and are shown for review only.",
            self._records_table(
                self.outlier_records.drop(
                    columns=[
                        "Is Outlier",
                        "Is Overlapping",
                        "Absolute Error",
                    ],
                    errors="ignore",
                ),
                empty_message="No target outliers detected.",
            ),
        )

    def _high_error_section(self) -> str:
        return self._section(
            "High-Error Records",
            f"Rows with absolute out-of-fold prediction error greater than "
            f"MAE × (1 + {self.high_error_percentage:g}/100) = {self.error_threshold:.4f}. "
            "Sorted by squared error; cumulative SSE share is measured against the entire dataset.",
            self._records_table(
                self.high_error_records,
                empty_message="No records exceed the high-error threshold.",
            ),
        )

    def _overlap_section(self) -> str:
        return self._section(
            "Overlapping Records",
            "Rows flagged both as target outliers and as high-error predictions. "
            "These are useful candidates for manual investigation, not automatic deletion.",
            self._records_table(
                self.overlapping_records.drop(
                    columns=[
                        "Is Outlier",
                        "Is High Error",
                        "Is Overlapping",
                        "Absolute Error",
                    ],
                    errors="ignore",
                ),
                empty_message="No overlap between outliers and high-error records.",
            ),
        )

    def _error_contribution_section(self) -> str:
        total = self.n_samples
        rows = [
            (
                "Target Outliers",
                len(self.outlier_records),
                self.outlier_mae_share,
                self.outlier_mse_share,
            ),
            (
                "High-Error Records",
                len(self.high_error_records),
                self.high_error_mae_share,
                self.high_error_mse_share,
            ),
        ]
        table_df = pd.DataFrame(
            [
                {
                    "Group": name,
                    "Records": count,
                    "Dataset Share (%)": 100 * count / total if total else 0.0,
                    "Absolute Error Share (%)": mae_share,
                    "Squared Error Share (%)": mse_share,
                }
                for name, count, mae_share, mse_share in rows
            ]
        )
        return self._section(
            "Error Contribution Analysis",
            "Each percentage is the group's share of the total absolute or squared error. "
            "Groups may overlap, so their shares must not be added together.",
            self._table(table_df, index=False),
        )

    def _error_patterns_section(self) -> str:
        target_df = self.target_error_analysis.copy()
        if "Target Range" in target_df.columns:
            target_df["Target Range"] = target_df["Target Range"].astype(str)
        target_section = self._section(
            "Error by Target Range",
            "Error metrics for quantile-based target ranges; differences may indicate difficult target regions.",
            self._table(
                target_df,
                index=False,
                empty_message="Not enough distinct target values for range analysis.",
            ),
        )
        feature_df = self.feature_error_analysis.copy()
        feature_section = self._section(
            "Error by Feature",
            "Correlation between numeric feature values and absolute prediction errors. "
            "Correlation is descriptive and does not establish causation.",
            self._table(
                feature_df,
                index=False,
                empty_message="No numeric features with sufficient variation for correlation analysis.",
            ),
        )
        return target_section + feature_section

    def _insights_section(self) -> str:
        cards = "".join(
            '<div class="card">'
            f'<div class="card-title">{escape(title)}</div>'
            f"<p>{escape(description)}</p>"
            "</div>"
            for title, description in self.insights
        )
        return self._section(
            "Diagnostic Insights",
            "Signals worth investigating, not proof of faulty data.",
            f'<div class="grid">{cards}</div>',
        )

    def _plots_section(self) -> str:
        plots = [
            ("Actual vs Predicted", self._actual_vs_predicted_plot()),
            ("Residuals vs Predicted", self._residuals_vs_predicted_plot()),
            ("Residual Distribution", self._residual_distribution_plot()),
            ("Absolute Error vs Actual Target", self._error_by_target_plot()),
        ]
        cards = "".join(
            '<div class="card">'
            f'<div class="card-title">{escape(title)}</div>'
            f'<img src="data:image/png;base64,{encoded}" alt="{escape(title)}" style="width:100%;height:auto;">'
            "</div>"
            for title, encoded in plots
        )
        return self._section(
            "Prediction Diagnostics",
            "Out-of-fold prediction and residual visualizations.",
            f'<div class="grid">{cards}</div>',
        )

    def content(self) -> str:
        return f"{self._summary_section()}\n{self._plots_section()}\n{self._error_contribution_section()}\n{self._outlier_section()}\n{self._high_error_section()}\n{self._overlap_section()}\n{self._error_patterns_section()}\n{self._insights_section()}"
