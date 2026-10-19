import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)

from ....base.reporting.base_template import BaseTemplate


class DiagnosticsTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        metrics: dict[str, float],
        classes: np.ndarray,
        confusion: np.ndarray,
        class_report: pd.DataFrame,
        error_records: pd.DataFrame,
        class_error_analysis: pd.DataFrame,
        feature_error_analysis: pd.DataFrame,
        insights: list[tuple[str, str]],
        n_samples: int,
        n_features: int,
        n_folds: int,
        feature_names: list[str],
        y_true: np.ndarray,
        probabilities: np.ndarray | None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title=f"Classification Diagnostics -- {model_name}",
        )
        self.model_name = model_name
        self.metrics = metrics
        self.classes = classes
        self.confusion = confusion
        self.class_report = class_report
        self.error_records = error_records
        self.class_error_analysis = class_error_analysis
        self.feature_error_analysis = feature_error_analysis
        self.insights = insights
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_folds = n_folds
        self.feature_names = feature_names
        self.y_true = y_true
        self.probabilities = probabilities
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
    def _table(
        df: pd.DataFrame,
        *,
        index: bool = False,
        empty_message: str = "No records detected.",
    ) -> str:
        if df.empty:
            return f'<div class="card"><p>{escape(empty_message)}</p></div>'
        html = df.to_html(
            classes="data-table",
            index=index,
            border=0,
            escape=True,
            na_rep="—",
            float_format=lambda value: f"{value:,.4f}",
        )
        return f'<div style="overflow-x:auto;">{html}</div>'

    @staticmethod
    def _section(title: str, description: str, body: str) -> str:
        return (
            '<section class="table-section">'
            f"<h2>{escape(title)}</h2><p>{escape(description)}</p>{body}"
            "</section>"
        )

    def _records_table(self, df: pd.DataFrame) -> str:
        if df.empty:
            return self._table(df, empty_message="No misclassified records detected.")

        view = df.drop(columns=["Is Error"], errors="ignore").copy()
        view = view[
            [
                *[
                    column
                    for column in (
                        "Actual",
                        "Predicted",
                        "P(Actual)",
                        "P(Predicted)",
                        "Error Type",
                    )
                    if column in view.columns
                ],
                *[
                    column
                    for column in view.columns
                    if column
                    not in (
                        "Actual",
                        "Predicted",
                        "P(Actual)",
                        "P(Predicted)",
                        "Error Type",
                    )
                ],
            ]
        ]
        feature_columns = [col for col in self.feature_names if col in view.columns]
        if not feature_columns:
            return self._table(view, index=True)

        tokens = []
        replacements = {}
        for position, (_, record) in enumerate(view.iterrows()):
            token = f"FEATURE_DETAILS_TOKEN_{position}_END"
            details = "".join(
                f"<div><strong>{escape(str(feature))}</strong>: "
                f"{escape(str(record[feature]))}</div>"
                for feature in feature_columns
            )
            replacements[token] = (
                '<details class="record-features">'
                "<summary>View Features</summary>"
                f'<div class="record-features-content">{details}</div>'
                "</details>"
            )
            tokens.append(token)

        view = view.drop(columns=feature_columns)
        view["Features"] = tokens
        html = self._table(view, index=True)
        for token, markup in replacements.items():
            html = html.replace(token, markup)
        return html

    def _confusion_plot(self, normalize: bool = False) -> str:
        matrix = self.confusion.astype(float if normalize else int)
        if normalize:
            totals = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(
                matrix,
                totals,
                out=np.zeros_like(matrix),
                where=totals != 0,
            )

        size = max(5.5, min(12.0, 4.5 + len(self.classes) * 0.45))
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[str(label) for label in self.classes],
        )
        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=True,
            values_format=".2f" if normalize else "d",
            xticks_rotation=45,
        )
        ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
        fig.tight_layout()
        return self._figure_to_base64(fig)

    # -----------------------------------
    # Precision / Recall / F1 vs Threshold
    # -----------------------------------

    def _threshold_metrics_plot(self) -> str:
        actual = (self.y_true == self.classes[1]).astype(int)
        probabilities = self.probabilities

        precision_values, recall_values, thresholds = precision_recall_curve(
            actual,
            probabilities,
        )

        precision_values = precision_values[:-1]
        recall_values = recall_values[:-1]

        f1_values = np.divide(
            2 * precision_values * recall_values,
            precision_values + recall_values,
            out=np.zeros_like(precision_values),
            where=(precision_values + recall_values) != 0,
        )

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.plot(
            thresholds,
            precision_values,
            label="Precision",
            color="tab:blue",
            linewidth=2,
        )
        ax.plot(
            thresholds,
            recall_values,
            label="Recall",
            color="tab:orange",
            linewidth=2,
        )
        ax.plot(
            thresholds,
            f1_values,
            label="F1 Score",
            color="tab:green",
            linewidth=2,
        )

        ax.axvline(
            0.5,
            color="tab:red",
            linestyle="--",
            linewidth=1.8,
            label="Default Threshold (0.50)",
        )

        ax.set_title("Precision, Recall and F1 vs Decision Threshold")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Metric Score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        return self._figure_to_base64(fig)

    # -----------------------------------
    # Probability Distribution
    # -----------------------------------

    def _probability_distribution_plot(self) -> str:
        actual = self.y_true
        probabilities = self.probabilities

        negative = actual == self.classes[0]
        positive = actual == self.classes[1]

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.scatter(
            probabilities[negative],
            np.zeros(negative.sum()),
            alpha=0.5,
            s=30,
            color="tab:blue",
            label=f"Actual Class {self.classes[0]} (n={negative.sum()})",
        )

        ax.scatter(
            probabilities[positive],
            np.ones(positive.sum()),
            alpha=0.5,
            s=30,
            color="tab:orange",
            label=f"Actual Class {self.classes[1]} (n={positive.sum()})",
        )

        ax.axvline(
            0.5,
            color="tab:red",
            linestyle="--",
            linewidth=2,
            label="Default Threshold (0.50)",
        )

        ax.set_title("Predicted Probability Distribution")
        ax.set_xlabel(f"Predicted Probability of Class {self.classes[1]}")
        ax.set_ylabel("Actual Class")
        ax.set_xlim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([str(self.classes[0]), str(self.classes[1])])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        return self._figure_to_base64(fig)

    # -----------------------------------
    # ROC Curve
    # -----------------------------------

    def _roc_plot(self) -> str:
        actual = (self.y_true == self.classes[1]).astype(int)

        fpr, tpr, thresholds = roc_curve(
            actual,
            self.probabilities,
        )

        roc_auc = auc(fpr, tpr)

        predicted_default = (self.probabilities >= 0.5).astype(int)

        default_fp = np.count_nonzero((predicted_default == 1) & (actual == 0))
        default_tn = np.count_nonzero((predicted_default == 0) & (actual == 0))
        default_tp = np.count_nonzero((predicted_default == 1) & (actual == 1))
        default_fn = np.count_nonzero((predicted_default == 0) & (actual == 1))

        default_fpr = default_fp / (default_fp + default_tn)
        default_tpr = default_tp / (default_tp + default_fn)

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(
            fpr,
            tpr,
            color="tab:blue",
            linewidth=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            linestyle="--",
            label="Random Classifier (AUC = 0.50)",
        )

        ax.scatter(
            default_fpr,
            default_tpr,
            color="tab:red",
            s=90,
            zorder=5,
            label="Threshold = 0.50",
        )

        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

        fig.tight_layout()
        return self._figure_to_base64(fig)

    # -----------------------------------
    # Prediction Probability Scatter
    # -----------------------------------

    def _probability_scatter_plot(self) -> str:
        actual = (self.y_true == self.classes[1]).astype(int)
        probabilities = self.probabilities

        predicted = (probabilities >= 0.5).astype(int)
        indices = np.arange(len(probabilities))

        categories = [
            (
                "True Negative (TN)",
                (actual == 0) & (predicted == 0),
                "tab:blue",
                "o",
            ),
            (
                "True Positive (TP)",
                (actual == 1) & (predicted == 1),
                "tab:green",
                "o",
            ),
            (
                "False Positive (FP)",
                (actual == 0) & (predicted == 1),
                "tab:orange",
                "x",
            ),
            (
                "False Negative (FN)",
                (actual == 1) & (predicted == 0),
                "tab:red",
                "x",
            ),
        ]

        fig, ax = plt.subplots(figsize=(10, 5.5))

        for label, mask, color, marker in categories:
            ax.scatter(
                indices[mask],
                probabilities[mask],
                color=color,
                marker=marker,
                s=32,
                alpha=0.7,
                label=f"{label} (n={int(mask.sum())})",
            )

        ax.axhline(
            0.5,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Decision Threshold (0.50)",
        )

        ax.axhspan(
            0.5,
            1.0,
            color="tab:green",
            alpha=0.04,
        )
        ax.axhspan(
            0.0,
            0.5,
            color="tab:blue",
            alpha=0.04,
        )

        ax.set_title("Prediction Probabilities and Decision Boundary")
        ax.set_xlabel("Observation Position")
        ax.set_ylabel(f"Predicted Probability of Class {self.classes[1]}")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        fig.tight_layout()
        return self._figure_to_base64(fig)

    # -----------------------------------
    # Threshold Analysis Section
    # -----------------------------------

    def _threshold_section(self) -> str:
        if self.probabilities is None:
            return ""

        plots = [
            (
                "Precision / Recall / F1 vs Threshold",
                "Shows how precision, recall and F1 change as the decision "
                "threshold moves from 0 to 1. The dashed line marks 0.50.",
                self._threshold_metrics_plot(),
            ),
            (
                "Probability Distribution",
                "Each point represents an observation. The horizontal position "
                "shows its predicted positive-class probability, while the "
                "vertical position identifies its actual class. "
                "The dashed line marks the default decision threshold of 0.50.",
                self._probability_distribution_plot(),
            ),
            (
                "ROC Curve",
                "Shows the trade-off between true positive rate and false "
                "positive rate across thresholds. AUC summarizes discrimination.",
                self._roc_plot(),
            ),
            (
                "Prediction Probability Scatter",
                "Each point represents an observation. Points above the dashed "
                "threshold are classified as positive; points below are negative. "
                "Colors and markers identify TP, TN, FP and FN.",
                self._probability_scatter_plot(),
            ),
        ]

        cards = "".join(
            '<div class="card">'
            f'<div class="card-title">{escape(title)}</div>'
            f'<img src="data:image/png;base64,{encoded}" '
            f'alt="{escape(title)}" style="width:100%;height:auto;">'
            f"<p>{escape(description)}</p>"
            "</div>"
            for title, description, encoded in plots
        )

        return self._section(
            "Decision Threshold Analysis",
            f"Binary classification: negative class = '{self.classes[0]}', "
            f"positive class = '{self.classes[1]}'. "
            "All probabilities are generated from out-of-fold predictions. "
            "The displayed decision threshold is 0.50. "
            "Changing the threshold changes classification decisions, "
            "not the underlying predicted probabilities.",
            f'<div class="grid">{cards}</div>',
        )

    def _summary_section(self) -> str:
        items = [
            ("Model", str(self.model_name)),
            ("Accuracy", f"{self.metrics['Accuracy']:.4f}"),
            ("Precision (Weighted)", f"{self.metrics['Precision']:.4f}"),
            ("Recall (Weighted)", f"{self.metrics['Recall']:.4f}"),
            ("F1 (Weighted)", f"{self.metrics['F1']:.4f}"),
            ("Observations", f"{self.n_samples:,}"),
            ("Features", str(self.n_features)),
            ("Classes", str(len(self.classes))),
            ("Misclassified", f"{len(self.error_records):,}"),
            ("Evaluation", f"{self.n_folds}-Fold Stratified CV"),
        ]
        cards = "".join(
            '<div class="card">'
            f'<div class="card-title">{escape(label)}</div>'
            f"<pre>{escape(value)}</pre></div>"
            for label, value in items
        )
        return self._section(
            "Model Performance Summary",
            "Metrics are computed from out-of-fold predictions, "
            "not from a separate held-out test set.",
            f'<div class="grid">{cards}</div>',
        )

    def _plots_section(self) -> str:
        plots = [
            ("Confusion Matrix", self._confusion_plot()),
            ("Normalized Confusion Matrix", self._confusion_plot(normalize=True)),
        ]
        cards = "".join(
            '<div class="card">'
            f'<div class="card-title">{escape(title)}</div>'
            f'<img src="data:image/png;base64,{encoded}" '
            f'alt="{escape(title)}" style="width:100%;height:auto;">'
            "</div>"
            for title, encoded in plots
        )
        return self._section(
            "Prediction Diagnostics",
            "Rows represent actual classes; columns represent predicted classes. "
            "The normalized matrix divides each row by the number of actual "
            "observations in that class.",
            f'<div class="grid">{cards}</div>',
        )

    # -----------------------------------
    # Confusion Matrix Breakdown
    # -----------------------------------

    def _confusion_breakdown_section(self) -> str:
        matrix = self.confusion.astype(int)

        total = int(matrix.sum())

        records = []

        for index, label in enumerate(self.classes):

            tp = int(matrix[index, index])

            fn = int(matrix[index, :].sum() - tp)

            fp = int(matrix[:, index].sum() - tp)

            tn = int(total - tp - fn - fp)

            records.append(
                {
                    "Class": str(label),
                    "True Positive (TP)": tp,
                    "True Negative (TN)": tn,
                    "False Positive (FP)": fp,
                    "False Negative (FN)": fn,
                }
            )

        breakdown = pd.DataFrame(records)

        description = (
            "Each class is evaluated independently using a One-vs-Rest approach. "
            "TP: Correctly predicted as this class. "
            "TN: Correctly predicted as another class. "
            "FP: Incorrectly predicted as this class. "
            "FN: Incorrectly predicted as another class."
        )

        return self._section(
            "Confusion Matrix Breakdown (TP / TN / FP / FN)",
            description,
            self._table(breakdown),
        )

    def _class_report_section(self) -> str:
        return self._section(
            "Classification Report",
            "Precision, recall, F1-score and support for each class "
            "and aggregate summaries.",
            self._table(self.class_report, index=True),
        )

    def _class_error_section(self) -> str:
        return self._section(
            "Error by Actual Class",
            "Error rate is the proportion of actual observations of each "
            "class that were assigned a different label.",
            self._table(self.class_error_analysis),
        )

    def _error_records_section(self) -> str:
        return self._section(
            "Misclassified Records",
            "Out-of-fold errors ordered by predicted-class probability (highest first). "
            "P(Actual) is the probability assigned to the true class; "
            "P(Predicted) is the probability assigned to the incorrect predicted class. "
            "Expand Features to inspect original values. Do not automatically remove records.",
            self._records_table(self.error_records),
        )

    def _feature_error_section(self) -> str:
        return self._section(
            "Error by Numeric Feature",
            "Pearson correlation between a numeric feature and the binary "
            "misclassification indicator (0 = correct, 1 = incorrect). "
            "This does not establish causality.",
            self._table(
                self.feature_error_analysis,
                empty_message="No numeric features with sufficient variation for analysis.",
            ),
        )

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
            "Descriptive signals for investigation, not proof of a model defect.",
            f'<div class="grid">{cards}</div>',
        )

    def content(self) -> str:
        return (
            f"{self._summary_section()}\n"
            f"{self._plots_section()}\n"
            f"{self._threshold_section()}\n"
            f"{self._confusion_breakdown_section()}\n"
            f"{self._class_report_section()}\n"
            f"{self._class_error_section()}\n"
            f"{self._error_records_section()}\n"
            f"{self._feature_error_section()}\n"
            f"{self._insights_section()}"
        )
