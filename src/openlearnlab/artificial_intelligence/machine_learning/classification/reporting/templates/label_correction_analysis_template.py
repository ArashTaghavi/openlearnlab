import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class LabelCorrectionAnalysisTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        candidates: pd.DataFrame,
        original: dict,
        corrected: dict,
        classes: np.ndarray,
        n_neighbors: int,
        min_disagreement: float,
        n_records: int,
        dataset_filename: str,
        review_filename: str,
        features: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title=f"Label Correction Analysis -- {model_name}",
        )
        self.model_name = model_name
        self.candidates = candidates
        self.original = original
        self.corrected = corrected
        self.classes = classes
        self.n_neighbors = n_neighbors
        self.min_disagreement = min_disagreement
        self.n_records = n_records
        self.dataset_filename = dataset_filename
        self.review_filename = review_filename
        self.full_width = True

    @staticmethod
    def _table(df: pd.DataFrame, index: bool = False) -> str:
        if df.empty:
            return '<div class="card"><p>No records found.</p></div>'
        return (
            '<div style="overflow-x:auto;">'
            + df.to_html(
                classes="data-table",
                index=index,
                border=0,
                escape=True,
                na_rep="—",
                float_format=lambda value: f"{value:,.4f}",
            )
            + "</div>"
        )

    @staticmethod
    def _section(title: str, description: str, body: str) -> str:
        return (
            '<section class="table-section">'
            f"<h2>{escape(title)}</h2><p>{escape(description)}</p>{body}"
            "</section>"
        )

    @staticmethod
    def _figure(fig) -> str:
        buffer = BytesIO()
        try:
            fig.tight_layout()
            fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        finally:
            plt.close(fig)
            buffer.close()
        return (
            '<div class="card">'
            f'<img src="data:image/png;base64,{encoded}" '
            'style="width:100%;height:auto;" alt="Model comparison chart">'
            "</div>"
        )

    @staticmethod
    def _comparison_cards(original_body: str, corrected_body: str) -> str:
        return (
            '<div style="display:grid;grid-template-columns:'
            'repeat(2,minmax(0,1fr));gap:16px;align-items:start;">'
            '<div class="card" style="min-width:0;">'
            "<h3>Original Dataset</h3>" + original_body + "</div>"
            '<div class="card" style="min-width:0;">'
            "<h3>Corrected Dataset</h3>" + corrected_body + "</div>"
            "</div>"
        )

    def _evaluation_comparison(self) -> str:
        def matrix_table(evaluation: dict) -> str:
            matrix = pd.DataFrame(
                evaluation["matrix"],
                index=[f"Actual {label}" for label in self.classes],
                columns=[f"Predicted {label}" for label in self.classes],
            )
            return f'<p>Accuracy: {evaluation["accuracy"]:.4f}</p>' + self._table(
                matrix, index=True
            )

        confusion = self._section(
            "Confusion Matrix",
            "Original vs corrected — 5-fold cross-validation.",
            self._comparison_cards(
                matrix_table(self.original), matrix_table(self.corrected)
            ),
        )
        classification = self._section(
            "Classification Report",
            "Original vs corrected — 5-fold cross-validation.",
            self._comparison_cards(
                self._table(self.original["report"], index=True),
                self._table(self.corrected["report"], index=True),
            ),
        )
        return confusion + classification

    def _classification_chart(self) -> str:
        # Compare per-class metrics; support is a count, so use a separate panel.
        labels = [str(label) for label in self.classes]
        original = self.original["report"]
        corrected = self.corrected["report"]
        metrics = ["precision", "recall", "f1-score"]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        width = 0.36
        for ax, metric in zip(axes[:3], metrics):
            before = [float(original.loc[label, metric]) for label in labels]
            after = [float(corrected.loc[label, metric]) for label in labels]
            ax.bar(x - width / 2, before, width, label="Original")
            ax.bar(x + width / 2, after, width, label="Corrected")
            ax.set_title(metric.title())
            ax.set_xticks(x, labels)
            ax.set_xlabel("Actual Class")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.25)
        support_before = [float(original.loc[label, "support"]) for label in labels]
        support_after = [float(corrected.loc[label, "support"]) for label in labels]
        axes[3].bar(x - width / 2, support_before, width, label="Original")
        axes[3].bar(x + width / 2, support_after, width, label="Corrected")
        axes[3].set_title("Support")
        axes[3].set_xticks(x, labels)
        axes[3].set_xlabel("Actual Class")
        axes[3].grid(axis="y", alpha=0.25)
        axes[0].legend()
        fig.suptitle("Classification Report — Original vs Corrected")
        return self._section(
            "Classification Report Comparison",
            "Per-class precision, recall, F1-score and support from 5-fold CV.",
            self._figure(fig),
        )

    def _confusion_chart(self) -> str:
        labels = [str(label) for label in self.classes]
        n = len(labels)
        positions = np.arange(n * n)
        before = self.original["matrix"].ravel()
        after = self.corrected["matrix"].ravel()
        fig, ax = plt.subplots(figsize=(max(8, n * n * 1.5), 5))
        width = 0.36
        ax.bar(positions - width / 2, before, width, label="Original")
        ax.bar(positions + width / 2, after, width, label="Corrected")
        ax.set_xticks(
            positions,
            [f"Actual {a}\nPredicted {p}" for a in labels for p in labels],
        )
        ax.set_ylabel("Number of Records")
        ax.set_title("Confusion Matrix — Original vs Corrected")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        return self._section(
            "Confusion Matrix Comparison",
            "Counts for each actual/predicted class combination from 5-fold CV.",
            self._figure(fig),
        )

    def content(self) -> str:
        changed = int(self.candidates["Selected"].sum())
        configuration = pd.DataFrame(
            [
                ("Model", self.model_name),
                ("Dataset observations", self.n_records),
                (
                    "Evaluation",
                    "5-Fold Stratified CV (full dataset, separately for each version)",
                ),
                ("Nearest neighbors", self.n_neighbors),
                ("Minimum disagreement (%)", self.min_disagreement),
                (
                    "Required opposing votes",
                    f"at least {int(np.ceil(self.n_neighbors * self.min_disagreement / 100))} of {self.n_neighbors}, and a strict majority",
                ),
                ("Corrected labels", changed),
            ],
            columns=["Setting", "Value"],
        )
        return (
            self._section(
                "Test Configuration",
                "All rows are retained. Each version is evaluated using 5-fold "
                "out-of-fold predictions. The 10 nearest neighbors are selected by "
                "Euclidean distance after numeric scaling and categorical encoding; "
                "the threshold applies to the percentage of opposing labels.",
                self._table(configuration),
            )
            + self._evaluation_comparison()
            + self._classification_chart()
            + self._confusion_chart()
        )
