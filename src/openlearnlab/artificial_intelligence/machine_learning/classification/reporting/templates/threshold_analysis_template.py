import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class ThresholdAnalysisTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        metric: str,
        average: int | str,
        classes: np.ndarray,
        default: dict,
        results: list[dict],
        n_samples: int,
        n_features: int,
        n_folds: int,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title=f"Classification Threshold Analysis -- {model_name}",
        )

        self.model_name = model_name
        self.metric = metric
        self.average = average
        self.classes = classes
        self.default = default
        self.results = results
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_folds = n_folds
        self.full_width = True

    @staticmethod
    def _table(
        df: pd.DataFrame,
        *,
        index: bool = False,
    ) -> str:
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
    def _section(
        title: str,
        description: str,
        body: str,
    ) -> str:
        return (
            '<section class="table-section">'
            f"<h2>{escape(title)}</h2>"
            f"<p>{escape(description)}</p>"
            f"{body}"
            "</section>"
        )

    def _confusion_table(self, result: dict) -> str:
        matrix = pd.DataFrame(
            result["Confusion Matrix"],
            index=[f"Actual {label}" for label in self.classes],
            columns=[f"Predicted {label}" for label in self.classes],
        )

        return "<h3>Confusion Matrix</h3>" f"{self._table(matrix, index=True)}"

    def _classification_table(self, result: dict) -> str:
        return (
            "<h3>Classification Report</h3>"
            f"{self._table(result['Classification Report'], index=True)}"
        )

    def _default_section(self) -> str:
        return self._section(
            "Default Model Performance",
            "Original model performance at decision threshold 0.50.",
            (
                '<div class="grid">'
                '<div class="card">'
                '<div class="card-title">Threshold</div>'
                f"<pre>{self.default['Threshold']:.4f}</pre>"
                "</div>"
                '<div class="card">'
                '<div class="card-title">Target Score</div>'
                f"<pre>{self.default['Target Score']:.4f}</pre>"
                "</div>"
                '<div class="card">'
                '<div class="card-title">Accuracy</div>'
                f"<pre>{self.default['Accuracy']:.4f}</pre>"
                "</div>"
                "</div>"
                f"{self._confusion_table(self.default)}"
                f"{self._classification_table(self.default)}"
            ),
        )

    def _results_section(self) -> str:
        cards = []

        for index, result in enumerate(self.results):
            is_default = index == len(self.results) - 1

            title = "Default Threshold" if is_default else f"Result #{index + 1}"

            cards.append(
                '<div class="card">'
                f'<div class="card-title">{escape(title)}</div>'
                '<div class="grid">'
                '<div class="card">'
                '<div class="card-title">Threshold</div>'
                f"<pre>{result['Threshold']:.6f}</pre>"
                "</div>"
                '<div class="card">'
                '<div class="card-title">Target Score</div>'
                f"<pre>{result['Target Score']:.4f}</pre>"
                "</div>"
                '<div class="card">'
                '<div class="card-title">Accuracy</div>'
                f"<pre>{result['Accuracy']:.4f}</pre>"
                "</div>"
                "</div>"
                f"{self._confusion_table(result)}"
                f"{self._classification_table(result)}"
                "</div>"
            )

        return self._section(
            "Threshold Optimization Results",
            "Results are sorted by the selected metric, from the "
            "highest score down to the default model score. "
            "Each distinct score appears once, using the threshold "
            "with the highest accuracy when scores are equal.",
            f'<div class="grid">{"".join(cards)}</div>',
        )

    def _metrics_plot_section(self) -> str:
        results = sorted(
            self.results,
            key=lambda row: row["Threshold"],
        )

        thresholds = [row["Threshold"] for row in results]
        precision = [row["Precision"] for row in results]
        recall = [row["Recall"] for row in results]
        accuracy = [row["Accuracy"] for row in results]

        fig, ax = plt.subplots(figsize=(10, 5.5))

        ax.plot(
            thresholds,
            recall,
            label=f"Recall ({self.average})",
            color="tab:orange",
            linewidth=2,
            marker="o",
            markersize=3,
        )

        ax.plot(
            thresholds,
            precision,
            label=f"Precision ({self.average})",
            color="tab:blue",
            linewidth=2,
            marker="o",
            markersize=3,
        )

        ax.plot(
            thresholds,
            accuracy,
            label="Accuracy",
            color="tab:green",
            linewidth=2,
            marker="o",
            markersize=3,
        )

        ax.axvline(
            0.5,
            color="tab:red",
            linestyle="--",
            linewidth=1.8,
            label="Default Threshold (0.50)",
        )

        ax.set_title("Precision, Recall and Accuracy vs Threshold")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Metric Score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()

        buffer = BytesIO()

        try:
            fig.savefig(
                buffer,
                format="png",
                dpi=140,
                bbox_inches="tight",
            )
            buffer.seek(0)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        finally:
            plt.close(fig)
            buffer.close()

        return self._section(
            "Precision / Recall / Accuracy vs Threshold",
            "Shows how precision, recall and accuracy change across "
            "the evaluated decision thresholds. Precision and recall "
            "use the selected class or averaging method. "
            "The dashed line marks the default threshold of 0.50.",
            (
                '<div class="card">'
                f'<img src="data:image/png;base64,{encoded}" '
                'alt="Precision Recall Accuracy vs Threshold" '
                'style="width:100%;height:auto;">'
                "</div>"
            ),
        )

    def content(self) -> str:
        target = (
            "Accuracy"
            if self.metric == "accuracy"
            else f"{self.metric.title()} / {self.average}"
        )

        configuration = self._section(
            "Analysis Configuration",
            "Selected metric and evaluation settings.",
            (
                '<div class="grid">'
                '<div class="card">'
                '<div class="card-title">Model</div>'
                f"<pre>{escape(self.model_name)}</pre>"
                "</div>"
                '<div class="card">'
                '<div class="card-title">Target Metric</div>'
                f"<pre>{escape(target)}</pre>"
                "</div>"
                '<div class="card">'
                '<div class="card-title">Evaluation</div>'
                f"<pre>{self.n_folds}-Fold Stratified CV</pre>"
                "</div>"
                "</div>"
            ),
        )

        return (
            f"{configuration}\n"
            f"{self._default_section()}\n"
            f"{self._results_section()}\n"
            f"{self._metrics_plot_section()}"
        )
