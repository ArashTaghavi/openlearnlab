import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class LabelQualityAnalysisTemplate(BaseTemplate):
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
        min_confidence: float,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title=f"Label Quality Analysis -- {model_name}",
        )
        self.model_name = model_name
        self.candidates = candidates
        self.original = original
        self.corrected = corrected
        self.classes = classes
        self.n_neighbors = n_neighbors
        self.min_disagreement = min_disagreement
        self.n_records = n_records
        self.min_confidence = min_confidence
        self.full_width = True

    @staticmethod
    def _table(df: pd.DataFrame, index: bool = False) -> str:
        if df.empty:
            return '<div class="card"><p>No records found.</p></div>'

        df = df.copy()

        def format_cell(value):
            if not isinstance(value, str):
                return value

            return escape(value).replace("\\n", "\n").replace("\n", "<br>")

        df = df.map(format_cell)

        return (
            '<div class="table-wrapper">'
            + df.to_html(
                classes="data-table",
                index=index,
                border=0,
                escape=False,
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
    def _change(
        before: float,
        after: float,
        *,
        percentage: bool = False,
    ) -> str:
        delta = after - before

        if percentage:
            return f"{delta * 100:+.2f}%"

        return f"{delta:+,.0f}"

    def _model_comparison_results(self) -> str:
        versions = {
            "Original Dataset": self.original["comparison"],
            "Corrected Dataset": self.corrected["comparison"],
        }
        metrics = {
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1": "f1",
            "ROC AUC": "roc_auc",
        }
        best = {
            metric: max(scores[metric]["test_mean"] for scores in versions.values())
            for metric in metrics.values()
        }
        best_fit = min(scores["fit_time"]["mean"] for scores in versions.values())
        best_predict = min(
            scores["predict_time"]["mean"] for scores in versions.values()
        )

        def loss(metric: str, scores: dict) -> str:
            difference = best[metric] - scores[metric]["test_mean"]
            relative = 100 * difference / best[metric] if best[metric] else 0.0
            return f"{relative:.2f}% ({difference:.3f})"

        def speed(best_time: float, time_value: float) -> str:
            if time_value == 0 or best_time == 0:
                return "1.00x" if time_value == best_time else "N/A"
            ratio = time_value / best_time
            return f"{1 / ratio:.2f}x" if ratio <= 1 else f"-{ratio:.2f}x"

        rows = []
        for name, scores in versions.items():
            row = {"Dataset": name}
            for title, metric in metrics.items():
                item = scores[metric]
                gap = item["train_mean"] - item["test_mean"]
                row[title] = (
                    f'Train: {item["train_mean"]:.3f} ± {item["train_std"]:.3f}\n'
                    f'Test: {item["test_mean"]:.3f} ± {item["test_std"]:.3f}\n'
                    f"Gap: {gap:.3f}"
                )
            row["Test Metrics Loss"] = "\n".join(
                f"{title}: {loss(metric, scores)}"
                for title, metric in list(metrics.items())[:4]
            )
            row["Test ROC AUC Loss"] = loss("roc_auc", scores)
            row["Fit / Predict Time (s)"] = (
                f'Fit: {scores["fit_time"]["mean"]:.3f} ± {scores["fit_time"]["std"]:.3f}\n'
                f'Predict: {scores["predict_time"]["mean"]:.3f} ± {scores["predict_time"]["std"]:.3f}'
            )
            row["Speed vs Best"] = (
                f'Predict: {speed(best_predict, scores["predict_time"]["mean"])}\n'
                f'Fit: {speed(best_fit, scores["fit_time"]["mean"])}'
            )
            rows.append(row)

        return self._section(
            "Model Comparison Results",
            "The same model evaluated separately on original and corrected datasets "
            "using 5-fold stratified cross-validation.",
            self._table(pd.DataFrame(rows)),
        )

    def _classification_comparison(self) -> str:
        before = self.original["report"]
        after = self.corrected["report"]
        rows = []

        def add_row(
            name: str,
            original_value: float,
            corrected_value: float,
            *,
            percentage: bool = True,
        ) -> None:
            original_text = (
                f"{original_value * 100:.2f}%"
                if percentage
                else f"{original_value:,.0f}"
            )
            corrected_text = (
                f"{corrected_value * 100:.2f}%"
                if percentage
                else f"{corrected_value:,.0f}"
            )

            rows.append(
                {
                    "Metric": name,
                    "Values": (
                        f"Original: {original_text}\n" f"Corrected: {corrected_text}"
                    ),
                    "Improvement": self._change(
                        original_value,
                        corrected_value,
                        percentage=percentage,
                    ),
                }
            )

        add_row(
            "accuracy",
            float(self.original["accuracy"]),
            float(self.corrected["accuracy"]),
        )

        for label in map(str, self.classes):
            for metric in (
                "precision",
                "recall",
                "f1-score",
                "support",
            ):
                add_row(
                    f"{metric}_{label}",
                    float(before.loc[label, metric]),
                    float(after.loc[label, metric]),
                    percentage=metric != "support",
                )

        for aggregate in ("macro avg", "weighted avg"):
            for metric in (
                "precision",
                "recall",
                "f1-score",
                "support",
            ):
                add_row(
                    f"{metric}_{aggregate.replace(' ', '_')}",
                    float(before.loc[aggregate, metric]),
                    float(after.loc[aggregate, metric]),
                    percentage=metric != "support",
                )

        return self._section(
            "Classification Report Comparison",
            "Per-class and overall results from 5-fold CV.",
            self._table(pd.DataFrame(rows)),
        )

    def _confusion_comparison(self) -> str:
        before = np.asarray(self.original["matrix"])
        after = np.asarray(self.corrected["matrix"])
        rows = []

        for i, actual in enumerate(self.classes):
            for j, predicted in enumerate(self.classes):
                original_count = int(before[i, j])
                corrected_count = int(after[i, j])

                # More correct predictions and fewer errors are improvements.
                direction = 1 if i == j else -1
                improvement = direction * (corrected_count - original_count)
                improvement_percent = (
                    improvement / original_count * 100 if original_count != 0 else None
                )

                change = f"{improvement:+,d}"
                if improvement_percent is not None:
                    change += f" ({improvement_percent:+.2f}%)"
                elif improvement == 0:
                    change += " (0.00%)"
                else:
                    change += " (N/A)"

                rows.append(
                    {
                        "Actual": str(actual),
                        "Predicted": str(predicted),
                        "Type": "Correct" if i == j else "Error",
                        "Values": (
                            f"Original: {original_count:,}\n"
                            f"Corrected: {corrected_count:,}"
                        ),
                        "Improvement": change,
                    }
                )

        return self._section(
            "Confusion Matrix Comparison",
            "Positive improvement means better performance; "
            "negative means worse. "
            "Percentages are relative to the original cell count.",
            self._table(pd.DataFrame(rows)),
        )

    def _classification_chart(self) -> str:
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

            ax.bar(
                x - width / 2,
                before,
                width,
                label="Original",
            )
            ax.bar(
                x + width / 2,
                after,
                width,
                label="Corrected",
            )

            for bars in ax.containers:
                ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

            ax.set_title(metric.title())
            ax.set_xticks(x, labels)
            ax.set_xlabel("Actual Class")
            ax.set_ylim(0, 1.16)
            ax.grid(axis="y", alpha=0.25)

        support_before = [float(original.loc[label, "support"]) for label in labels]
        support_after = [float(corrected.loc[label, "support"]) for label in labels]

        axes[3].bar(
            x - width / 2,
            support_before,
            width,
            label="Original",
        )
        axes[3].bar(
            x + width / 2,
            support_after,
            width,
            label="Corrected",
        )

        for bars in axes[3].containers:
            axes[3].bar_label(bars, fmt="%.0f", padding=3, fontsize=8)

        axes[3].set_title("Support")
        axes[3].set_xticks(x, labels)
        axes[3].set_xlabel("Actual Class")
        axes[3].grid(axis="y", alpha=0.25)

        axes[0].legend()

        fig.suptitle("Classification Report — Original vs Corrected")

        return self._section(
            "Classification Report Comparison Chart",
            "Per-class precision, recall, F1-score and support " "from 5-fold CV.",
            self._figure(fig),
        )

    def _confusion_chart(self) -> str:
        labels = [str(label) for label in self.classes]
        n = len(labels)

        positions = np.arange(n * n)

        before = np.asarray(self.original["matrix"]).ravel()
        after = np.asarray(self.corrected["matrix"]).ravel()

        fig, ax = plt.subplots(figsize=(max(8, n * n * 1.5), 5))

        width = 0.36

        ax.bar(
            positions - width / 2,
            before,
            width,
            label="Original",
        )
        ax.bar(
            positions + width / 2,
            after,
            width,
            label="Corrected",
        )

        ax.set_xticks(
            positions,
            [f"Actual {a}\nPredicted {p}" for a in labels for p in labels],
        )

        for bars in ax.containers:
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)

        ax.set_ylabel("Number of Records")
        ax.set_title("Confusion Matrix — Original vs Corrected")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

        return self._section(
            "Confusion Matrix Comparison Chart",
            "Counts for each actual/predicted class combination " "from 5-fold CV.",
            self._figure(fig),
        )

    @staticmethod
    def _side_by_side(title: str, left: str, right: str) -> str:
        return (
            '<section class="table-section">'
            f"<h2>{escape(title)}</h2>"
            '<div style="display:grid;grid-template-columns:'
            'repeat(auto-fit,minmax(min(100%,420px),1fr));gap:16px;align-items:start;">'
            f'<div class="card"><h3>Original</h3>{left}</div>'
            f'<div class="card"><h3>Corrected</h3>{right}</div>'
            "</div></section>"
        )

    def _separate_confusion_matrices(self) -> str:
        labels = [str(label) for label in self.classes]

        def make_matrix(matrix):
            return self._table(
                pd.DataFrame(np.asarray(matrix), index=labels, columns=labels),
                index=True,
            )

        return self._side_by_side(
            "Confusion Matrices — Original vs Corrected",
            make_matrix(self.original["matrix"]),
            make_matrix(self.corrected["matrix"]),
        )

    def _separate_classification_reports(self) -> str:
        return self._side_by_side(
            "Classification Reports — Original vs Corrected",
            self._table(self.original["report"], index=True),
            self._table(self.corrected["report"], index=True),
        )

    def content(self) -> str:
        changed = int(self.candidates["Selected"].sum())

        configuration = pd.DataFrame(
            [
                ("Model", self.model_name),
                ("Dataset observations", self.n_records),
                (
                    "Evaluation",
                    "5-Fold Stratified CV "
                    "(full dataset, separately for each version)",
                ),
                ("Nearest neighbors", self.n_neighbors),
                ("Minimum confidence (%)", self.min_confidence),
                ("Minimum disagreement (%)", self.min_disagreement),
                (
                    "Confidently wrong records",
                    len(self.candidates),
                ),
                (
                    "Required opposing votes",
                    f"at least "
                    f"{int(np.ceil(self.n_neighbors * self.min_disagreement / 100))} "
                    f"of {self.n_neighbors}, and a strict majority",
                ),
                ("Corrected labels", changed),
            ],
            columns=["Setting", "Value"],
        )

        return (
            self._section(
                "Test Configuration",
                "All rows are retained. Each version is evaluated "
                "using 5-fold out-of-fold predictions. "
                f"The {self.n_neighbors} nearest neighbors are selected "
                "by Euclidean distance after numeric scaling and "
                "categorical encoding; the disagreement threshold "
                "applies to opposing labels. "
                "Confidently misclassified records are excluded "
                "from the neighbor pool.",
                self._table(configuration),
            )
            + self._model_comparison_results()
            + self._classification_comparison()
            + self._confusion_comparison()
            + self._classification_chart()
            + self._confusion_chart()
            + self._separate_confusion_matrices()
            + self._separate_classification_reports()
        )
