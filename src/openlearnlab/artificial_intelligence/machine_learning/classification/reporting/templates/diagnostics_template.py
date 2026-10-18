import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from ....base.reporting.base_template import BaseTemplate


class DiagnosticsTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: np.ndarray,
        confusion_matrix: np.ndarray,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        roc_auc: float | None,
        log_loss: float | None,
        brier_score: float | None,
        error_rate: float,
        n_errors: int,
        insights: list[tuple[str, str]],
        n_samples: int,
        n_features: int,
        n_folds: int,
        critical_errors: pd.DataFrame,
        probabilities: np.ndarray | None,
        decision_scores: np.ndarray | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Classification Diagnostics",
            full_width=True,
        )

        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.classes = np.asarray(classes)
        self.confusion_matrix = np.asarray(confusion_matrix)

        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.roc_auc = roc_auc
        self.log_loss = log_loss
        self.brier_score = brier_score
        self.error_rate = error_rate
        self.n_errors = n_errors

        self.insights = insights
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_folds = n_folds
        self.critical_errors = critical_errors.copy()
        self.probabilities = (
            np.asarray(probabilities, dtype=float)
            if probabilities is not None
            else None
        )
        self.decision_scores = (
            np.asarray(decision_scores, dtype=float)
            if decision_scores is not None
            else None
        )

        if len(self.classes) != 2 or self.confusion_matrix.shape != (2, 2):
            raise ValueError("DiagnosticsTemplate requires binary classification.")

        for scores in (self.probabilities, self.decision_scores):
            if scores is not None and (
                scores.ndim != 1
                or len(scores) != len(self.y_true)
                or not np.all(np.isfinite(scores))
            ):
                raise ValueError("Prediction scores must be finite and 1-dimensional.")

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
    def _metric(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "N/A"

    def _binary_target(self) -> np.ndarray:
        return (self.y_true == self.classes[1]).astype(int)

    def _ranking_scores(self) -> np.ndarray | None:
        if self.probabilities is not None:
            return self.probabilities
        return self.decision_scores

    def _confusion_matrix_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cm = self.confusion_matrix
        image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks([0, 1], labels=self.classes.astype(str))
        ax.set_yticks([0, 1], labels=self.classes.astype(str))
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("Confusion Matrix")

        threshold = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=13,
                    fontweight="bold",
                )

        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _roc_curve_plot(self) -> str | None:
        scores = self._ranking_scores()
        if scores is None:
            return None

        fpr, tpr, _ = roc_curve(self._binary_target(), scores)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {self._metric(self.roc_auc)}")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.3, label="Random")
        ax.set(
            xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right")
        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _precision_recall_plot(self) -> str | None:
        scores = self._ranking_scores()
        if scores is None:
            return None

        y_binary = self._binary_target()
        precision, recall, _ = precision_recall_curve(y_binary, scores)
        ap = average_precision_score(y_binary, scores)
        prevalence = float(y_binary.mean())

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(recall, precision, linewidth=2, label=f"Average Precision = {ap:.4f}")
        ax.axhline(
            prevalence,
            color="red",
            linestyle="--",
            linewidth=1.3,
            label=f"Baseline = {prevalence:.4f}",
        )
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left")
        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _confidence_distribution_plot(self) -> str | None:
        if self.probabilities is None:
            return None

        confidence = np.where(
            self.y_pred == self.classes[1],
            self.probabilities,
            1 - self.probabilities,
        )
        correct = self.y_true == self.y_pred
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bins = np.linspace(0, 1, 21)
        ax.hist(
            confidence[correct],
            bins=bins,
            alpha=0.65,
            label="Correct",
            edgecolor="white",
        )
        ax.hist(
            confidence[~correct],
            bins=bins,
            alpha=0.65,
            label="Incorrect",
            edgecolor="white",
        )
        ax.axvline(0.9, color="red", linestyle="--", label="90% Confidence")
        ax.set(
            xlabel="Predicted Class Confidence",
            ylabel="Frequency",
            title="Prediction Confidence Distribution",
        )
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        return self._figure_to_base64(fig)

    def _classification_report_section(self) -> str:
        report = classification_report(
            self.y_true,
            self.y_pred,
            labels=self.classes.tolist(),
            output_dict=True,
            zero_division=0,
        )

        rows = []
        for class_label in self.classes:
            label = str(class_label)
            metrics = report[label]
            rows.append((label, metrics))

        rows.extend(
            [
                ("Macro Avg", report["macro avg"]),
                ("Weighted Avg", report["weighted avg"]),
            ]
        )

        table_rows = "".join(
            "<tr>"
            f"<th>{escape(label)}</th>"
            f"<td>{metrics['precision']:.4f}</td>"
            f"<td>{metrics['recall']:.4f}</td>"
            f"<td>{metrics['f1-score']:.4f}</td>"
            f"<td>{int(metrics['support']):,}</td>"
            "</tr>"
            for label, metrics in rows
        )

        return f"""
        <section class="table-section">
            <h2>Classification Report</h2>
            <div style="overflow-x:auto;">
                <table class="data-table">
                    <thead><tr>
                        <th>Class / Average</th>
                        <th>Precision</th><th>Recall</th>
                        <th>F1 Score</th><th>Support</th>
                    </tr></thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
            <p>
                <strong>Macro Avg:</strong> Equal weight per class.
                <strong>Weighted Avg:</strong> Weighted by class support.
            </p>
        </section>
        """

    def _confusion_matrix_section(self) -> str:
        tn, fp, fn, tp = (int(value) for value in self.confusion_matrix.ravel())
        negative_label = escape(str(self.classes[0]))
        positive_label = escape(str(self.classes[1]))
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0

        return f"""
        <section class="table-section">
            <h2>Confusion Matrix Summary</h2>
            <div style="overflow-x:auto;">
                <table class="data-table">
                    <thead><tr>
                        <th>Actual / Predicted</th>
                        <th>Predicted {negative_label}</th>
                        <th>Predicted {positive_label}</th>
                        <th>Total</th>
                    </tr></thead>
                    <tbody>
                        <tr><th>Actual {negative_label}</th>
                            <td>{tn:,} (TN)</td><td>{fp:,} (FP)</td>
                            <td>{tn + fp:,}</td></tr>
                        <tr><th>Actual {positive_label}</th>
                            <td>{fn:,} (FN)</td><td>{tp:,} (TP)</td>
                            <td>{fn + tp:,}</td></tr>
                        <tr><th>Total</th>
                            <td>{tn + fn:,}</td><td>{fp + tp:,}</td>
                            <td>{tn + fp + fn + tp:,}</td></tr>
                    </tbody>
                </table>
            </div>
            <p>
                <strong>TN:</strong> Correct negative;
                <strong>FP:</strong> Incorrect positive;
                <strong>FN:</strong> Missed positive;
                <strong>TP:</strong> Correct positive.
                <strong>FPR:</strong> {fpr:.2%};
                <strong>FNR:</strong> {fnr:.2%}.
            </p>
        </section>
        """

    def _summary_section(self) -> str:
        return f"""
        <section class="table-section">
            <h2>Classification Overview</h2>
            <p>
                <strong>Accuracy:</strong> {self.accuracy:.4f} &nbsp; | &nbsp;
                <strong>F1:</strong> {self.f1:.4f} &nbsp; | &nbsp;
                <strong>ROC AUC:</strong> {self._metric(self.roc_auc)} &nbsp; | &nbsp;
                <strong>Errors:</strong> {self.n_errors:,}/{self.n_samples:,}
                ({self.error_rate:.2%})
            </p>
            <p>
                <strong>Log Loss:</strong> {self._metric(self.log_loss)} &nbsp; | &nbsp;
                <strong>Brier Score:</strong> {self._metric(self.brier_score)} &nbsp; | &nbsp;
                <strong>Features:</strong> {self.n_features:,} &nbsp; | &nbsp;
                <strong>Evaluation:</strong> {self.n_folds}-Fold Stratified CV
            </p>
        </section>
        """

    def _high_error_observations(self) -> str:
        if self.critical_errors.empty:
            table_html = "<p>No misclassified observations were detected.</p>"
        else:
            display_df = self.critical_errors.copy()
            display_df.index.name = "Row Index"
            table_html = display_df.to_html(
                classes="data-table",
                index=True,
                border=0,
                escape=True,
                na_rep="—",
                float_format=lambda value: f"{value:.4f}",
            )
            table_html = f'<div style="overflow-x:auto;">{table_html}</div>'

        return f"""
        <section class="table-section">
            <h2>Misclassified Observations</h2>
            <p>
                Out-of-fold errors, sorted by predicted confidence when
                probability estimates are available. An error does not
                necessarily indicate an incorrect data label.
            </p>
            {table_html}
        </section>
        """

    def _insights_section(self) -> str:
        cards = "".join(
            f'<div class="card"><div class="card-title">{escape(title)}</div>'
            f"<p>{escape(description)}</p></div>"
            for title, description in self.insights
        )
        return f"""
        <section class="table-section">
            <h2>Diagnostic Insights</h2>
            <div class="grid">{cards}</div>
        </section>
        """

    def _plots_section(self) -> str:
        plots = [
            ("Confusion Matrix", self._confusion_matrix_plot()),
            ("ROC Curve", self._roc_curve_plot()),
            ("Precision–Recall Curve", self._precision_recall_plot()),
            (
                "Prediction Confidence Distribution",
                self._confidence_distribution_plot(),
            ),
        ]

        cards = "".join(
            f'<div class="card"><div class="card-title">{escape(title)}</div>'
            f'<img src="data:image/png;base64,{plot}" '
            f'alt="{escape(title)}" style="width:100%;height:auto;"></div>'
            for title, plot in plots
            if plot is not None
        )
        return f"""
        <section class="table-section">
            <h2>Classification Diagnostics</h2>
            <div class="grid">{cards}</div>
        </section>
        """

    def content(self) -> str:
        return f"""
        {self._summary_section()}
        {self._classification_report_section()}
        {self._confusion_matrix_section()}
        {self._plots_section()}
        {self._high_error_observations()}
        {self._insights_section()}
        """
