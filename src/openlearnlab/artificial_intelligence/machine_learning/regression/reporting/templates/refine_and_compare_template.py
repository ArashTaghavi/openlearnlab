from html import escape

import numpy as np
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class RefineAndCompareTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        folds: pd.DataFrame,
        n_samples: int,
        n_high_error: int,
        n_outlier: int,
        n_overlap: int,
        threshold: float,
        strategy_counts: dict[str, int],
        n_folds: int,
    ):
        super().__init__(dataset_name=dataset_name, title="Refine and Compare")
        self.folds = folds.copy()
        self.n_samples = n_samples
        self.n_high_error = n_high_error
        self.n_outlier = n_outlier
        self.n_overlap = n_overlap
        self.threshold = threshold
        self.strategy_counts = strategy_counts.copy()
        self.n_folds = n_folds
        self.full_width = True

    @staticmethod
    def _section(title: str, description: str, body: str) -> str:
        return (
            '<section class="table-section">'
            f"<h2>{escape(title)}</h2><p>{escape(description)}</p>{body}"
            "</section>"
        )

    def _summary(self) -> str:
        removed = self.strategy_counts["High Error + Outlier"]
        percentage = 100 * removed / self.n_samples if self.n_samples else 0.0

        values = [
            ("Original Records", f"{self.n_samples:,}"),
            ("Removed Records", f"{removed:,} ({percentage:.2f}%)"),
            ("Refined Records", f"{self.n_samples - removed:,}"),
            ("High-Error Flags", f"{self.n_high_error:,}"),
            ("Target Outlier Flags", f"{self.n_outlier:,}"),
            ("Both Flags", f"{self.n_overlap:,}"),
        ]

        cards = "".join(
            '<div class="card">'
            f'<div class="card-title">{escape(label)}</div>'
            f"<pre>{escape(value)}</pre>"
            "</div>"
            for label, value in values
        )

        return self._section(
            "Refinement Summary",
            f"High error: absolute OOF error > 1.5 × MAE = {self.threshold:.4f}. "
            "Target outlier: 1.5 × IQR. Counts describe the full-data exported "
            "High Error + Outlier dataset; overlapping flags count once.",
            f'<div class="grid">{cards}</div>',
        )

    @staticmethod
    def _mean_std(
        frame: pd.DataFrame,
        column: str,
    ) -> tuple[float, float]:
        if frame.empty or column not in frame.columns:
            return np.nan, np.nan

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        # Do not silently average only successful folds.
        if not np.isfinite(values).all():
            return np.nan, np.nan

        return (
            float(values.mean()),
            float(values.std(ddof=0)),
        )

    @staticmethod
    def _number(
        value: float,
        signed: bool = False,
    ) -> str:
        if not np.isfinite(value):
            return "N/A"

        return f"{value:+.4f}" if signed else f"{value:.4f}"

    def _metric_cell(
        self,
        folds: pd.DataFrame,
        metric: str,
    ) -> str:
        train_mean, train_std = self._mean_std(
            folds,
            f"Train {metric}",
        )

        test_mean, test_std = self._mean_std(
            folds,
            f"Test {metric}",
        )

        gap = train_mean - test_mean if metric == "R²" else test_mean - train_mean

        return (
            '<div style="white-space:nowrap;line-height:1.9;text-align:center">'
            f"Train: {self._number(train_mean)} ± {self._number(train_std)}<br>"
            f"Test: {self._number(test_mean)} ± {self._number(test_std)}<br>"
            f"Gap: {self._number(gap)}"
            "</div>"
        )

    def _comparison(self) -> str:
        scenarios = []

        for (strategy, evaluation), folds in self.folds.groupby(
            ["Strategy", "Evaluation"],
            sort=False,
        ):
            test_r2, _ = self._mean_std(
                folds,
                "Test R²",
            )

            test_rmse, _ = self._mean_std(
                folds,
                "Test RMSE",
            )

            removed, _ = self._mean_std(
                folds,
                "Removed Train (%)",
            )

            test_records, _ = self._mean_std(
                folds,
                "Test Records",
            )

            scenarios.append(
                {
                    "strategy": strategy,
                    "evaluation": evaluation,
                    "folds": folds,
                    "r2": test_r2,
                    "rmse": test_rmse,
                    "removed": removed,
                    "test_records": test_records,
                }
            )

        # Sort all scenarios by descending Test R².
        scenarios.sort(
            key=lambda item: (item["r2"] if np.isfinite(item["r2"]) else -np.inf),
            reverse=True,
        )

        # Scenarios evaluated on untouched original test folds.
        original_test = [
            item for item in scenarios if item["evaluation"] != "Refined → Refined"
        ]

        # Scenarios evaluated on refined test subsets.
        refined_test = [
            item for item in scenarios if item["evaluation"] == "Refined → Refined"
        ]

        valid_original = [item for item in original_test if np.isfinite(item["r2"])]

        valid_refined = [item for item in refined_test if np.isfinite(item["r2"])]

        # Best Generalization: highest mean Test R²
        # on untouched original test folds.
        best_original = (
            max(
                valid_original,
                key=lambda item: item["r2"],
            )
            if valid_original
            else None
        )

        # Best Refined Subset: highest mean Test R²
        # on outcome-selected refined test subsets.
        best_refined = (
            max(
                valid_refined,
                key=lambda item: item["r2"],
            )
            if valid_refined
            else None
        )

        style = 'style="text-align:center;vertical-align:middle"'

        rows = []

        for item in scenarios:
            reference = (
                best_refined
                if item["evaluation"] == "Refined → Refined"
                else best_original
            )

            r2_delta = item["r2"] - reference["r2"] if reference else np.nan

            rmse_delta = item["rmse"] - reference["rmse"] if reference else np.nan

            # Highlight winning rows with dark text.
            if item is best_original:
                background = "#dcfce7"
                foreground = "#14532d"
            elif item is best_refined:
                background = "#dbeafe"
                foreground = "#1e3a8a"
            else:
                background = None
                foreground = None

            cell_style = (
                'style="text-align:center;vertical-align:middle;'
                f"background-color:{background} !important;"
                f'color:{foreground} !important"'
                if background
                else style
            )
            
            metric_cells = "".join(
                f"<td {cell_style}>"
                f'{self._metric_cell(item["folds"], metric)}'
                "</td>"
                for metric in ("R²", "RMSE", "MAE")
            )

            changes = (
                '<div style="white-space:nowrap;line-height:1.9;text-align:center">'
                f"Test R²: {self._number(r2_delta, signed=True)}<br>"
                f"Test RMSE: {self._number(rmse_delta, signed=True)}"
                "</div>"
            )

            rows.append(
                "<tr>"
                f"<td {cell_style}>"
                f'<strong>{escape(item["strategy"])}</strong>'
                "</td>"
                f"<td {cell_style}>"
                f'{escape(item["evaluation"])}'
                "</td>"
                f"{metric_cells}"
                f"<td {cell_style}>"
                f'{self._number(item["removed"])}%'
                "</td>"
                f"<td {cell_style}>"
                f'{self._number(item["test_records"])}'
                "</td>"
                f"<td {cell_style}>"
                f"{changes}"
                "</td>"
                "</tr>"
            )

        headers = (
            "Strategy",
            "Evaluation",
            "R²",
            "RMSE",
            "MAE",
            "Removed (%)",
            "Test Records / Fold",
            "Improvements",
        )

        table = (
            '<div style="overflow-x:auto">'
            '<table class="data-table" '
            'style="width:100%;text-align:center">'
            "<thead><tr>"
            + "".join(
                f'<th style="text-align:center">' f"{escape(header)}" "</th>"
                for header in headers
            )
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

        # Best Results table.
        def best_row(
            title: str,
            item: dict | None,
        ) -> str:
            if item is None:
                return (
                    f"<tr><th {style}>"
                    f"{escape(title)}"
                    "</th>"
                    f'<td {style} colspan="4">N/A</td>'
                    "</tr>"
                )

            return (
                "<tr>"
                f"<th {style}>{escape(title)}</th>"
                f"<td {style}>"
                f'<strong>{escape(item["strategy"])}</strong>'
                "</td>"
                f'<td {style}>{self._number(item["r2"])}</td>'
                f'<td {style}>{self._number(item["rmse"])}</td>'
                f'<td {style}>{escape(item["evaluation"])}</td>'
                "</tr>"
            )

        best_headers = (
            "Result",
            "Best Strategy",
            "Test R²",
            "Test RMSE",
            "Evaluation",
        )

        best_table = (
            '<div style="overflow-x:auto">'
            '<table class="data-table" '
            'style="width:100%;text-align:center">'
            "<thead><tr>"
            + "".join(
                f'<th style="text-align:center">' f"{escape(header)}" "</th>"
                for header in best_headers
            )
            + "</tr></thead><tbody>"
            + best_row(
                "Best Generalization",
                best_original,
            )
            + best_row(
                "Best Refined Subset",
                best_refined,
            )
            + "</tbody></table></div>"
        )

        description = (
            f"{self.n_folds}-fold outer cross-validation. "
            "All training filters are learned inside each outer training fold. "
            "Original-test results use identical, untouched test folds. "
            "Refined-test results evaluate post-hoc subsets selected using "
            "test outcomes and are descriptive, not unbiased generalization "
            "estimates. "
            "Rows are sorted by Test R² across both populations; "
            "these R² values are not directly comparable. "
            "Improvements are current minus the best within the same "
            "evaluation population. "
            "Removed (%) describes training rows only. "
            "Best Generalization (green) is the strategy with the highest "
            "mean Test R² across outer folds when evaluated on untouched "
            "original test data. "
            "Best Refined Subset (blue) is the strategy with the highest "
            "mean Test R² on refined test subsets; this result is "
            "descriptive only."
        )

        return self._section(
            "Model Comparison",
            description,
            table + "<h3>Best Results</h3>" + best_table,
        )

    def content(self) -> str:
        return f"{self._summary()}\n{self._comparison()}"
