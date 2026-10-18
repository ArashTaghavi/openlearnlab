import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class RefineAndEvaluateTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        comparison: pd.DataFrame,
        scenarios: list[dict],
        n_samples: int,
        n_train: int,
        n_test: int,
        n_removed: int,
        n_retained: int,
        n_typical_test: int,
        n_folds: int,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Refinement and Evaluation",
        )

        self.comparison = comparison.copy()
        self.scenarios = scenarios

        self.n_samples = n_samples
        self.n_removed = n_removed
        self.n_retained = n_retained
        self.n_folds = n_folds

    # -----------------------------------
    # Figure Utilities
    # -----------------------------------

    @staticmethod
    def _figure_to_base64(fig) -> str:
        buffer = BytesIO()

        try:
            fig.savefig(
                buffer,
                format="png",
                dpi=140,
                bbox_inches="tight",
            )

            buffer.seek(0)

            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        finally:
            plt.close(fig)
            buffer.close()

    # -----------------------------------
    # Metric Utilities
    # -----------------------------------

    @staticmethod
    def _mean_std(
        values: pd.Series | np.ndarray,
    ) -> tuple[float, float]:
        values_array = np.asarray(
            values,
            dtype=float,
        )

        values_array = values_array[np.isfinite(values_array)]

        if len(values_array) == 0:
            return float("nan"), float("nan")

        return (
            float(np.mean(values_array)),
            float(np.std(values_array, ddof=0)),
        )

    @staticmethod
    def _format_mean_std(
        mean: float,
        std: float,
    ) -> str:
        if not np.isfinite(mean):
            return "N/A"

        return f"{mean:.3f} ± {std:.3f}"

    @staticmethod
    def _format_change(value: float) -> str:
        if not np.isfinite(value):
            return "N/A"

        return f"{value:+.4f}"

    # -----------------------------------
    # Actual vs Predicted
    # -----------------------------------

    def _actual_vs_predicted_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))

        if len(y_true):
            ax.scatter(
                y_true,
                y_pred,
                alpha=0.65,
                s=25,
            )

            minimum = float(
                min(
                    np.min(y_true),
                    np.min(y_pred),
                )
            )

            maximum = float(
                max(
                    np.max(y_true),
                    np.max(y_pred),
                )
            )

            ax.plot(
                [minimum, maximum],
                [minimum, maximum],
                linestyle="--",
                color="red",
                linewidth=1.5,
            )

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")
        ax.grid(alpha=0.2)

        fig.tight_layout()

        return self._figure_to_base64(fig)

    # -----------------------------------
    # Residuals vs Predicted
    # -----------------------------------

    def _residuals_vs_predicted_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))

        residuals = y_true - y_pred

        if len(y_true):
            ax.scatter(
                y_pred,
                residuals,
                alpha=0.65,
                s=25,
            )

        ax.axhline(
            0,
            linestyle="--",
            color="red",
            linewidth=1.5,
        )

        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Predicted")
        ax.grid(alpha=0.2)

        fig.tight_layout()

        return self._figure_to_base64(fig)

    # -----------------------------------
    # Residual Distribution
    # -----------------------------------

    def _residual_distribution_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))

        residuals = y_true - y_pred

        if len(residuals):
            ax.hist(
                residuals,
                bins="auto",
                edgecolor="white",
                alpha=0.85,
            )

        ax.axvline(
            0,
            linestyle="--",
            color="red",
            linewidth=1.5,
        )

        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution")
        ax.grid(axis="y", alpha=0.2)

        fig.tight_layout()

        return self._figure_to_base64(fig)

    # -----------------------------------
    # Summary Cards
    # -----------------------------------

    def _summary_section(self) -> str:
        removed_share = (
            100 * self.n_removed / self.n_samples if self.n_samples > 0 else 0.0
        )

        retained_share = (
            100 * self.n_retained / self.n_samples if self.n_samples > 0 else 0.0
        )

        return f"""
        <div class="grid">
            <div class="card">
                <div class="card-title">Original Records</div>
                <pre>{self.n_samples:,}</pre>
            </div>

            <div class="card">
                <div class="card-title">Removed Records</div>
                <pre>{self.n_removed:,}</pre>
                <p>{removed_share:.2f}% of original data</p>
            </div>

            <div class="card">
                <div class="card-title">Retained Records</div>
                <pre>{self.n_retained:,}</pre>
                <p>{retained_share:.2f}% of original data</p>
            </div>

            <div class="card">
                <div class="card-title">Evaluation Method</div>
                <pre>{self.n_folds}-Fold CV</pre>
                <p>Same evaluation method for both datasets</p>
            </div>
        </div>
        """

    # -----------------------------------
    # Metric Cell
    # -----------------------------------

    def _metric_cell(
        self,
        fold_metrics: pd.DataFrame,
        metric: str,
    ) -> str:
        train_mean, train_std = self._mean_std(fold_metrics[f"Train {metric}"])

        test_mean, test_std = self._mean_std(fold_metrics[f"Test {metric}"])

        train_text = self._format_mean_std(
            train_mean,
            train_std,
        )

        test_text = self._format_mean_std(
            test_mean,
            test_std,
        )

        if np.isfinite(train_mean) and np.isfinite(test_mean):
            gap = train_mean - test_mean if metric == "R²" else test_mean - train_mean

            gap_text = f"{gap:.3f}"
        else:
            gap_text = "N/A"

        return f"""
        <td style="
            text-align:center;
            vertical-align:middle;
            white-space:nowrap;
            line-height:1.8;
            padding:12px 16px;
        ">
            <div>Train: {escape(train_text)}</div>
            <div>Test: {escape(test_text)}</div>
            <div>Gap: {escape(gap_text)}</div>
        </td>
        """

    # -----------------------------------
    # Main Comparison Table
    # -----------------------------------

    def _comparison_table_section(self) -> str:
        rows = []

        scenario_metrics = {
            scenario["name"]: scenario["fold_metrics"] for scenario in self.scenarios
        }

        for _, row in self.comparison.iterrows():
            scenario_name = str(row["Scenario"])
            fold_metrics = scenario_metrics[scenario_name]

            metric_cells = "".join(
                self._metric_cell(fold_metrics, metric)
                for metric in ["R²", "RMSE", "MAE"]
            )

            rows.append(f"""
                <tr>
                    <th>{escape(scenario_name)}</th>

                    <td>
                        {escape(str(row["Training Data"]))}
                    </td>

                    <td>
                        {escape(str(row["Evaluation Data"]))}
                    </td>

                    <td>
                        {int(row["Test Records"]):,}
                    </td>

                    {metric_cells}
                </tr>
            """)

        table_rows = "\n".join(rows)

        return f"""
        <section class="table-section">
            <h2>Model Performance Comparison</h2>

            <div class="table-wrapper">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Scenario</th>
                            <th>Training Data</th>
                            <th>Evaluation Data</th>
                            <th>Records</th>
                            <th>R²</th>
                            <th>RMSE</th>
                            <th>MAE</th>
                        </tr>
                    </thead>

                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </section>
        """

    # -----------------------------------
    # Scenario Plots
    # -----------------------------------

    def _scenario_plots_section(
        self,
        scenario: dict,
    ) -> str:
        y_true = np.asarray(
            scenario["y_true"],
            dtype=float,
        )

        y_pred = np.asarray(
            scenario["y_pred"],
            dtype=float,
        )

        actual_plot = self._actual_vs_predicted_plot(
            y_true,
            y_pred,
        )

        residual_plot = self._residuals_vs_predicted_plot(
            y_true,
            y_pred,
        )

        distribution_plot = self._residual_distribution_plot(
            y_true,
            y_pred,
        )

        scenario_name = escape(str(scenario["name"]))
        training_name = escape(str(scenario["training"]))
        evaluation_name = escape(str(scenario["evaluation"]))

        return f"""
        <section class="table-section">
            <h2>{scenario_name}</h2>

            <p>
                <strong>Dataset:</strong> {training_name}
                &nbsp; | &nbsp;
                <strong>Evaluation:</strong> {evaluation_name}
                &nbsp; | &nbsp;
                <strong>Records:</strong> {len(y_true):,}
            </p>

            <div
                class="grid"
                style="
                    display:grid;
                    grid-template-columns:repeat(3,minmax(0,1fr));
                    gap:16px;
                "
            >
                <div class="card" style="min-width:0;">
                    <div class="card-title">
                        Actual vs Predicted
                    </div>

                    <img
                        src="data:image/png;base64,{actual_plot}"
                        alt="Actual vs Predicted"
                        style="width:100%;height:auto;"
                    >
                </div>

                <div class="card" style="min-width:0;">
                    <div class="card-title">
                        Residuals vs Predicted
                    </div>

                    <img
                        src="data:image/png;base64,{residual_plot}"
                        alt="Residuals vs Predicted"
                        style="width:100%;height:auto;"
                    >
                </div>

                <div class="card" style="min-width:0;">
                    <div class="card-title">
                        Residual Distribution
                    </div>

                    <img
                        src="data:image/png;base64,{distribution_plot}"
                        alt="Residual Distribution"
                        style="width:100%;height:auto;"
                    >
                </div>
            </div>
        </section>
        """

    # -----------------------------------
    # Refinement Impact
    # -----------------------------------

    def _impact_section(self) -> str:
        scenario_metrics = {
            scenario["name"]: scenario["fold_metrics"] for scenario in self.scenarios
        }

        original = scenario_metrics.get("Original Model")
        refined = scenario_metrics.get("Refined Model")

        if original is None or refined is None:
            return ""

        changes = {}

        for metric in ["R²", "RMSE", "MAE"]:
            original_mean, _ = self._mean_std(original[f"Test {metric}"])

            refined_mean, _ = self._mean_std(refined[f"Test {metric}"])

            changes[metric] = self._format_change(refined_mean - original_mean)

        return f"""
        <section class="table-section">
            <h2>Refinement Impact</h2>

            <p>
                Changes are calculated as Refined Model
                minus Original Model, using mean test
                scores from 5-Fold Cross Validation.

                Higher R² and lower RMSE/MAE are preferable.
            </p>

            <div class="grid">
                <div class="card">
                    <div class="card-title">Test R² Change</div>
                    <pre>{changes["R²"]}</pre>
                </div>

                <div class="card">
                    <div class="card-title">Test RMSE Change</div>
                    <pre>{changes["RMSE"]}</pre>
                </div>

                <div class="card">
                    <div class="card-title">Test MAE Change</div>
                    <pre>{changes["MAE"]}</pre>
                </div>
            </div>
        </section>
        """

    # -----------------------------------
    # Evaluation Information
    # -----------------------------------

    def _evaluation_information_section(self) -> str:
        return f"""
        <section class="table-section">
            <h2>Evaluation Information</h2>

            <div class="card">
                <p>
                    <strong>Original Model:</strong>
                    The model is evaluated on the complete
                    original dataset using {self.n_folds}-Fold
                    Cross Validation, without removing records.
                </p>

                <p>
                    <strong>Outlier Detection:</strong>
                    Out-of-fold predictions are generated
                    on the original dataset.

                    Absolute prediction errors are analyzed
                    using Q3 + 1.5 × IQR.

                    Records exceeding this threshold
                    are removed from the refined dataset.
                </p>

                <p>
                    <strong>Refined Model:</strong>
                    The same model is evaluated again using
                    {self.n_folds}-Fold Cross Validation
                    on the dataset remaining after
                    high-error records are removed.
                </p>

                <p>
                    <strong>Train / Test Statistics:</strong>
                    Mean and standard deviation are calculated
                    across {self.n_folds} Cross Validation folds.

                    Each fold uses its own training
                    and validation observations.
                </p>

                <p>
                    <strong>Gap:</strong>
                    For R², Gap = Train − Test.

                    For RMSE and MAE, Gap = Test − Train.

                    A positive gap indicates worse
                    validation performance compared
                    with training performance.
                </p>

                <p>
                    <strong>Prediction Charts:</strong>
                    Both scenarios use out-of-fold
                    predictions from their respective
                    datasets.
                </p>

                <p>
                    <strong>Interpretation:</strong>
                    The refined dataset is selected using
                    prediction errors calculated on the
                    original dataset before the second
                    Cross Validation.

                    Therefore, refined results describe
                    performance on the filtered dataset
                    and are not an independent estimate
                    of generalization to unseen data.
                </p>
            </div>
        </section>
        """

    # -----------------------------------
    # Report Content
    # -----------------------------------

    def content(self) -> str:
        summary_section = self._summary_section()
        comparison_section = self._comparison_table_section()

        scenario_sections = "".join(
            self._scenario_plots_section(scenario) for scenario in self.scenarios
        )

        impact_section = self._impact_section()
        evaluation_section = self._evaluation_information_section()

        return f"""
        {summary_section}

        {comparison_section}

        {scenario_sections}

        {impact_section}

        {evaluation_section}
        """
