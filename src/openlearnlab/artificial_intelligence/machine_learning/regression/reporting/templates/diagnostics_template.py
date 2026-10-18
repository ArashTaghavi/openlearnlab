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
        insights: list[tuple[str, str]],
        n_samples: int,
        n_features: int,
        n_folds: int,
        critical_errors: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Regression Diagnostics",
        )

        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = residuals

        self.r2 = r2
        self.rmse = rmse
        self.mae = mae

        self.insights = insights

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_folds = n_folds

        self.critical_errors = critical_errors

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

    def _actual_vs_predicted_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            self.y_true,
            self.y_pred,
            alpha=0.65,
            s=25,
        )

        minimum = float(
            min(
                np.min(self.y_true),
                np.min(self.y_pred),
            )
        )

        maximum = float(
            max(
                np.max(self.y_true),
                np.max(self.y_pred),
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

    def _residuals_vs_predicted_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            self.y_pred,
            self.residuals,
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

    def _residual_distribution_plot(self) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(
            self.residuals,
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

    def _summary_section(self) -> str:
        n_critical = len(self.critical_errors)

        dataset_share = 100 * n_critical / self.n_samples if self.n_samples > 0 else 0.0

        sse_share = (
            float(self.critical_errors["SSE Share (%)"].sum())
            if n_critical > 0
            else 0.0
        )

        return f"""
        <div class="grid">
            <div class="card">
                <div class="card-title">R² Score</div>
                <pre>{self.r2:.4f}</pre>
            </div>

            <div class="card">
                <div class="card-title">RMSE</div>
                <pre>{self.rmse:.4f}</pre>
            </div>

            <div class="card">
                <div class="card-title">MAE</div>
                <pre>{self.mae:.4f}</pre>
            </div>

            <div class="card">
                <div class="card-title">Observations</div>
                <pre>{self.n_samples:,}</pre>
            </div>

            <div class="card">
                <div class="card-title">Dataset Share</div>
                <pre>{dataset_share:.2f}%</pre>
            </div>

            <div class="card">
                <div class="card-title">Total SSE Share</div>
                <pre>{sse_share:.2f}%</pre>
            </div>

            <div class="card">
                <div class="card-title">Evaluation Method</div>
                <pre>{self.n_folds}-Fold CV</pre>
            </div>
        </div>
        """

    def _high_error_observations(self) -> str:
        if self.critical_errors.empty:
            table_html = """
            <div class="card">
                <p>
                    No unusually high prediction errors were detected.
                </p>
            </div>
            """
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

            table_html = f"""
            <div style="overflow-x:auto;">
                {table_html}
            </div>
            """

        return f"""
        <section class="table-section">
            <h2>High Error Observations</h2>

            <p>
                The following observations have unusually large
                out-of-fold prediction errors and may require
                further investigation.
            </p>

            {table_html}

            <p>
                These observations have unusually high prediction
                errors, but they are not necessarily incorrect data.
            </p>
        </section>
        """

    def _insights_section(self) -> str:
        insight_cards = []

        for title, description in self.insights:
            insight_cards.append(f"""
                <div class="card">
                    <div class="card-title">{escape(title)}</div>
                    <p>{escape(description)}</p>
                </div>
                """)

        return f"""
        <section class="table-section">
            <h2>Diagnostic Insights</h2>

            <div class="grid">
                {''.join(insight_cards)}
            </div>
        </section>
        """

    def _plots_section(self) -> str:
        actual_plot = self._actual_vs_predicted_plot()
        residual_plot = self._residuals_vs_predicted_plot()
        distribution_plot = self._residual_distribution_plot()

        return f"""
        <section class="table-section">
            <h2>Prediction Diagnostics</h2>

            <div class="grid">
                <div class="card">
                    <div class="card-title">Actual vs Predicted</div>
                    <img
                        src="data:image/png;base64,{actual_plot}"
                        alt="Actual vs Predicted"
                        style="width:100%;height:auto;"
                    >
                </div>

                <div class="card">
                    <div class="card-title">Residuals vs Predicted</div>
                    <img
                        src="data:image/png;base64,{residual_plot}"
                        alt="Residuals vs Predicted"
                        style="width:100%;height:auto;"
                    >
                </div>

                <div class="card">
                    <div class="card-title">Residual Distribution</div>
                    <img
                        src="data:image/png;base64,{distribution_plot}"
                        alt="Residual Distribution"
                        style="width:100%;height:auto;"
                    >
                </div>
            </div>
        </section>
        """

    def content(self) -> str:
        summary_section = self._summary_section()
        high_error_section = self._high_error_observations()
        plots_section = self._plots_section()
        insights_section = self._insights_section()

        return f"""
        {summary_section}

        {high_error_section}

        {plots_section}

        {insights_section}
        """
