import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..base_template import BaseTemplate


class LearningCurveTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
        sizes: np.ndarray,
        train_mean: np.ndarray,
        train_std: np.ndarray,
        test_mean: np.ndarray,
        test_std: np.ndarray,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Learning Curve",
        )

        self.results = results.copy()
        self.sizes = np.asarray(sizes)
        self.train_mean = np.asarray(train_mean)
        self.train_std = np.asarray(train_std)
        self.test_mean = np.asarray(test_mean)
        self.test_std = np.asarray(test_std)

    # -----------------------------------
    # Learning Curve Chart
    # -----------------------------------

    def _plot(self) -> str:
        fig, ax = plt.subplots(
            figsize=(9, 5),
        )

        ax.plot(
            self.sizes,
            self.train_mean,
            marker="o",
            label="Train",
        )

        ax.plot(
            self.sizes,
            self.test_mean,
            marker="o",
            label="Validation",
        )

        ax.fill_between(
            self.sizes,
            self.train_mean - self.train_std,
            self.train_mean + self.train_std,
            alpha=0.15,
        )

        ax.fill_between(
            self.sizes,
            self.test_mean - self.test_std,
            self.test_mean + self.test_std,
            alpha=0.15,
        )

        ax.set_xlabel("Training Samples")
        ax.set_ylabel("R2")
        ax.set_title("Learning Curve")

        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        buffer = BytesIO()

        try:
            fig.savefig(
                buffer,
                format="png",
                dpi=150,
                bbox_inches="tight",
            )

            buffer.seek(0)

            return base64.b64encode(buffer.read()).decode("utf-8")

        finally:
            plt.close(fig)
            buffer.close()

    # -----------------------------------
    # Chart Section
    # -----------------------------------

    def _chart_section(self) -> str:
        chart = self._plot()

        return f"""
        <section class="table-section">
            <h2>Learning Curve Chart</h2>

            <div class="card">
                <img
                    src="data:image/png;base64,{chart}"
                    alt="Learning Curve"
                    style="
                        display:block;
                        width:100%;
                        height:auto;
                    "
                >
            </div>
        </section>
        """

    # -----------------------------------
    # Results Table
    # -----------------------------------

    def _table_section(self) -> str:
        headers = "".join(
            f"<th>{escape(str(column))}</th>" for column in self.results.columns
        )

        rows = []

        for _, row in self.results.iterrows():
            cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)

            rows.append(f"<tr>{cells}</tr>")

        return f"""
        <section class="table-section">
            <h2>Learning Curve Results</h2>

            <div class="table-wrapper">
                <table class="data-table">
                    <thead>
                        <tr>{headers}</tr>
                    </thead>

                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
        </section>
        """

    # -----------------------------------
    # Report Content
    # -----------------------------------

    def content(self) -> str:
        return self._chart_section() + self._table_section()
