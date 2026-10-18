import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class FeatureExtractionTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        summary: dict,
        info: pd.DataFrame,
        loadings: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="PCA Feature Extraction",
            full_width=True,
        )

        self.summary = summary.copy()
        self.info = info.copy()
        self.loadings = loadings.copy()

    # -----------------------------------
    # Summary
    # -----------------------------------

    def _summary_section(self) -> str:
        cards = []

        for key, value in self.summary.items():
            cards.append(f"""
                <div class="card">
                    <div class="card-title">
                        {escape(str(key))}
                    </div>

                    <pre>{escape(str(value))}</pre>
                </div>
                """)

        return f"""
        <section class="table-section">
            <h2>PCA Summary</h2>

            <div class="grid">
                {"".join(cards)}
            </div>
        </section>
        """

    # -----------------------------------
    # Explained Variance Table
    # -----------------------------------

    def _info_section(self) -> str:
        headers = "".join(
            f"<th>{escape(str(column))}</th>" for column in self.info.columns
        )

        rows = []

        for _, row in self.info.iterrows():
            cells = "".join(
                f"<td>{escape(f'{float(value):.4f}' if isinstance(value, (float, np.floating)) else str(value))}</td>"
                for value in row
            )

            rows.append(f"<tr>{cells}</tr>")

        return f"""
        <section class="table-section">
            <h2>Explained Variance</h2>

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
    # Feature Loadings Table
    # -----------------------------------

    def _loadings_section(self) -> str:
        headers = "<th>Feature</th>" + "".join(
            f"<th>{escape(str(column))}</th>" for column in self.loadings.columns
        )

        rows = []

        for feature, row in self.loadings.iterrows():
            cells = f"<th>{escape(str(feature))}</th>" + "".join(
                f"<td>{float(value):.4f}</td>" for value in row
            )

            rows.append(f"<tr>{cells}</tr>")

        return f"""
        <section class="table-section">
            <h2>Feature Loadings</h2>

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
    # Cumulative Variance Chart
    # -----------------------------------

    def _plot(self) -> str:
        x = np.arange(
            1,
            len(self.info) + 1,
        )

        cumulative = self.info["Cumulative Variance"].to_numpy(dtype=float)

        fig, ax = plt.subplots(
            figsize=(10, 5),
        )

        ax.plot(
            x,
            cumulative,
            marker="o",
            linewidth=2,
        )

        ax.axhline(
            y=float(self.summary["Variance Threshold"].strip("%")) / 100,
            linestyle="--",
            label="Variance Threshold",
        )

        ax.set_title("Cumulative Explained Variance")

        ax.set_xlabel("Number of Principal Components")

        ax.set_ylabel("Cumulative Explained Variance")

        ax.set_xticks(x)

        ax.set_ylim(0, 1.05)

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend()

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
            <h2>Cumulative Explained Variance</h2>

            <div class="card">
                <img
                    src="data:image/png;base64,{chart}"
                    alt="Cumulative Explained Variance"
                    style="
                        display:block;
                        width:100%;
                        max-width:1000px;
                        height:auto;
                        margin:0 auto;
                    "
                >
            </div>
        </section>
        """

    # -----------------------------------
    # Report Content
    # -----------------------------------

    def content(self) -> str:
        return (
            self._summary_section()
            + self._info_section()
            + self._chart_section()
            + self._loadings_section()
        )
