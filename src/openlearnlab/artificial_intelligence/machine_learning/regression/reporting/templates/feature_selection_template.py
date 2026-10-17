import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ..base_template import BaseTemplate


class FeatureSelectionTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
        summary: dict,
        performance: pd.DataFrame,
        selected_count: int,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="RFECV Feature Selection",
        )

        self.results = results.copy()
        self.summary = summary.copy()
        self.performance = performance.copy()
        self.selected_count = selected_count

    # -----------------------------------
    # Summary Section
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
            <h2>Feature Selection Summary</h2>

            <div class="grid">
                {"".join(cards)}
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
            cells = []

            for column, value in row.items():
                if column == "Selected":
                    selected = bool(value)

                    label = "Selected" if selected else "Removed"

                    color = "#22c55e" if selected else "#f87171"

                    cells.append(f"""
                        <td>
                            <span style="
                                color:{color};
                                font-weight:bold;
                            ">
                                {label}
                            </span>
                        </td>
                        """)
                else:
                    cells.append(f"<td>{escape(str(value))}</td>")

            rows.append(f"<tr>{''.join(cells)}</tr>")

        return f"""
        <section class="table-section">
            <h2>Feature Selection Results</h2>

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

            <p style="
                color:#94a3b8;
                font-size:12px;
                margin-top:12px;
            ">
                Ranking = 1 indicates a selected feature.
                Rankings do not represent relative
                importance among selected features.
            </p>
        </section>
        """

    # -----------------------------------
    # Performance Chart
    # -----------------------------------

    def _plot(self) -> str:
        x = self.performance["Number of Features"].to_numpy(dtype=int)

        mean = self.performance["Mean CV Score"].to_numpy(dtype=float)

        std = self.performance["Std CV Score"].to_numpy(dtype=float)

        fig, ax = plt.subplots(
            figsize=(10, 5),
        )

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label="Mean CV Score",
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2,
            label="± 1 Std",
        )

        ax.axvline(
            x=self.selected_count,
            linestyle="--",
            linewidth=2,
            label=(f"Selected Features: " f"{self.selected_count}"),
        )

        ax.set_title("RFECV Performance by Number of Features")

        ax.set_xlabel("Number of Features")
        ax.set_ylabel(f"Cross Validation Score ({self.summary['Scoring']})")

        ax.set_xticks(x)

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
            <h2>RFECV Performance Chart</h2>

            <div class="card">
                <img
                    src="data:image/png;base64,{chart}"
                    alt="RFECV Performance"
                    style="
                        display:block;
                        width:100%;
                        max-width:1000px;
                        height:auto;
                        margin:0 auto;
                    "
                >
            </div>

            <p style="
                color:#94a3b8;
                font-size:12px;
                margin-top:12px;
            ">
                Scores are from the cross-validation
                process used for feature selection.
                They are not an independent estimate
                of final model generalization performance.
            </p>
        </section>
        """

    # -----------------------------------
    # Report Content
    # -----------------------------------

    def content(self) -> str:
        return self._summary_section() + self._table_section() + self._chart_section()
