import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..base_template import BaseTemplate


class CorrelationTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        correlation_matrix: pd.DataFrame,
        high_correlation_pairs: pd.DataFrame,
        suspicious_features: set[str],
        threshold: float,
        method: str,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Correlation Analysis",
        )

        self.correlation_matrix = correlation_matrix
        self.high_correlation_pairs = high_correlation_pairs
        self.suspicious_features = suspicious_features
        self.threshold = threshold
        self.method = method

    def _heatmap(self) -> str:
        corr = self.correlation_matrix

        size = max(6, len(corr.columns) * 0.75)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))

        mask = pd.DataFrame(
            False,
            index=corr.index,
            columns=corr.columns,
        )

        for i in range(len(corr)):
            for j in range(i + 1, len(corr.columns)):
                mask.iloc[i, j] = True

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Correlation"},
            ax=ax,
        )

        ax.set_title(f"{self.method.title()} Correlation Matrix")
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def content(self) -> str:
        heatmap_image = self._heatmap()

        pairs_table = (
            self.high_correlation_pairs.to_html(
                index=False,
                classes="data-table",
                border=0,
                escape=True,
            )
            if not self.high_correlation_pairs.empty
            else "<p style='padding:0 15px'>No high correlation pairs found.</p>"
        )

        matrix_table = self.correlation_matrix.round(3).to_html(
            classes="data-table",
            border=0,
            escape=True,
        )

        numeric_features = ", ".join(
            str(column) for column in self.correlation_matrix.columns
        )

        suspicious = (
            ", ".join(sorted(self.suspicious_features))
            if self.suspicious_features
            else "None"
        )

        return f"""
        <div class="grid">
            <div class="card">
                <div class="card-title" style="color:#38bdf8;">
                    Numeric Features
                </div>
                <pre>{escape(numeric_features)}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#a78bfa;">
                    Correlation Method
                </div>
                <pre>{escape(self.method.title())}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#fbbf24;">
                    Correlation Threshold
                </div>
                <pre>{self.threshold:.2f}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#fb7185;">
                    Suspicious Features
                </div>
                <pre>{escape(suspicious)}</pre>
            </div>
        </div>

        <section class="table-section">
            <h2>Correlation Heatmap</h2>
            <div class="table-wrapper" style="
                padding:20px;
                background:#ffffff;
                text-align:center;
            ">
                <img
                    src="data:image/png;base64,{heatmap_image}"
                    alt="Correlation Heatmap"
                    style="max-width:100%; height:auto;"
                >
            </div>
        </section>

        <section class="table-section">
            <h2>High Correlation Pairs</h2>
            <div class="table-wrapper">
                {pairs_table}
            </div>
        </section>

        <section class="table-section">
            <h2>Correlation Matrix</h2>
            <div class="table-wrapper">
                {matrix_table}
            </div>
        </section>
        """
