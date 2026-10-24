import base64
from html import escape
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class VisualizeDataTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        projections: dict,
        errors: dict | None = None,
        colors: pd.Series | None = None,
        color_by: str | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name, title="Clustering Data Visualization"
        )
        self.projections = projections
        self.errors = errors or {}
        self.colors = colors
        self.color_by = color_by

    def _plot(self, points: np.ndarray) -> str:
        fig, ax = plt.subplots(figsize=(7, 5))
        if self.colors is None:
            ax.scatter(
                points[:, 0], points[:, 1], s=13, alpha=0.7, c="#0284c7", linewidths=0
            )
        else:
            series = self.colors.reset_index(drop=True)
            numeric = (
                pd.api.types.is_numeric_dtype(series)
                and series.nunique(dropna=True) > 12
            )
            if numeric:
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(values)
                plot = ax.scatter(
                    points[valid, 0],
                    points[valid, 1],
                    c=values[valid],
                    cmap="viridis",
                    s=15,
                    alpha=0.8,
                    linewidths=0,
                )
                if valid.any():
                    fig.colorbar(plot, ax=ax, label=self.color_by, shrink=0.8)
                if (~valid).any():
                    ax.scatter(
                        points[~valid, 0],
                        points[~valid, 1],
                        s=15,
                        c="#94a3b8",
                        label="Missing",
                        alpha=0.7,
                    )
                    ax.legend(loc="best", fontsize=8)
            else:
                categories = series.astype("string").fillna("Missing")
                unique = list(pd.unique(categories))
                palette = plt.get_cmap("tab20", max(len(unique), 1))
                for i, category in enumerate(unique):
                    mask = (categories == category).to_numpy(dtype=bool)
                    ax.scatter(
                        points[mask, 0],
                        points[mask, 1],
                        s=15,
                        alpha=0.75,
                        linewidths=0,
                        color=palette(i),
                        label=str(category),
                    )
                ax.legend(
                    title=self.color_by,
                    loc="best",
                    fontsize=7,
                    title_fontsize=8,
                    ncol=2 if len(unique) > 8 else 1,
                )
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(
            buffer, format="png", dpi=140, bbox_inches="tight", facecolor="white"
        )
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def content(self):
        charts = "".join(
            f'<section class="table-section" style="margin:0;min-width:0;">'
            f"<h2>{escape(name)}</h2>"
            f'<div style="padding:16px;">'
            f'<img alt="{escape(name)}" src="data:image/png;base64,{self._plot(points)}" '
            f'style="display:block;width:100%;height:auto;background:white;border-radius:8px;">'
            f"</div></section>"
            for name, points in self.projections.items()
        )
        skipped = "".join(
            f"<p><strong>{escape(name)}:</strong> {escape(reason)}</p>"
            for name, reason in self.errors.items()
        )
        return f"""
        <style>
            .visualization-grid {{
                display:grid;
                grid-template-columns:repeat(2,minmax(0,1fr));
                gap:20px;
            }}
            @media(max-width:900px) {{
                .visualization-grid {{grid-template-columns:1fr;}}
            }}
        </style>
        <div class="visualization-grid">{charts}</div>
        {f'<section class="table-section"><h2>Unavailable Visualizations</h2><div style="padding:16px">{skipped}</div></section>' if skipped else ''}
        """
