from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class CompareDiagnosticsTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Regression Diagnostics Comparison",
        )

        self.results = results
        self.full_width = True

    @staticmethod
    def _table(df: pd.DataFrame) -> str:
        html = df.to_html(
            classes="data-table",
            index=False,
            border=0,
            escape=True,
            na_rep="—",
            float_format=lambda value: f"{value:,.4f}",
        )

        return f'<div style="overflow-x:auto;">{html}</div>'

    @staticmethod
    def _section(title: str, description: str, body: str) -> str:
        return (
            '<section class="table-section">'
            f"<h2>{escape(title)}</h2>"
            f"<p>{escape(description)}</p>"
            f"{body}"
            "</section>"
        )

    def content(self) -> str:
        return self._section(
            "High-Error Comparison",
            "Models are sorted by the number of high-error records. "
            "High-error records have absolute out-of-fold prediction errors "
            "greater than 1.5 times the model's MAE.",
            self._table(self.results),
        )
