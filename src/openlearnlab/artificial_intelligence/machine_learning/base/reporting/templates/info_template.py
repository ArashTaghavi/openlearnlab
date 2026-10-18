from html import escape

import pandas as pd

from ..base_template import BaseTemplate


class InfoTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        shape: tuple,
        duplicated: int,
        null_counts: dict,
        columns: list,
        invalid_data: dict,
        df_info: pd.DataFrame,
        df_describe: pd.DataFrame,
        df_head: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Dataset Information",
        )

        self.shape = shape
        self.duplicated = duplicated
        self.null_counts = null_counts
        self.columns = columns
        self.invalid_data = invalid_data
        self.df_info = df_info
        self.df_describe = df_describe
        self.df_head = df_head

    def content(self) -> str:
        null_counts = self.null_counts or "None"
        invalid_data = self.invalid_data or "None"

        info_table = self.df_info.to_html(
            index=False,
            classes="data-table",
            border=0,
            escape=True,
        )

        if self.df_describe.empty:
            describe_table = "<p>No descriptive statistics available.</p>"
        else:
            describe_table = self.df_describe.round(3).to_html(
                classes="data-table",
                border=0,
                escape=True,
                na_rep="-",
            )

        head_table = self.df_head.to_html(
            index=False,
            classes="data-table",
            border=0,
            escape=True,
            na_rep="-",
        )

        return f"""
        <div class="grid">
            <div class="card">
                <div class="card-title" style="color:#38bdf8;">
                    Shape
                </div>
                <pre>{escape(str(self.shape))}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#a78bfa;">
                    Duplicated
                </div>
                <pre>{self.duplicated}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#34d399;">
                    Null Counts
                </div>
                <pre>{escape(str(null_counts))}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#fbbf24;">
                    Columns
                </div>
                <pre>{escape(str(self.columns))}</pre>
            </div>

            <div class="card card-full" style="grid-column:1 / -1;">
                <div class="card-title" style="color:#fb7185;">
                    Invalid Data
                </div>
                <pre>{escape(str(invalid_data))}</pre>
            </div>
        </div>

        <section class="table-section">
            <h2>Dataset Preview (First 5 Rows)</h2>
            <div class="table-wrapper">
                {head_table}
            </div>
        </section>

        <section class="table-section">
            <h2>DataFrame Info</h2>
            <div class="table-wrapper">
                {info_table}
            </div>
        </section>

        <section class="table-section">
            <h2>Descriptive Statistics</h2>
            <div class="table-wrapper">
                {describe_table}
            </div>
        </section>
        """
