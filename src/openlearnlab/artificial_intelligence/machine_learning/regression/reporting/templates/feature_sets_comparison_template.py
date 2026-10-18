from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class FeatureSetsComparisonTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Feature Sets Comparison",
            full_width=True,
        )

        self.results = results.copy()

    def content(self) -> str:
        headers = "<th>Feature Set</th>" + "".join(
            f"<th>{escape(str(column))}</th>" for column in self.results.columns
        )

        rows = []

        for index, row in self.results.iterrows():

            cells = f"<th>{escape(str(index))}</th>" + "".join(
                "<td style='white-space:pre-line;'>" f"{escape(str(value))}" "</td>"
                for value in row
            )

            rows.append(f"<tr>{cells}</tr>")

        table_rows = "\n".join(rows)

        return f"""
        <section class="table-section">
            <h2>Feature Sets Comparison</h2>

            <div class="table-wrapper">
                <table class="data-table">
                    <thead>
                        <tr>
                            {headers}
                        </tr>
                    </thead>

                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </section>
        """
