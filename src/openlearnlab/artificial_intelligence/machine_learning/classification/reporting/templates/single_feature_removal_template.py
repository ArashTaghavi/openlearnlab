from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class SingleFeatureRemovalTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Classification - Single Feature Removal Analysis",
        )

        self.results = results.copy()
        self.full_width = True

    def content(self) -> str:
        headers = "".join(
            f"<th>{escape(str(column))}</th>" for column in self.results.columns
        )

        rows = []

        for index, row in self.results.iterrows():
            cells = "".join(
                f'<td style="white-space: pre-line;">' f"{escape(str(value))}" "</td>"
                for value in row
            )

            rows.append("<tr>" f"<th>{escape(str(index))}</th>" f"{cells}" "</tr>")

        table_rows = "\n".join(rows)

        return f"""
        <section class="table-section">
            <h2>Classification - Single Feature Removal Analysis</h2>

            <p>
                Comparison of classification performance using all features
                versus removing each feature individually.
                Results are based on 5-fold stratified cross-validation.
            </p>

            <div class="table-wrapper">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Feature Removal</th>
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
