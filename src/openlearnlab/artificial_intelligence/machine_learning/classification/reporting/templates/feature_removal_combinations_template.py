from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class FeatureRemovalCombinationsTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Classification - Feature Removal Combinations Analysis",
            full_width=True,
        )

        self.results = results.copy()

    def content(self) -> str:
        headers = "".join(
            f"<th>{escape(str(column))}</th>" for column in self.results.columns
        )

        rows = []

        for _, row in self.results.iterrows():
            cells = "".join(
                f'<td style="white-space: pre-line;">' f"{escape(str(value))}" "</td>"
                for value in row
            )

            rows.append(f"<tr>{cells}</tr>")

        table_rows = "\n".join(rows)

        return f"""
        <section class="table-section">
            <h2>Classification - Feature Removal Combinations Analysis</h2>

            <p>
                Comparison of classification performance across different
                feature removal combinations using 5-fold stratified
                cross-validation.
            </p>

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
