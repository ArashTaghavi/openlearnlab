from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class RegularizationComparisonTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Regularization Comparison",
            full_width=True,
        )

        self.results = results.copy()

    # -----------------------------------
    # Results Table
    # -----------------------------------

    def content(self) -> str:
        headers = "".join(
            f"<th>{escape(str(column))}</th>" for column in self.results.columns
        )

        rows = []

        for _, row in self.results.iterrows():

            cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)

            rows.append(f"<tr>{cells}</tr>")

        return f"""
        <section class="table-section">
            <h2>Regularization Results</h2>

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
