from html import escape

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from ..base_template import BaseTemplate


class PermutationImportanceTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
        importance_r2: pd.DataFrame,
        importance_rmse: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Permutation Importance",
        )

        self.results = results.copy()
        self.importance_r2 = importance_r2.copy()
        self.importance_rmse = importance_rmse.copy()

    # -----------------------------------
    # Results Table
    # -----------------------------------

    def _table_section(self) -> str:
        headers = "".join(
            f"<th>{escape(str(column))}</th>" for column in self.results.columns
        )

        rows = []

        for _, row in self.results.iterrows():
            cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)

            rows.append(f"<tr>{cells}</tr>")

        return f"""
        <section class="table-section">
            <h2>Permutation Importance Results</h2>

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
    # Report Content
    # -----------------------------------

    def content(self) -> str:
        return self._table_section()
