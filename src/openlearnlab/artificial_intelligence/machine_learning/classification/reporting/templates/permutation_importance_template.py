from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class PermutationImportanceTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
        importance_accuracy: pd.DataFrame,
        importance_precision: pd.DataFrame,
        importance_recall: pd.DataFrame,
        importance_f1: pd.DataFrame,
        importance_roc_auc: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Classification - Permutation Importance",
            full_width=True,
        )

        self.results = results.copy()
        self.importance_accuracy = importance_accuracy.copy()
        self.importance_precision = importance_precision.copy()
        self.importance_recall = importance_recall.copy()
        self.importance_f1 = importance_f1.copy()
        self.importance_roc_auc = importance_roc_auc.copy()

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

        table_rows = "\n".join(rows)

        return f"""
        <section class="table-section">
            <h2>Classification - Permutation Importance Results</h2>

            <p>
                Feature importance is measured by the decrease in
                classification performance after randomly permuting
                each feature on the test dataset.
                Results are reported as mean ± standard deviation
                across 30 permutations.
            </p>

            <div class="table-wrapper">
                <table class="data-table">
                    <thead>
                        <tr>{headers}</tr>
                    </thead>

                    <tbody>
                        {table_rows}
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
