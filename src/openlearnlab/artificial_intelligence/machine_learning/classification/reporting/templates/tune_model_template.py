from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class TuneModelTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
        best_params: dict,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Base vs Tuned Model",
            full_width=True,
        )

        self.results = results.copy()
        self.best_params = best_params.copy()

    # -----------------------------------
    # Model Comparison
    # -----------------------------------

    def _comparison_section(self) -> str:
        headers = "<th>Model</th>" + "".join(
            f"<th>{escape(str(column))}</th>" for column in self.results.columns
        )

        rows = []

        for index, row in self.results.iterrows():

            cells = f"<th>{escape(str(index))}</th>" + "".join(
                "<td style='white-space:pre-line;'>" f"{escape(str(value))}" "</td>"
                for value in row
            )

            rows.append(f"<tr>{cells}</tr>")

        return f"""
        <section class="table-section">
            <h2>Base vs Tuned Model</h2>

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
    # Best Parameters
    # -----------------------------------

    def _parameters_section(self) -> str:
        rows = []

        for parameter, value in self.best_params.items():
            rows.append(f"""
                <tr>
                    <th>{escape(str(parameter))}</th>
                    <td>{escape(str(value))}</td>
                </tr>
                """)

        return f"""
        <section class="table-section">
            <h2>Best Parameters</h2>

            <div class="table-wrapper">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Best Value</th>
                        </tr>
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
        return self._comparison_section() + self._parameters_section()
