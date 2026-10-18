from html import escape

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class CompareModelsTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        plot_data: pd.DataFrame,
        best_model: str,
        n_samples: int,
        n_features: int,
        n_folds: int,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Comparing Classification Models",
            full_width=True,
        )

        self.df = df.copy()
        self.plot_data = plot_data.copy()
        self.best_model = best_model
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_folds = n_folds

    def content(self) -> str:
        # -----------------------------------
        # Model Comparison Table
        # -----------------------------------

        rows = []

        for model_name, row in self.df.iterrows():
            is_best = str(model_name) == self.best_model

            badge = (
                '<span style="'
                "color:#34d399;"
                "font-size:11px;"
                "font-weight:600;"
                "margin-left:8px;"
                '">BEST</span>'
                if is_best
                else ""
            )

            cells = "".join(
                f"<td style='white-space:pre-line;'>" f"{escape(str(value))}</td>"
                for value in row
            )

            row_style = ' style="background:rgba(52,211,153,0.08);"' if is_best else ""

            rows.append(
                f"<tr{row_style}>"
                f"<th>{escape(str(model_name))}{badge}</th>"
                f"{cells}"
                f"</tr>"
            )

        headers = "".join(
            f"<th>{escape(str(column))}</th>" for column in self.df.columns
        )

        table_rows = "\n".join(rows)

        comparison_table = f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Model</th>
                    {headers}
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """

        # -----------------------------------
        # Summary
        # -----------------------------------

        best_model = escape(self.best_model)
        models_count = len(self.df)

        dataset_info = f"{self.n_samples:,} rows / " f"{self.n_features} features"

        # -----------------------------------
        # HTML Content
        # -----------------------------------

        return f"""
        <div class="grid">
            <div class="card">
                <div class="card-title" style="color:#38bdf8;">
                    Best Model (Test F1)
                </div>
                <pre>{best_model}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#a78bfa;">
                    Models Evaluated
                </div>
                <pre>{models_count}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#fbbf24;">
                    Dataset
                </div>
                <pre>{dataset_info}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#fb7185;">
                    Cross Validation
                </div>
                <pre>{self.n_folds} Folds</pre>
            </div>
        </div>

        <section class="table-section">
            <h2>Model Comparison Results</h2>

            <div class="table-wrapper">
                {comparison_table}
            </div>
        </section>

        <section class="table-section">
            <h2>Evaluation Information</h2>

            <div style="padding:20px;">
                <p>
                    <strong>Ranking:</strong>
                    Models are ranked by mean Test F1
                    (descending), then mean Test ROC AUC
                    (descending), then mean Predict Time
                    (ascending), and finally mean Fit Time
                    (ascending).
                </p>

                <p>
                    <strong>Relative Loss:</strong>
                    Test Accuracy, Precision, Recall, F1,
                    and ROC AUC losses are calculated
                    relative to the top-ranked model.
                    Negative losses indicate that a model
                    outperforms the top-ranked model
                    on that individual metric.
                </p>

                <p>
                    <strong>Speed Comparison:</strong>
                    Fit and prediction speeds are compared
                    against the top-ranked model.
                    Positive ratios indicate faster performance,
                    while negative ratios indicate slower performance.
                </p>

                <p>
                    <strong>Cross Validation:</strong>
                    {self.n_folds}-Fold Stratified Cross Validation.
                </p>
            </div>
        </section>
        """
