from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class CompareModelsTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
        best_model: str,
        n_samples: int,
        n_features: int,
        max_noise_ratio: float,
        silhouette_sample_size: int | None,
        chart: str | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Clustering Model Comparison",
        )

        self.results = results
        self.best_model = best_model
        self.n_samples = n_samples
        self.n_features = n_features
        self.max_noise_ratio = max_noise_ratio
        self.silhouette_sample_size = silhouette_sample_size
        self.chart = chart

    def content(self) -> str:
        # -----------------------------------
        # Model Comparison Table
        # -----------------------------------

        rows = []

        for model_name, row in self.results.iterrows():
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
                ('<td style="white-space:pre-line;">' f"{escape(str(value))}" "</td>")
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
            f"<th>{escape(str(column))}</th>" for column in self.results.columns
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

        best_model = escape(str(self.best_model))

        models_count = len(self.results)

        dataset_info = f"{self.n_samples:,} rows / " f"{self.n_features} features"

        # -----------------------------------
        # Fit Time Chart
        # -----------------------------------

        chart_html = ""

        if self.chart:
            chart_html = f"""
            <section class="table-section">
                <h2>Clustering Fit Time Comparison</h2>

                <div style="padding:20px;">
                    <img
                        src="data:image/png;base64,{self.chart}"
                        alt="Clustering Fit Time Comparison"
                        style="width:100%;height:auto;"
                    >
                </div>
            </section>
            """

        # -----------------------------------
        # Silhouette Sampling
        # -----------------------------------

        sampling_info = (
            "Full dataset"
            if self.silhouette_sample_size is None
            else f"Up to {self.silhouette_sample_size:,} samples"
        )

        # -----------------------------------
        # HTML Content
        # -----------------------------------

        return f"""
        <div class="grid">
            <div class="card">
                <div class="card-title" style="color:#38bdf8;">
                    Best Model (Silhouette)
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
                    Evaluation
                </div>
                <pre>Unsupervised</pre>
            </div>
        </div>

        <section class="table-section">
            <h2>Model Comparison Results</h2>

            <div class="table-wrapper">
                {comparison_table}
            </div>
        </section>

        {chart_html}

        <section class="table-section">
            <h2>Evaluation Information</h2>

            <div style="padding:20px;">
                <p>
                    <strong>Ranking:</strong>
                    Models are ranked by eligibility, followed by
                    Silhouette Score (descending), Davies-Bouldin
                    Index (ascending), Calinski-Harabasz Index
                    (descending), and Fit Time (ascending).
                </p>

                <p>
                    <strong>Evaluation Metrics:</strong>
                    Models are evaluated using Silhouette Score,
                    Davies-Bouldin Index, and Calinski-Harabasz
                    Index. Higher Silhouette and Calinski-Harabasz
                    scores indicate better clustering, while lower
                    Davies-Bouldin scores are preferred.
                </p>

                <p>
                    <strong>Clusters:</strong>
                    The number of clusters identified by each
                    algorithm, excluding noise samples.
                </p>

                <p>
                    <strong>Noise:</strong>
                    Noise Count represents the number of samples
                    assigned label -1. Noise (%) represents their
                    percentage of the dataset. Noise samples are
                    excluded from clustering metric calculations.
                </p>

                <p>
                    <strong>Eligibility:</strong>
                    Models must have a valid Silhouette Score and
                    a noise percentage not exceeding
                    {self.max_noise_ratio:.2f}% to be eligible
                    for best model selection.
                </p>

                <p>
                    <strong>Relative Loss:</strong>
                    Silhouette, Davies-Bouldin, and
                    Calinski-Harabasz losses are calculated
                    relative to the top-ranked eligible model.
                    Negative loss indicates an improvement.
                </p>

                <p>
                    <strong>Speed Comparison:</strong>
                    Fit speeds are compared against the
                    top-ranked eligible model. Positive values
                    above 1.00x indicate faster execution,
                    while negative values indicate slower
                    execution. A value of 1.00x indicates
                    equal speed.
                </p>

                <p>
                    <strong>Silhouette Sampling:</strong>
                    {sampling_info}. Davies-Bouldin and
                    Calinski-Harabasz are calculated using
                    all non-noise samples.
                </p>

                <p>
                    <strong>Evaluation Method:</strong>
                    Clustering models are evaluated on the
                    fitted dataset without cross-validation.
                    Metrics are calculated in the feature space
                    used by each clustering algorithm.
                </p>
            </div>
        </section>
        """
