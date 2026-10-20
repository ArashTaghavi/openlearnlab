from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class FeatureExtractionComparisonTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
        n_samples: int,
        n_features: int,
        variance_threshold: float,
        n_folds: int,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Classification - Feature Extraction Comparison",
        )

        self.results = results
        self.n_samples = n_samples
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.n_folds = n_folds

        self.full_width = True

    def content(self) -> str:
        # -----------------------------------
        # Feature Extraction Comparison Table
        # -----------------------------------

        columns = [
            column
            for column in self.results.columns
            if column
            not in (
                "Model",
                "Dataset Version",
                "Features",
            )
        ]

        headers = "".join(f"<th>{escape(str(column))}</th>" for column in columns)

        column_count = len(columns) + 1

        model_groups = []

        for model_name, model_df in self.results.groupby(
            "Model",
            sort=False,
        ):
            # -----------------------------------
            # Model Group Header
            # -----------------------------------

            model_header = f"""
            <tr class="feature-model-header">
                <th colspan="{column_count}">
                    <span class="feature-model-name">
                        {escape(str(model_name))}
                    </span>
                </th>
            </tr>
            """

            version_rows = []

            for _, row in model_df.iterrows():
                version = str(row["Dataset Version"])
                is_original = version == "Original"

                version_class = (
                    "feature-original-row" if is_original else "feature-pca-row"
                )

                version_label = "ORIGINAL" if is_original else "PCA"

                badge_class = (
                    "feature-original-badge" if is_original else "feature-pca-badge"
                )

                # -----------------------------------
                # Table Cells
                # -----------------------------------

                cells = "".join(
                    (
                        '<td style="white-space:pre-line;">'
                        f"{escape(str(row[column]))}"
                        "</td>"
                    )
                    for column in columns
                )

                version_rows.append(f"""
                    <tr class="{version_class}">
                        <th>
                            <span class="feature-version-badge {badge_class}">
                                {version_label}
                            </span>
                        </th>
                        {cells}
                    </tr>
                    """)

            model_groups.append(f"""
                <tbody class="feature-model-group">
                    {model_header}
                    {''.join(version_rows)}
                </tbody>
                """)

        comparison_table = f"""
        <table class="data-table feature-comparison-table">
            <thead>
                <tr>
                    <th>Version</th>
                    {headers}
                </tr>
            </thead>

            {''.join(model_groups)}
        </table>
        """

        # -----------------------------------
        # Summary
        # -----------------------------------

        models_count = self.results["Model"].nunique()

        dataset_info = f"{self.n_samples:,} rows"

        original_features = str(self.n_features)

        # -----------------------------------
        # PCA Feature Count
        # -----------------------------------

        pca_rows = self.results.loc[
            self.results["Dataset Version"] == "PCA",
            "Features",
        ]

        pca_feature_counts = []

        for value in pca_rows.dropna():
            feature_text = str(value).strip()

            # Normalize possible range separators
            feature_text = feature_text.replace("–", "-")
            feature_text = feature_text.replace("—", "-")

            parts = feature_text.split("-")

            for part in parts:
                part = part.strip()

                if part.isdigit():
                    pca_feature_counts.append(int(part))

        if pca_feature_counts:
            min_features = min(pca_feature_counts)
            max_features = max(pca_feature_counts)

            if min_features == max_features:
                pca_features = str(min_features)
            else:
                pca_features = f"{min_features}–{max_features}"
        else:
            pca_features = "N/A"

        variance_info = f"{self.variance_threshold:.0%}"

        # -----------------------------------
        # Table Styles
        # -----------------------------------

        table_styles = """
        <style>
            .feature-comparison-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
            }

            .feature-comparison-table th,
            .feature-comparison-table td {
                white-space: pre-line;
                vertical-align: middle;
            }

            .feature-comparison-table thead th {
                white-space: normal;
            }

            /* Model group header */

            .feature-comparison-table
            .feature-model-header th {
                text-align: center !important;
                padding: 16px 18px;
                background: rgba(148, 163, 184, 0.12);
                border-top: 2px solid rgba(148, 163, 184, 0.45);
                border-bottom: 1px solid rgba(148, 163, 184, 0.25);
                border-left: 3px solid #64748b;
            }

            .feature-model-name {
                display: inline-block;
                font-size: 16px;
                font-weight: 700;
                letter-spacing: 0.4px;
                text-align: center;
            }

            /* Version rows */

            .feature-comparison-table
            .feature-original-row {
                background: rgba(56, 189, 248, 0.035);
            }

            .feature-comparison-table
            .feature-pca-row {
                background: rgba(52, 211, 153, 0.055);
            }

            .feature-comparison-table
            .feature-original-row th,
            .feature-comparison-table
            .feature-original-row td {
                border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            }

            .feature-comparison-table
            .feature-pca-row th,
            .feature-comparison-table
            .feature-pca-row td {
                border-bottom: 2px solid rgba(148, 163, 184, 0.35);
            }

            /* Version badges */

            .feature-version-badge {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.4px;
                white-space: nowrap;
            }

            .feature-original-badge {
                color: #38bdf8;
                background: rgba(56, 189, 248, 0.10);
                border: 1px solid rgba(56, 189, 248, 0.25);
            }

            .feature-pca-badge {
                color: #34d399;
                background: rgba(52, 211, 153, 0.10);
                border: 1px solid rgba(52, 211, 153, 0.25);
            }

            /* Group spacing */

            .feature-comparison-table
            .feature-model-group:first-of-type
            .feature-model-header th {
                border-top-width: 1px;
            }

            /* Hover */

            .feature-comparison-table
            .feature-original-row:hover {
                background: rgba(56, 189, 248, 0.09);
            }

            .feature-comparison-table
            .feature-pca-row:hover {
                background: rgba(52, 211, 153, 0.11);
            }
        </style>
        """

        # -----------------------------------
        # HTML Content
        # -----------------------------------

        return f"""
        {table_styles}

        <div class="grid">
            <div class="card">
                <div class="card-title" style="color:#38bdf8;">
                    Models Evaluated
                </div>
                <pre>{models_count}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#a78bfa;">
                    Dataset
                </div>
                <pre>{dataset_info}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#38bdf8;">
                    Original Features
                </div>
                <pre>{original_features}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#34d399;">
                    PCA Features
                </div>
                <pre>{pca_features}</pre>
            </div>

            <div class="card">
                <div class="card-title" style="color:#fbbf24;">
                    PCA Variance Threshold
                </div>
                <pre>{variance_info}</pre>
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
                    <strong>Feature Extraction:</strong>
                    PCA automatically selects the minimum number
                    of components required to preserve the
                    specified variance threshold.
                </p>

                <p>
                    <strong>Comparison:</strong>
                    Each model is evaluated using the original
                    features and PCA-transformed features.
                    Results are grouped by model.
                </p>

                <p>
                    <strong>Relative Loss:</strong>
                    Test Accuracy and Test F1 Weighted losses are calculated
                    relative to the original version of
                    the same model.
                    Negative loss indicates an improvement.
                </p>

                <p>
                    <strong>Speed Comparison:</strong>
                    Fit and prediction speeds are compared
                    against the original version of
                    the same model.
                    Positive values indicate faster execution,
                    while negative values indicate slower execution.
                    A value of 1.00x indicates equal speed.
                </p>

                <p>
                    <strong>Preprocessing:</strong>
                    PCA is fitted separately within each
                    cross-validation training fold.
                    Fit and prediction times include
                    PCA preprocessing.
                </p>

                <p>
                    <strong>Cross Validation:</strong>
                    {self.n_folds}-Fold Cross Validation.
                </p>
            </div>
        </section>
        """
