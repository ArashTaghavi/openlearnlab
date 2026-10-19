from html import escape

import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class SimilarityAnalysisTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        summary: pd.DataFrame,
        neighbor_details: dict,
        X: pd.DataFrame,
        n_samples: int,
        n_features: int,
        n_neighbors: int,
        n_folds: int,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title=f"Classification Similarity Analysis -- {model_name}",
        )
        self.model_name = model_name
        self.summary = summary
        self.neighbor_details = neighbor_details
        self.X = X
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.n_folds = n_folds
        self.full_width = True

    @staticmethod
    def _table(df: pd.DataFrame) -> str:
        return (
            '<div style="overflow-x:auto;">'
            + df.to_html(
                classes="data-table",
                index=False,
                border=0,
                escape=True,
                na_rep="—",
                float_format=lambda value: f"{value:,.4f}",
            )
            + "</div>"
        )

    @staticmethod
    def _feature_details(features: dict) -> str:
        values = "".join(
            f"<div><strong>{escape(str(name))}</strong>: " f"{escape(str(value))}</div>"
            for name, value in features.items()
        )
        return (
            '<details class="record-features">'
            "<summary>View Features</summary>"
            f'<div class="record-features-content">{values}</div>'
            "</details>"
        )

    def content(self) -> str:
        configuration = (
            '<section class="table-section">'
            "<h2>Analysis Configuration</h2>"
            "<p>Out-of-fold errors are identified using stratified cross-validation. "
            "Neighbors are selected using feature distances without considering labels.</p>"
            '<div class="grid">'
            f'<div class="card"><div class="card-title">Model</div><pre>{escape(self.model_name)}</pre></div>'
            f'<div class="card"><div class="card-title">Observations</div><pre>{self.n_samples:,}</pre></div>'
            f'<div class="card"><div class="card-title">Features</div><pre>{self.n_features}</pre></div>'
            f'<div class="card"><div class="card-title">Misclassified</div><pre>{len(self.summary):,}</pre></div>'
            f'<div class="card"><div class="card-title">Neighbors per Record</div><pre>{self.n_neighbors}</pre></div>'
            f'<div class="card"><div class="card-title">Evaluation</div><pre>{self.n_folds}-Fold Stratified CV</pre></div>'
            "</div></section>"
        )

        if self.summary.empty:
            return (
                configuration
                + '<section class="table-section"><h2>Similar Records</h2>'
                "<p>No misclassified records detected.</p></section>"
            )

        cards = []
        for _, record in self.summary.iterrows():
            position = int(record["Position"])
            details = self.neighbor_details[position]
            table_rows = []
            for neighbor in details:
                table_rows.append(
                    "<tr>"
                    f'<td>{escape(neighbor["Record"])}</td>'
                    f'<td>{escape(str(neighbor["Actual"]))}</td>'
                    f'<td>{neighbor["Distance"]:.4f}</td>'
                    f'<td>{escape(neighbor["Label Match"])}</td>'
                    f'<td>{self._feature_details(neighbor["Features"])}</td>'
                    "</tr>"
                )

            neighbor_table = (
                '<div style="overflow-x:auto;"><table class="data-table">'
                "<thead><tr><th>Neighbor Record</th><th>Actual</th>"
                "<th>Distance</th><th>Label Match</th><th>Features</th>"
                "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table></div>"
            )

            cards.append(
                '<div class="card">'
                f'<div class="card-title">Record #{escape(record["Record"])}</div>'
                '<div class="grid">'
                f'<div class="card"><div class="card-title">Actual</div><pre>{escape(str(record["Actual"]))}</pre></div>'
                f'<div class="card"><div class="card-title">Predicted</div><pre>{escape(str(record["Predicted"]))}</pre></div>'
                f'<div class="card"><div class="card-title">Label Disagreement</div><pre>{record["Label Disagreement (%)"]:.2f}%</pre></div>'
                f'<div class="card"><div class="card-title">Opposite Neighbors</div><pre>{int(record["Opposite Neighbors"])}</pre></div>'
                f'<div class="card"><div class="card-title">Same Neighbors</div><pre>{int(record["Same Neighbors"])}</pre></div>'
                f'<div class="card"><div class="card-title">Mean Distance</div><pre>{record["Mean Distance"]:.4f}</pre></div>'
                "</div>"
                "<h3>Record Features</h3>"
                f"{self._feature_details(self.X.iloc[position].to_dict())}"
                "<h3>Nearest Neighbors</h3>"
                f"{neighbor_table}"
                "</div>"
            )

        return (
            configuration + '<section class="table-section">'
            "<h2>Similarity Analysis Results</h2>"
            "<p>Misclassified records are ordered by label disagreement "
            "(highest first), then mean neighbor distance (lowest first). "
            "Distances are Euclidean distances after numeric scaling and "
            "categorical one-hot encoding. High disagreement suggests a "
            "label worth reviewing, not proof of an incorrect label.</p>"
            f'<div class="grid">{"".join(cards)}</div>'
            "</section>"
        )
