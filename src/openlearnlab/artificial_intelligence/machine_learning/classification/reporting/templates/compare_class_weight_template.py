import base64
from html import escape
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ....base.reporting.base_template import BaseTemplate


class CompareClassWeightTemplate(BaseTemplate):
    def __init__(
        self,
        dataset_name: str,
        results: pd.DataFrame,
        class_distribution: pd.DataFrame,
    ):
        super().__init__(
            dataset_name=dataset_name,
            title="Classification - Class Weight Comparison",
        )

        self.results = results.copy()
        self.class_distribution = class_distribution.copy()

    # -----------------------------------
    # Class Distribution Chart
    # -----------------------------------

    def _plot(self) -> str:
        fig, ax = plt.subplots(
            figsize=(9, 5),
        )

        classes = self.class_distribution["Class"].astype(str)
        counts = self.class_distribution["Count"]
        percentages = self.class_distribution["Percentage"]

        bars = ax.bar(
            classes,
            counts,
        )

        for bar, count, percentage in zip(
            bars,
            counts,
            percentages,
        ):
            ax.annotate(
                f"{count:,} ({percentage:.2f}%)",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Class Distribution")
        ax.set_ylim(0, counts.max() * 1.15)
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        buffer = BytesIO()

        try:
            fig.savefig(
                buffer,
                format="png",
                dpi=150,
                bbox_inches="tight",
            )

            buffer.seek(0)

            return base64.b64encode(buffer.read()).decode("utf-8")

        finally:
            plt.close(fig)
            buffer.close()

    # -----------------------------------
    # Class Balance Statistics
    # -----------------------------------

    def _statistics_section(self) -> str:
        distribution = self.class_distribution

        total_samples = int(distribution["Count"].sum())

        majority_count = int(distribution["Count"].max())
        minority_count = int(distribution["Count"].min())

        majority_class = distribution.loc[distribution["Count"].idxmax(), "Class"]

        minority_class = distribution.loc[distribution["Count"].idxmin(), "Class"]

        imbalance_ratio = majority_count / minority_count

        class_counts = distribution.set_index("Class")["Count"]
        scale_pos_weight = class_counts.loc[0] / class_counts.loc[1]

        statistics = pd.DataFrame(
            {
                "Metric": [
                    "Total Samples",
                    "Majority Class",
                    "Majority Count",
                    "Majority Percentage",
                    "Minority Class",
                    "Minority Count",
                    "Minority Percentage",
                    "Imbalance Ratio",
                    "Suggested Scale Pos Weight",
                ],
                "Value": [
                    f"{total_samples:,}",
                    str(majority_class),
                    f"{majority_count:,}",
                    f"{majority_count / total_samples * 100:.2f}%",
                    str(minority_class),
                    f"{minority_count:,}",
                    f"{minority_count / total_samples * 100:.2f}%",
                    f"{imbalance_ratio:.2f}:1",
                    f"{scale_pos_weight:.4f}",
                ],
            }
        )

        headers = "".join(
            f"<th>{escape(str(column))}</th>" for column in statistics.columns
        )

        rows = []

        for _, row in statistics.iterrows():
            cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)

            rows.append(f"<tr>{cells}</tr>")

        return f"""
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
        """

    # -----------------------------------
    # Class Distribution Section
    # -----------------------------------

    def _distribution_section(self) -> str:
        chart = self._plot()
        statistics = self._statistics_section()

        return f"""
        <section class="table-section">
            <h2>Class Distribution & Balance Statistics</h2>

            <div style="
                display:grid;
                grid-template-columns:1fr 1fr;
                gap:20px;
                align-items:start;
            ">
                <div class="card" style="min-width:0;">
                    <img
                        src="data:image/png;base64,{chart}"
                        alt="Class Distribution"
                        style="
                            display:block;
                            width:100%;
                            height:auto;
                        "
                    >
                </div>

                <div style="min-width:0;">
                    {statistics}
                </div>
            </div>

            <div class="card" style="margin-top:20px;">
                <p>
                    <strong>Class Distribution:</strong>
                    Shows the number and percentage of samples
                    belonging to each target class.
                </p>

                <p>
                    <strong>Imbalance Ratio:</strong>
                    Represents the ratio between the majority
                    and minority classes. A value of 1:1 indicates
                    perfectly balanced classes.
                </p>

                <p>
                    <strong>Suggested Scale Pos Weight:</strong>
                    The ratio of negative to positive samples,
                    used as an initial weighting value for
                    binary XGBoost classification.
                    During cross-validation, the balanced weight
                    is calculated separately from each training fold.
                </p>
            </div>
        </section>
        """

    # -----------------------------------
    # Description Section
    # -----------------------------------

    def _description_section(self) -> str:
        return """
        <section class="table-section">
            <h2>Class Weight Comparison</h2>

            <div class="card">
                <p>
                    This report compares model performance with
                    <strong>class_weight=None</strong> and
                    <strong>class_weight="balanced"</strong>
                    using stratified cross-validation.
                </p>

                <p>
                    For models supporting scale_pos_weight,
                    the comparison uses scale_pos_weight=1
                    and a balanced positive-class weight calculated
                    from the training data of each fold.
                </p>

                <p>
                    Balanced weighting increases the importance
                    of underrepresented classes during training.
                    This may improve minority-class recall,
                    although precision or accuracy may decrease.
                </p>

                <p>
                    Results are reported as mean ± standard deviation
                    across cross-validation folds.
                </p>
            </div>
        </section>
        """

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
            <h2>Cross Validation Results</h2>

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
        return (
            self._distribution_section()
            + self._description_section()
            + self._table_section()
        )
