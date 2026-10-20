import base64
import time
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

from ..reporting.templates import CompareModelsTemplate


def compare_models(
    df: pd.DataFrame,
    features,
    models: dict,
    dataset_name: str,
    output: Path,
    max_noise_ratio: float = 30.0,
    silhouette_sample_size: int | None = 5000,
    random_state: int = 42,
):
    features = list(features)

    if not models:
        raise ValueError("No clustering models were provided.")

    if not 0 <= max_noise_ratio <= 100:
        raise ValueError("max_noise_ratio must be between 0 and 100.")

    if silhouette_sample_size is not None and silhouette_sample_size < 2:
        raise ValueError("silhouette_sample_size must be at least 2.")

    X = df[features]

    results = {}

    for name, model in models.items():

        estimator = clone(model)

        # -----------------------------------
        # Fit + labels
        # -----------------------------------

        start = time.perf_counter()

        if isinstance(estimator, Pipeline):
            X_transformed = estimator[:-1].fit_transform(X)
            clusterer = estimator[-1]
        else:
            X_transformed = X
            clusterer = estimator

        if hasattr(clusterer, "fit_predict"):
            labels = clusterer.fit_predict(X_transformed)
        else:
            clusterer.fit(X_transformed)

            if hasattr(clusterer, "labels_"):
                labels = clusterer.labels_
            else:
                labels = clusterer.predict(X_transformed)

        fit_time = time.perf_counter() - start

        labels = np.asarray(labels)

        # -----------------------------------
        # Cluster / Noise information
        # -----------------------------------

        noise_mask = labels == -1
        valid_mask = ~noise_mask

        n_noise = int(noise_mask.sum())
        noise_ratio = float(noise_mask.mean() * 100)

        n_clusters = len(np.unique(labels[valid_mask]))
        n_valid = int(valid_mask.sum())

        # -----------------------------------
        # Metrics
        # -----------------------------------

        silhouette = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan

        if 2 <= n_clusters < n_valid:

            valid_X = X_transformed[valid_mask]
            valid_labels = labels[valid_mask]

            # Silhouette sampling
            if silhouette_sample_size is not None and n_valid > silhouette_sample_size:
                rng = np.random.default_rng(random_state)

                cluster_indices = [
                    np.flatnonzero(valid_labels == label)
                    for label in np.unique(valid_labels)
                ]

                if silhouette_sample_size > n_clusters:

                    selected_indices = [
                        rng.choice(indices) for indices in cluster_indices
                    ]

                    remaining_indices = np.setdiff1d(
                        np.arange(n_valid),
                        selected_indices,
                    )

                    remaining_size = silhouette_sample_size - n_clusters

                    if remaining_size > 0:
                        selected_indices.extend(
                            rng.choice(
                                remaining_indices,
                                size=remaining_size,
                                replace=False,
                            )
                        )

                    selected_indices = np.asarray(
                        selected_indices,
                        dtype=int,
                    )

                    sampled_labels = valid_labels[selected_indices]

                    if 2 <= len(np.unique(sampled_labels)) < len(sampled_labels):
                        silhouette = silhouette_score(
                            valid_X[selected_indices],
                            sampled_labels,
                        )

            else:
                silhouette = silhouette_score(
                    valid_X,
                    valid_labels,
                )

            davies_bouldin = davies_bouldin_score(
                valid_X,
                valid_labels,
            )

            calinski_harabasz = calinski_harabasz_score(
                valid_X,
                valid_labels,
            )

        # -----------------------------------
        # Eligibility
        # -----------------------------------

        eligible = pd.notna(silhouette) and noise_ratio <= max_noise_ratio

        # -----------------------------------
        # Internal results
        # -----------------------------------

        results[name] = {
            "Clusters": n_clusters,
            "Noise Count": n_noise,
            "Noise (%)": noise_ratio,
            "Silhouette": silhouette,
            "Davies-Bouldin": davies_bouldin,
            "Calinski-Harabasz": calinski_harabasz,
            "Fit Time (s)": fit_time,
            "Eligible": eligible,
        }

    result = pd.DataFrame.from_dict(
        results,
        orient="index",
    )

    result.index.name = "Model"

    # -----------------------------------
    # Sort
    # -----------------------------------

    result["Eligible"] = result["Eligible"].astype(bool)

    result = result.sort_values(
        by=[
            "Eligible",
            "Silhouette",
            "Davies-Bouldin",
            "Calinski-Harabasz",
            "Fit Time (s)",
        ],
        ascending=[
            False,
            False,
            True,
            False,
            True,
        ],
        na_position="last",
    )

    # -----------------------------------
    # Best model
    # -----------------------------------

    eligible_models = result[result["Eligible"]]

    if eligible_models.empty:
        best_model = "N/A"
        best_silhouette = np.nan
        best_db = np.nan
        best_ch = np.nan
        best_fit_time = np.nan

    else:
        best_model = str(eligible_models.index[0])

        best_silhouette = float(result.loc[best_model, "Silhouette"])

        best_db = float(result.loc[best_model, "Davies-Bouldin"])

        best_ch = float(result.loc[best_model, "Calinski-Harabasz"])

        best_fit_time = float(result.loc[best_model, "Fit Time (s)"])

    # -----------------------------------
    # Loss vs Best
    # -----------------------------------

    loss_values = []

    for _, row in result.iterrows():

        losses = []

        for column, best_value, higher_is_better in [
            ("Silhouette", best_silhouette, True),
            ("Davies-Bouldin", best_db, False),
            ("Calinski-Harabasz", best_ch, True),
        ]:

            value = float(row[column])

            if pd.isna(value) or pd.isna(best_value):
                losses.append(f"{column}: N/A")
                continue

            if higher_is_better:
                loss = best_value - value
            else:
                loss = value - best_value

            percent = loss / abs(best_value) * 100 if best_value != 0 else 0.0

            losses.append(f"{column}: {percent:.2f}% ({loss:.3f})")

        loss_values.append("\n".join(losses))

    result["Loss vs Best"] = loss_values

    # -----------------------------------
    # Speed vs Best
    # -----------------------------------

    speed_values = []

    for fit_time in result["Fit Time (s)"]:

        fit_time = float(fit_time)

        if pd.isna(best_fit_time) or best_fit_time == 0 or fit_time == 0:
            speed_values.append("N/A")
            continue

        if np.isclose(fit_time, best_fit_time):
            speed = 1.0

        elif fit_time < best_fit_time:
            speed = best_fit_time / fit_time

        else:
            speed = -(fit_time / best_fit_time)

        speed_values.append(f"{speed:.2f}x")

    result["Speed vs Best"] = speed_values

    # -----------------------------------
    # Fit time chart
    # -----------------------------------

    plot_data = result["Fit Time (s)"].sort_values()

    fig, ax = plt.subplots(figsize=(11, 8))

    bars = ax.barh(
        plot_data.index.astype(str),
        plot_data.to_numpy(dtype=float),
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Model")
    ax.set_title("Clustering Fit Time Comparison")
    ax.invert_yaxis()

    max_value = float(plot_data.max())

    if max_value > 0:
        ax.set_xlim(0, max_value * 1.18)

    for bar, value in zip(bars, plot_data):

        ax.text(
            bar.get_width() + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}s",
            ha="left",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    buffer.seek(0)

    chart = base64.b64encode(buffer.read()).decode("utf-8")

    buffer.close()

    # -----------------------------------
    # Format report values
    # -----------------------------------

    result["Noise (%)"] = result["Noise (%)"].map(lambda x: f"{x:.2f}%")

    for column in [
        "Silhouette",
        "Davies-Bouldin",
        "Calinski-Harabasz",
    ]:
        result[column] = result[column].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )

    result["Fit Time (s)"] = result["Fit Time (s)"].map(lambda x: f"{x:.3f}")

    result["Clusters"] = result["Clusters"].astype(int)
    result["Noise Count"] = result["Noise Count"].astype(int)

    result["Eligible"] = result["Eligible"].map(
        {
            True: "Yes",
            False: "No",
        }
    )

    # -----------------------------------
    # Final column order
    # -----------------------------------

    result = result[
        [
            "Clusters",
            "Noise Count",
            "Noise (%)",
            "Silhouette",
            "Davies-Bouldin",
            "Calinski-Harabasz",
            "Loss vs Best",
            "Fit Time (s)",
            "Speed vs Best",
            "Eligible",
        ]
    ]

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = CompareModelsTemplate(
        dataset_name=dataset_name,
        results=result,
        best_model=best_model,
        n_samples=len(df),
        n_features=len(features),
        max_noise_ratio=max_noise_ratio,
        silhouette_sample_size=silhouette_sample_size,
        chart=chart,
    )

    return template.save(output=output)
