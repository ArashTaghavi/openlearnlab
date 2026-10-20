from __future__ import annotations

from hdbscan import HDBSCAN
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def clustering_models(
    random_state: int = 42,
    selected_models: list[str] | None = None,
    ignored_models: list[str] | None = None,
):

    models = {
        # Centroid-based
        "K-Means": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    KMeans(
                        n_clusters=3,
                        random_state=random_state,
                        n_init="auto",
                    ),
                ),
            ]
        ),
        "MiniBatch K-Means": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MiniBatchKMeans(
                        n_clusters=3,
                        random_state=random_state,
                        n_init="auto",
                    ),
                ),
            ]
        ),
        "Bisecting K-Means": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    BisectingKMeans(
                        n_clusters=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        # Hierarchical
        "Agglomerative": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", AgglomerativeClustering(n_clusters=3)),
            ]
        ),
        "BIRCH": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Birch(n_clusters=3)),
            ]
        ),
        # Density-based
        "DBSCAN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DBSCAN(eps=0.5, min_samples=5)),
            ]
        ),
        "HDBSCAN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HDBSCAN(min_cluster_size=5)),
            ]
        ),
        "OPTICS": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", OPTICS(min_samples=5)),
            ]
        ),
        # Distribution-based
        "Gaussian Mixture": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GaussianMixture(
                        n_components=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Bayesian Gaussian Mixture": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    BayesianGaussianMixture(
                        n_components=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        # Mode-seeking
        "Mean Shift": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MeanShift()),
            ]
        ),
        # Graph-based
        "Spectral": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SpectralClustering(
                        n_clusters=3,
                        random_state=random_state,
                        assign_labels="kmeans",
                    ),
                ),
            ]
        ),
        # Exemplar-based
        "Affinity Propagation": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    AffinityPropagation(
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if selected_models is not None:
        models = {name: models[name] for name in selected_models}

    if ignored_models is not None:
        models = {
            name: model for name, model in models.items() if name not in ignored_models
        }

    return models
