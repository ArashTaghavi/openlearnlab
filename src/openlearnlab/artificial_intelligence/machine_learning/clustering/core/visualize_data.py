from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.preprocessing import StandardScaler

from ..reporting.templates.visualize_data_template import VisualizeDataTemplate


def visualize_data(
    df: pd.DataFrame,
    features,
    dataset_name: str,
    output: Path,
    scale: bool = True,
    sample_size: int | None = 1500,
    color_by: str | None = None,
    random_state: int = 42,
) -> Path:
    if color_by is not None and color_by not in df.columns:
        raise ValueError(f"Color feature '{color_by}' does not exist in the dataframe.")

    X = (
        df.loc[:, list(features)]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(X) < 5 or X.shape[1] < 2:
        raise ValueError(
            "Visualization requires at least 5 valid rows and 2 numeric features."
        )
    if sample_size is not None and len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=random_state)

    # Select colors with exactly the same row indices and order as the embeddings.
    colors = df.loc[X.index, color_by].copy() if color_by is not None else None
    values = StandardScaler().fit_transform(X) if scale else X.to_numpy(dtype=float)
    n = len(values)
    neighbors = min(15, n - 1)
    perplexity = min(30.0, max(2.0, (n - 1) / 3.0))

    methods = {
        "PCA": lambda: PCA(n_components=2, random_state=random_state).fit_transform(
            values
        ),
        "t-SNE": lambda: TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(values),
        "UMAP": None,
        "MDS": lambda: MDS(
            n_components=2, random_state=random_state, n_init=1, max_iter=150
        ).fit_transform(values),
        "Isomap": lambda: Isomap(n_components=2, n_neighbors=neighbors).fit_transform(
            values
        ),
        "Spectral Embedding": lambda: SpectralEmbedding(
            n_components=2, n_neighbors=neighbors, random_state=random_state
        ).fit_transform(values),
        "Truncated SVD": lambda: TruncatedSVD(
            n_components=2, random_state=random_state
        ).fit_transform(values),
    }
    try:
        import umap

        methods["UMAP"] = lambda: umap.UMAP(
            n_components=2, n_neighbors=neighbors, random_state=random_state
        ).fit_transform(values)
    except ImportError:
        pass

    projections = {}
    errors = {}
    for name, compute in methods.items():
        if compute is None:
            errors[name] = "Install umap-learn to enable this visualization."
            continue
        try:
            embedding = np.asarray(compute())
            if embedding.shape != (n, 2) or not np.isfinite(embedding).all():
                raise ValueError("Invalid projection")
            projections[name] = embedding
        except Exception as exc:
            errors[name] = str(exc)

    if not projections:
        raise ValueError(f"No visualization could be generated: {errors}")
    template = VisualizeDataTemplate(
        dataset_name=dataset_name,
        projections=projections,
        errors=errors,
        colors=colors,
        color_by=color_by,
    )
    return template.save(output=output)
