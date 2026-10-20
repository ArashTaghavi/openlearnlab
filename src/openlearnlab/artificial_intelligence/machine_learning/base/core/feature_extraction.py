from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..reporting.templates import FeatureExtractionTemplate


def feature_extraction(
    df: pd.DataFrame,
    target: str,
    dataset_name: str,
    output: Path,
    features,
    variance_threshold: float,
) -> Path:

    X = df[features]

    # -----------------------------------
    # Scaling
    # -----------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------
    # PCA
    # -----------------------------------

    pca = PCA(
        n_components=variance_threshold,
    )

    X_pca = pca.fit_transform(X_scaled)

    # -----------------------------------
    # Extracted Features
    # -----------------------------------

    component_names = [f"PC{i + 1}" for i in range(pca.n_components_)]

    extracted_df = pd.DataFrame(
        X_pca,
        columns=component_names,
        index=df.index,
    )

    extracted_df[target] = df[target]

    # -----------------------------------
    # PCA Information
    # -----------------------------------

    info = pd.DataFrame(
        {
            "Component": component_names,
            "Explained Variance": pca.explained_variance_ratio_,
            "Cumulative Variance": (pca.explained_variance_ratio_.cumsum()),
        }
    )

    # -----------------------------------
    # Feature Loadings
    # -----------------------------------

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=component_names,
    )

    loadings.index.name = "Feature"

    # -----------------------------------
    # Summary
    # -----------------------------------

    summary = {
        "Original Features": len(features),
        "Extracted Components": int(pca.n_components_),
        "Variance Threshold": f"{variance_threshold:.2%}",
        "Explained Variance": (f"{pca.explained_variance_ratio_.sum():.2%}"),
    }

    # -----------------------------------
    # Save Extracted Dataset
    # -----------------------------------

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    extracted_df.to_csv(
        output / "extracted_dataset.csv",
        index=False,
    )

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = FeatureExtractionTemplate(
        dataset_name=dataset_name,
        summary=summary,
        info=info,
        loadings=loadings,
    )

    return template.save(output=output)
