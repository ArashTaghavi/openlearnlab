from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..reporting.templates import SimilarityAnalysisTemplate


def similarity_analysis(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    model_name,
    dataset_name: str,
    output: Path,
    random_state: int,
    n_neighbors: int = 10,
) -> Path:
    features = list(features)
    X = df[features]
    y = df[target]

    if y.isna().any():
        raise ValueError("Target contains missing values.")
    if not features:
        raise ValueError("At least one feature is required.")
    if len(df) < 2:
        raise ValueError("At least two observations are required.")
    if not 1 <= n_neighbors < len(df):
        raise ValueError("n_neighbors must be between 1 and len(df) - 1.")

    counts = y.value_counts(dropna=False)
    if len(counts) < 2:
        raise ValueError("Classification requires at least two classes.")

    n_splits = 5
    if counts.min() < n_splits:
        raise ValueError(
            f"Each class must have at least {n_splits} samples "
            "for 5-fold stratified cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    y_true = y.to_numpy()
    y_pred = cross_val_predict(clone(model), X, y, cv=cv, method="predict")
    incorrect = y_true != y_pred

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        feature for feature in features if feature not in numeric_features
    ]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessing = ColumnTransformer(transformers)
    transformed = preprocessing.fit_transform(X)
    neighbors_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    neighbors_model.fit(transformed)

    error_positions = np.flatnonzero(incorrect)
    summary_rows = []
    neighbor_details = {}

    if len(error_positions):
        distances, positions = neighbors_model.kneighbors(
            transformed[error_positions], n_neighbors=n_neighbors + 1
        )
        for row_index, position in enumerate(error_positions):
            # Exclude the observation itself, not other duplicate observations.
            selected = [
                (int(neighbor), float(distance))
                for neighbor, distance in zip(
                    positions[row_index], distances[row_index]
                )
                if neighbor != position
            ][:n_neighbors]

            # In the presence of duplicate feature vectors, a self-index may
            # be absent from the returned k+1 ties; the list still has k items.
            selected = selected[:n_neighbors]
            neighbor_positions = np.array([item[0] for item in selected], dtype=int)
            neighbor_distances = np.array([item[1] for item in selected], dtype=float)
            different = y_true[neighbor_positions] != y_true[position]
            disagreement = 100 * float(np.mean(different))

            summary_rows.append(
                {
                    "Position": int(position),
                    "Record": str(df.index[position]),
                    "Actual": y_true[position],
                    "Predicted": y_pred[position],
                    "Label Disagreement (%)": disagreement,
                    "Opposite Neighbors": int(different.sum()),
                    "Same Neighbors": int((~different).sum()),
                    "Mean Distance": float(neighbor_distances.mean()),
                }
            )

            neighbor_details[int(position)] = [
                {
                    "Record": str(df.index[neighbor]),
                    "Actual": y_true[neighbor],
                    "Distance": distance,
                    "Label Match": "No" if is_different else "Yes",
                    "Features": X.iloc[neighbor].to_dict(),
                }
                for (neighbor, distance), is_different in zip(selected, different)
            ]

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "Position",
            "Record",
            "Actual",
            "Predicted",
            "Label Disagreement (%)",
            "Opposite Neighbors",
            "Same Neighbors",
            "Mean Distance",
        ],
    ).sort_values(
        ["Label Disagreement (%)", "Mean Distance"],
        ascending=[False, True],
        kind="stable",
    )

    template = SimilarityAnalysisTemplate(
        dataset_name=dataset_name,
        model_name=str(model_name),
        summary=summary,
        neighbor_details=neighbor_details,
        X=X,
        n_samples=len(df),
        n_features=len(features),
        n_neighbors=n_neighbors,
        n_folds=cv.get_n_splits(),
    )
    return template.save(output=output, suffix_file_name=str(model_name))
