from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..reporting.templates import LabelCorrectionAnalysisTemplate


def label_correction_analysis(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    model_name,
    dataset_name: str,
    output: Path,
    random_state: int,
    n_neighbors: int = 10,
    min_disagreement: float = 80.0,
) -> Path:
    """Correct labels when at least min_disagreement% of k nearest neighbors disagree.

    Numeric features are standardized; categorical features are one-hot encoded.
    The original dataset is unchanged; all rows are retained. Both versions are
    evaluated with the same 5-fold CV configuration (no holdout split).
    """
    features = list(features)
    if not features or target in features:
        raise ValueError("Provide features without the target column.")
    if not df.index.is_unique:
        raise ValueError("DataFrame index must be unique to identify records.")
    if df[target].isna().any():
        raise ValueError("Target contains missing values.")
    if not 0 <= min_disagreement <= 100:
        raise ValueError("min_disagreement must be between 0 and 100.")
    if n_neighbors < 1 or n_neighbors >= len(df):
        raise ValueError("n_neighbors must be between 1 and len(df) - 1.")

    X = df[features]
    y = df[target]
    if y.nunique() != 2:
        raise ValueError("Automatic neighbor-vote correction requires two classes.")
    if y.value_counts().min() < 5:
        raise ValueError("Each class needs at least 5 records for 5-fold CV.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    labels = np.unique(y.to_numpy())
    y_array = y.to_numpy()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
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
                numeric,
            )
        )
    if categorical:
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
                categorical,
            )
        )
    transformed = ColumnTransformer(transformers).fit_transform(X)
    neighbor_model = NearestNeighbors(metric="euclidean", n_neighbors=n_neighbors + 1)
    neighbor_model.fit(transformed)

    distances, neighbor_indices = neighbor_model.kneighbors(
        transformed, n_neighbors=n_neighbors + 1
    )
    candidates = []
    neighbor_details = {}
    for position in range(len(df)):
        selected = [
            (int(idx), float(distance))
            for idx, distance in zip(neighbor_indices[position], distances[position])
            if idx != position
        ][:n_neighbors]
        neighbor_labels = y_array[[idx for idx, _ in selected]]
        opposing = neighbor_labels != y_array[position]
        opposite_count = int(opposing.sum())
        same_count = len(selected) - opposite_count
        disagreement = 100.0 * opposite_count / len(selected)
        eligible = disagreement >= min_disagreement and opposite_count > same_count
        proposed_label = (
            neighbor_labels[np.flatnonzero(opposing)[0]]
            if eligible
            else y_array[position]
        )
        candidates.append(
            {
                "Position": position,
                "Record": str(df.index[position]),
                "Actual": y_array[position],
                "Proposed Label": proposed_label,
                "Neighbors": len(selected),
                "Opposite Neighbors": opposite_count,
                "Same Neighbors": same_count,
                "Disagreement (%)": disagreement,
                "Mean Distance": float(np.mean([distance for _, distance in selected])),
                "Selected": bool(eligible),
            }
        )

    columns = [
        "Position",
        "Record",
        "Actual",
        "Proposed Label",
        "Neighbors",
        "Opposite Neighbors",
        "Same Neighbors",
        "Disagreement (%)",
        "Mean Distance",
        "Selected",
    ]
    candidates_df = pd.DataFrame(candidates, columns=columns)
    selected = candidates_df.loc[candidates_df["Selected"]]
    corrected = df.copy(deep=True)
    target_col = corrected.columns.get_loc(target)
    for _, record in selected.iterrows():
        corrected.iat[int(record["Position"]), target_col] = record["Proposed Label"]

    def evaluate(data):
        predictions = cross_val_predict(
            clone(model), data[features], data[target], cv=cv, method="predict"
        )
        return {
            "accuracy": float(accuracy_score(data[target], predictions)),
            "matrix": confusion_matrix(data[target], predictions, labels=labels),
            "report": pd.DataFrame(
                classification_report(
                    data[target],
                    predictions,
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )
            ).T,
        }

    original_eval = evaluate(df)
    if corrected[target].value_counts().reindex(labels, fill_value=0).min() < 5:
        raise ValueError(
            "Corrected labels leave fewer than 5 records in a class for CV."
        )
    corrected_eval = evaluate(corrected)

    output.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in str(model_name)
    )
    dataset_path = output / f"{safe_name}_corrected_dataset.csv"
    review_path = output / f"{safe_name}_label_review.csv"
    corrected.to_csv(dataset_path, index=True, index_label="Original Index")
    # Expert review: original dataset columns, only records selected for correction.
    review = df.iloc[selected["Position"].to_numpy(dtype=int)].copy()
    review.to_csv(review_path, index=True, index_label="Original Index")

    template = LabelCorrectionAnalysisTemplate(
        dataset_name=dataset_name,
        model_name=str(model_name),
        candidates=candidates_df,
        original=original_eval,
        corrected=corrected_eval,
        classes=labels,
        n_neighbors=n_neighbors,
        min_disagreement=min_disagreement,
        n_records=len(df),
        dataset_filename=dataset_path.name,
        review_filename=review_path.name,
        features=X,
    )
    return template.save(output=output, suffix_file_name=str(model_name))
