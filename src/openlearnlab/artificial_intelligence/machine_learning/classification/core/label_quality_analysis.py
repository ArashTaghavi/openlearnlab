from pathlib import Path
import time

import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..reporting.templates import LabelQualityAnalysisTemplate


def _save_excel(
    data: pd.DataFrame,
    path: Path,
    highlighted_positions=None,
) -> None:
    """Write the original columns and index into one Excel worksheet."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        data.to_excel(
            writer,
            sheet_name="Dataset",
            index=True,
            index_label="Original Index",
        )

        if highlighted_positions is not None:
            worksheet = writer.sheets["Dataset"]
            fill = PatternFill(
                start_color="D9EAD3",
                end_color="D9EAD3",
                fill_type="solid",
            )

            for position in highlighted_positions:
                excel_row = int(position) + 2

                for cell in worksheet[excel_row]:
                    cell.fill = fill


def label_quality_analysis(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    model_name,
    dataset_name: str,
    output: Path,
    random_state: int,
    n_neighbors: int = 10,
    min_confidence: float = 90.0,
    min_disagreement: float = 80.0,
) -> Path:
    """Flag confidently wrong predictions, vote using non-flagged neighbors,
    and compare original vs provisionally corrected labels using 5-fold CV.

    Intended for binary classifiers supporting predict_proba. Confidence is
    the out-of-fold probability assigned to the predicted class.
    """
    features = list(features)
    if not features or target in features or len(features) != len(set(features)):
        raise ValueError("Provide unique features excluding the target column.")
    if not df.index.is_unique:
        raise ValueError("DataFrame index must be unique.")
    if df[target].isna().any():
        raise ValueError("Target contains missing values.")
    if not 0 <= min_confidence <= 100 or not 0 <= min_disagreement <= 100:
        raise ValueError("Percentage thresholds must be between 0 and 100.")
    if not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")

    X = df[features]
    y = df[target]
    if y.nunique() != 2:
        raise ValueError("This analysis currently supports binary classification only.")
    if y.value_counts().min() < 5:
        raise ValueError("Each class must have at least 5 records for 5-fold CV.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # Out-of-fold predictions and probabilities use identical CV splits.
    # The estimator must implement predict_proba.
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba.")
    classes = np.unique(y.to_numpy())
    # CV probability columns follow the sorted classes order.
    probabilities = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
    y_pred = cross_val_predict(clone(model), X, y, cv=cv, method="predict")
    predicted_indices = np.searchsorted(classes, y_pred)
    confidence = probabilities[np.arange(len(df)), predicted_indices] * 100.0
    y_true = y.to_numpy()
    flagged = (y_pred != y_true) & (confidence >= min_confidence)
    flagged_positions = np.flatnonzero(flagged)
    pool_positions = np.flatnonzero(~flagged)

    if len(flagged_positions) and len(pool_positions) < n_neighbors:
        raise ValueError(
            "Not enough non-flagged records for the requested n_neighbors. "
            "Reduce n_neighbors or increase min_confidence."
        )

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

    candidates = []
    if len(flagged_positions):
        # Fit on NON-FLAGGED records only: every flagged record is excluded
        # from every other flagged record's neighbor pool.
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(transformed[pool_positions])
        distances, neighbor_local_indices = knn.kneighbors(
            transformed[flagged_positions]
        )
        for i, position in enumerate(flagged_positions):
            neighbor_positions = pool_positions[neighbor_local_indices[i]]
            opposing = y_true[neighbor_positions] != y_true[position]
            disagreement = 100.0 * float(np.mean(opposing))
            opposite_count = int(opposing.sum())
            eligible = (
                disagreement >= min_disagreement and opposite_count > n_neighbors / 2
            )
            proposed_label = y_pred[position] if eligible else y_true[position]
            candidates.append(
                {
                    "Position": int(position),
                    "Record": str(df.index[position]),
                    "Actual": y_true[position],
                    "Predicted": y_pred[position],
                    "Confidence (%)": float(confidence[position]),
                    "Proposed Label": proposed_label,
                    "Neighbors": n_neighbors,
                    "Opposite Neighbors": opposite_count,
                    "Same Neighbors": n_neighbors - opposite_count,
                    "Disagreement (%)": disagreement,
                    "Mean Distance": float(np.mean(distances[i])),
                    "Selected": bool(eligible),
                }
            )

    candidate_columns = [
        "Position",
        "Record",
        "Actual",
        "Predicted",
        "Confidence (%)",
        "Proposed Label",
        "Neighbors",
        "Opposite Neighbors",
        "Same Neighbors",
        "Disagreement (%)",
        "Mean Distance",
        "Selected",
    ]
    candidates_df = pd.DataFrame(candidates, columns=candidate_columns)
    selected = candidates_df[candidates_df["Selected"]]
    corrected = df.copy(deep=True)
    target_pos = corrected.columns.get_loc(target)
    for _, row in selected.iterrows():
        corrected.iat[int(row["Position"]), target_pos] = row["Proposed Label"]

    def evaluate(data: pd.DataFrame) -> dict:
        preds = cross_val_predict(
            clone(model), data[features], data[target], cv=cv, method="predict"
        )
        return {
            "accuracy": float(accuracy_score(data[target], preds)),
            "matrix": confusion_matrix(data[target], preds, labels=classes),
            "report": pd.DataFrame(
                classification_report(
                    data[target],
                    preds,
                    labels=classes,
                    output_dict=True,
                    zero_division=0,
                )
            ).T,
        }

    def comparison_scores(data: pd.DataFrame) -> dict:
        positive = classes[1]
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(
                precision_score, pos_label=positive, zero_division=0
            ),
            "recall": make_scorer(recall_score, pos_label=positive, zero_division=0),
            "f1": make_scorer(f1_score, pos_label=positive, zero_division=0),
            "roc_auc": "roc_auc",
        }
        result = cross_validate(
            clone(model),
            data[features],
            data[target],
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
        )
        predict_times = []
        for estimator, (_, test_indices) in zip(
            result["estimator"], cv.split(data[features], data[target])
        ):
            test_X = data[features].iloc[test_indices]
            estimator.predict(test_X.iloc[:1])
            start = time.perf_counter()
            estimator.predict(test_X)
            predict_times.append(time.perf_counter() - start)

        scores = {}
        for metric in scoring:
            scores[metric] = {
                "train_mean": float(np.mean(result[f"train_{metric}"])),
                "train_std": float(np.std(result[f"train_{metric}"])),
                "test_mean": float(np.mean(result[f"test_{metric}"])),
                "test_std": float(np.std(result[f"test_{metric}"])),
            }
        scores["fit_time"] = {
            "mean": float(np.mean(result["fit_time"])),
            "std": float(np.std(result["fit_time"])),
        }
        scores["predict_time"] = {
            "mean": float(np.mean(predict_times)),
            "std": float(np.std(predict_times)),
        }
        return scores

    original_eval = evaluate(df)
    if corrected[target].value_counts().reindex(classes, fill_value=0).min() < 5:
        raise ValueError("Corrected dataset has fewer than 5 records in a class.")
    corrected_eval = evaluate(corrected)
    original_eval["comparison"] = comparison_scores(df)
    corrected_eval["comparison"] = comparison_scores(corrected)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(model_name))
    corrected_path = output / f"{safe_name}_corrected_dataset.xlsx"
    review_path = output / f"{safe_name}_label_review.xlsx"

    _save_excel(
        corrected,
        corrected_path,
        highlighted_positions=selected["Position"].to_numpy(dtype=int),
    )

    # Expert receives ORIGINAL labels and columns, only selected records.
    review = df.iloc[selected["Position"].to_numpy(dtype=int)].copy()
    review["Wrong Prediction Confidence (%)"] = (
        selected["Confidence (%)"].round(2).to_numpy()
    )
    _save_excel(review, review_path)

    template = LabelQualityAnalysisTemplate(
        dataset_name=dataset_name,
        model_name=str(model_name),
        candidates=candidates_df,
        original=original_eval,
        corrected=corrected_eval,
        classes=classes,
        n_neighbors=n_neighbors,
        min_confidence=min_confidence,
        min_disagreement=min_disagreement,
        n_records=len(df),
    )
    return template.save(output=output, suffix_file_name=str(model_name))
