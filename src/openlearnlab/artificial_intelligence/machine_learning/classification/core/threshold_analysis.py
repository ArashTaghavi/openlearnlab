from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..reporting.templates import ThresholdAnalysisTemplate


def threshold_analysis(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    model_name,
    dataset_name: str,
    output: Path,
    random_state: int,
    metric: str = "recall",
    average: int | str = 1,
) -> Path:
    features = list(features)
    X = df[features]
    y = df[target]

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    counts = y.value_counts(dropna=False)

    if len(counts) != 2:
        raise ValueError("Threshold analysis requires exactly two classes.")

    n_splits = 5

    if counts.min() < n_splits:
        raise ValueError(
            f"Each class must have at least {n_splits} samples "
            "for 5-fold stratified cross-validation."
        )

    if metric not in ("precision", "recall", "f1", "accuracy"):
        raise ValueError("Metric must be precision, recall, f1 or accuracy.")

    classes = np.unique(y.to_numpy())

    if (
        metric != "accuracy"
        and average not in ("macro", "weighted")
        and average not in classes
    ):
        raise ValueError("Average must be a class label, macro or weighted.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    probabilities = cross_val_predict(
        clone(model),
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    y_true = y.to_numpy()

    def evaluate(threshold: float) -> dict:
        predicted = (probabilities >= threshold).astype(int)
        y_pred = classes[predicted]

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=classes,
        )

        report = pd.DataFrame(
            classification_report(
                y_true,
                y_pred,
                labels=classes,
                output_dict=True,
                zero_division=0,
            )
        ).T

        report.index.name = "Class / Summary"

        if metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            scoring = {
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            }[metric]

            if average in ("macro", "weighted"):
                score = scoring(
                    y_true,
                    y_pred,
                    average=average,
                    zero_division=0,
                )
            else:
                score = np.asarray(
                    scoring(
                        y_true,
                        y_pred,
                        labels=[average],
                        average=None,
                        zero_division=0,
                    )
                )[0]

        if average in ("macro", "weighted"):
            precision = precision_score(
                y_true,
                y_pred,
                average=average,
                zero_division=0,
            )
            recall = recall_score(
                y_true,
                y_pred,
                average=average,
                zero_division=0,
            )
        else:
            precision = np.asarray(
                precision_score(
                    y_true,
                    y_pred,
                    labels=[average],
                    average=None,
                    zero_division=0,
                )
            )[0]
            recall = np.asarray(
                recall_score(
                    y_true,
                    y_pred,
                    labels=[average],
                    average=None,
                    zero_division=0,
                )
            )[0]

        return {
            "Threshold": float(threshold),
            "Target Score": float(score),
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision),
            "Recall": float(recall),
            "Confusion Matrix": matrix,
            "Classification Report": report,
        }

    default = evaluate(0.5)

    unique_probabilities = np.unique(probabilities)

    thresholds = np.concatenate(
        (
            unique_probabilities,
            [np.nextafter(unique_probabilities[-1], np.inf)],
        )
    )

    best_by_score = {}

    for threshold in thresholds:
        result = evaluate(float(threshold))

        if result["Target Score"] <= default["Target Score"] + 1e-12:
            continue

        score = round(result["Target Score"], 12)
        current = best_by_score.get(score)

        if current is None or (
            result["Accuracy"],
            -abs(result["Threshold"] - 0.5),
        ) > (
            current["Accuracy"],
            -abs(current["Threshold"] - 0.5),
        ):
            best_by_score[score] = result

    results = sorted(
        best_by_score.values(),
        key=lambda row: row["Target Score"],
        reverse=True,
    )

    results.append(default)

    template = ThresholdAnalysisTemplate(
        dataset_name=dataset_name,
        model_name=str(model_name),
        metric=metric,
        average=average,
        classes=classes,
        default=default,
        results=results,
        n_samples=len(df),
        n_features=len(features),
        n_folds=cv.get_n_splits(),
    )

    return template.save(
        output=output,
        suffix_file_name=str(model_name),
    )
