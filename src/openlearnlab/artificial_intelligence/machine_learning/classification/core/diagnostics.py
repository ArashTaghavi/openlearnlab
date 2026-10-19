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

from ..reporting.templates import DiagnosticsTemplate


def diagnostics(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    model_name,
    dataset_name: str,
    output: Path,
    random_state: int,
) -> Path:
    features = list(features)

    X = df[features]
    y = df[target]
    if y.isna().any():
        raise ValueError("Target contains missing values.")

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

    # -----------------------------------
    # Out-of-Fold Predictions
    # -----------------------------------
    y_pred = cross_val_predict(clone(model), X, y, cv=cv, method="predict")
    y_true = y.to_numpy()
    classes = np.unique(y_true)
    incorrect = y_true != y_pred

    # -----------------------------------
    # Out-of-Fold Probabilities
    # Binary Classification
    # -----------------------------------
    probabilities = None

    if len(classes) == 2:
        probabilities = cross_val_predict(
            clone(model),
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]

    # -----------------------------------
    # Classification Metrics
    # -----------------------------------
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    class_report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
    ).T
    class_report.index.name = "Class / Summary"

    # -----------------------------------
    # Prediction Records
    # -----------------------------------
    records = X.copy()
    records["Actual"] = y_true
    records["Predicted"] = y_pred
    records["Is Error"] = incorrect

    error_records = records.loc[incorrect].copy()

    if probabilities is not None:
        error_probabilities = probabilities[incorrect]
        error_records["P(Actual)"] = np.where(
            y_true[incorrect] == classes[1],
            error_probabilities,
            1 - error_probabilities,
        )
        error_records["P(Predicted)"] = np.where(
            y_pred[incorrect] == classes[1],
            error_probabilities,
            1 - error_probabilities,
        )
        error_records["Error Type"] = np.where(
            y_true[incorrect] == classes[1], "FN", "FP"
        )
        error_records = error_records.sort_values(
            "P(Predicted)", ascending=False, kind="stable"
        )
    else:
        error_records["P(Actual)"] = np.nan
        error_records["P(Predicted)"] = np.nan
        error_records["Error Type"] = "Misclassified"

    # -----------------------------------
    # Per-Class Error Analysis
    # -----------------------------------
    class_rows = []
    for label in classes:
        mask = y_true == label
        count = int(mask.sum())
        errors = int(np.count_nonzero(incorrect & mask))
        class_rows.append(
            {
                "Class": str(label),
                "Samples": count,
                "Correct": count - errors,
                "Errors": errors,
                "Error Rate (%)": 100 * errors / count,
            }
        )

    class_error_analysis = pd.DataFrame(class_rows).sort_values(
        ["Error Rate (%)", "Errors"],
        ascending=[False, False],
        kind="stable",
    )

    # -----------------------------------
    # Error Association with Numeric Features
    # -----------------------------------
    feature_rows = []
    error_values = incorrect.astype(float)
    for feature in X.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        if (
            valid.sum() < 3
            or np.std(values[valid]) == 0
            or np.std(error_values[valid]) == 0
        ):
            continue
        correlation = float(np.corrcoef(values[valid], error_values[valid])[0, 1])
        if not np.isfinite(correlation):
            continue
        feature_rows.append(
            {
                "Feature": str(feature),
                "Error Correlation": correlation,
                "Absolute Correlation": abs(correlation),
            }
        )

    feature_error_analysis = pd.DataFrame(
        feature_rows,
        columns=["Feature", "Error Correlation", "Absolute Correlation"],
    ).sort_values("Absolute Correlation", ascending=False, kind="stable")

    # -----------------------------------
    # Diagnostic Insights
    # -----------------------------------
    error_count = int(incorrect.sum())
    insights = []
    if error_count == 0:
        insights.append(
            (
                "No Out-of-Fold Classification Errors",
                "All out-of-fold predictions were correct. "
                "Validate on independent data before drawing conclusions.",
            )
        )
    else:
        insights.append(
            (
                "Misclassified Observations",
                f"{error_count:,} of {len(df):,} observations "
                f"({100 * error_count / len(df):.2f}%) were misclassified.",
            )
        )
        worst = class_error_analysis.iloc[0]
        insights.append(
            (
                "Most Difficult Actual Class",
                f"Class '{worst['Class']}' has the highest error rate "
                f"({worst['Error Rate (%)']:.2f}%) among its actual observations.",
            )
        )

    if not feature_error_analysis.empty:
        strongest = feature_error_analysis.iloc[0]
        if strongest["Absolute Correlation"] >= 0.3:
            insights.append(
                (
                    "Feature-Associated Errors",
                    f"Feature '{strongest['Feature']}' has a correlation "
                    f"of {strongest['Error Correlation']:.2f} with the error indicator. "
                    "This is descriptive, not evidence of causation.",
                )
            )

    # -----------------------------------
    # HTML Report
    # -----------------------------------
    template = DiagnosticsTemplate(
        dataset_name=dataset_name,
        model_name=str(model_name),
        metrics=metrics,
        classes=classes,
        confusion=matrix,
        class_report=class_report,
        error_records=error_records,
        class_error_analysis=class_error_analysis,
        feature_error_analysis=feature_error_analysis,
        insights=insights,
        n_samples=len(df),
        n_features=len(features),
        n_folds=cv.get_n_splits(),
        feature_names=features,
        y_true=y_true,
        probabilities=probabilities,
    )
    return template.save(output=output, suffix_file_name=str(model_name))
