from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..reporting.templates import DiagnosticsTemplate


def diagnostics(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    dataset_name: str,
    output: Path,
    random_state: int,
) -> Path:

    features = list(features)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    X = df[features]
    y = df[target]

    classes = np.unique(y)

    if len(classes) != 2:
        raise ValueError(
            "Classification diagnostics currently supports binary classification."
        )

    # -----------------------------------
    # Out-of-Fold Predictions
    # -----------------------------------

    y_pred = cross_val_predict(
        clone(model),
        X,
        y,
        cv=cv,
        method="predict",
    )

    y_true = y.to_numpy()
    y_pred = np.asarray(y_pred)

    negative_class, positive_class = classes

    y_true_binary = (y_true == positive_class).astype(int)

    # -----------------------------------
    # Prediction Scores
    # -----------------------------------

    probabilities = None
    decision_scores = None

    if hasattr(model, "predict_proba"):
        probabilities = cross_val_predict(
            clone(model),
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]

    elif hasattr(model, "decision_function"):
        decision_scores = cross_val_predict(
            clone(model),
            X,
            y,
            cv=cv,
            method="decision_function",
        )

    # -----------------------------------
    # Classification Errors
    # -----------------------------------

    error_mask = y_true != y_pred

    error_positions = np.flatnonzero(error_mask)

    n_errors = int(error_mask.sum())
    error_rate = float(error_mask.mean())

    # -----------------------------------
    # Overall Metrics
    # -----------------------------------

    accuracy = float(accuracy_score(y_true, y_pred))

    precision = float(
        precision_score(
            y_true,
            y_pred,
            pos_label=positive_class,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            pos_label=positive_class,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            pos_label=positive_class,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=classes,
    )

    tn, fp, fn, tp = cm.ravel()

    roc_auc = None
    logloss = None
    brier = None

    if probabilities is not None:
        roc_auc = float(roc_auc_score(y_true_binary, probabilities))

        logloss = float(
            log_loss(
                y_true_binary,
                probabilities,
                labels=[0, 1],
            )
        )

        brier = float(
            brier_score_loss(
                y_true_binary,
                probabilities,
            )
        )

    elif decision_scores is not None:
        roc_auc = float(
            roc_auc_score(
                y_true_binary,
                decision_scores,
            )
        )

    # -----------------------------------
    # Error Records
    # -----------------------------------

    error_records = df[features].copy()

    error_records["Actual"] = y_true
    error_records["Predicted"] = y_pred

    error_records["Error"] = error_mask

    error_records["Error Type"] = np.select(
        [
            (y_true == negative_class) & (y_pred == positive_class),
            (y_true == positive_class) & (y_pred == negative_class),
        ],
        [
            "False Positive",
            "False Negative",
        ],
        default="Correct",
    )

    if probabilities is not None:
        error_records["Positive Probability"] = probabilities

        error_records["Predicted Confidence"] = np.where(
            y_pred == positive_class,
            probabilities,
            1 - probabilities,
        )

    # -----------------------------------
    # Critical Errors
    # -----------------------------------

    critical_errors = error_records.iloc[error_positions].copy()

    if probabilities is not None:
        critical_errors = critical_errors.sort_values(
            "Predicted Confidence",
            ascending=False,
            kind="stable",
        )

    # -----------------------------------
    # Diagnostic Insights
    # -----------------------------------

    insights: list[tuple[str, str]] = []

    if fn > fp:
        insights.append(
            (
                "False Negative Dominance",
                (
                    "The model produces more false negatives "
                    "than false positives. Consider examining "
                    "positive-class recall."
                ),
            )
        )

    elif fp > fn:
        insights.append(
            (
                "False Positive Dominance",
                (
                    "The model produces more false positives "
                    "than false negatives. Consider examining "
                    "positive-class precision."
                ),
            )
        )

    if recall < 0.5:
        insights.append(
            (
                "Low Positive Recall",
                (
                    "The model fails to identify more than half "
                    "of the positive samples."
                ),
            )
        )

    if precision < 0.5:
        insights.append(
            (
                "Low Positive Precision",
                (
                    "Fewer than half of the predicted positive "
                    "samples are actually positive."
                ),
            )
        )

    if probabilities is not None and n_errors > 0:
        error_confidence = critical_errors["Predicted Confidence"]

        high_confidence_errors = int((error_confidence >= 0.9).sum())

        if high_confidence_errors > 0:
            insights.append(
                (
                    "High-Confidence Misclassifications",
                    (
                        f"{high_confidence_errors} incorrect predictions "
                        "were made with at least 90% predicted confidence. "
                        "Inspect these observations for systematic errors."
                    ),
                )
            )

    if not insights:
        insights.append(
            (
                "No Strong Diagnostic Signals",
                (
                    "The selected checks did not identify strong "
                    "classification error patterns. This does not "
                    "guarantee that the model is free of problems."
                ),
            )
        )

    # -----------------------------------
    # Generate Diagnostics Report
    # -----------------------------------

    template = DiagnosticsTemplate(
        dataset_name=dataset_name,
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        confusion_matrix=cm,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        log_loss=logloss,
        brier_score=brier,
        error_rate=error_rate,
        n_errors=n_errors,
        insights=insights,
        n_samples=len(df),
        n_features=len(features),
        n_folds=cv.get_n_splits(),
        critical_errors=critical_errors,
        probabilities=probabilities,
        decision_scores=decision_scores,
    )

    return template.save(output=output)
