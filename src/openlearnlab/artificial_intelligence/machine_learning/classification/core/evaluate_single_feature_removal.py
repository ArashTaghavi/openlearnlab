import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from ..reporting.templates import SingleFeatureRemovalTemplate


def evaluate_single_feature_removal(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    dataset_name: str,
    output: Path,
    random_state: int,
) -> Path:

    features = list(features)

    if not features:
        raise ValueError("At least one feature is required.")

    if target in features:
        raise ValueError("Target must not be included in features.")

    if len(features) != len(set(features)):
        raise ValueError("Features must be unique.")

    if model is None:
        model = LogisticRegression(max_iter=1000)

    X = df[features]
    y = df[target]

    if y.nunique() != 2:
        raise ValueError("This analysis currently supports binary classification only.")

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if y.value_counts().min() < 5:
        raise ValueError("Each class must have at least 5 records for 5-fold CV.")

    classes = np.unique(y.to_numpy())
    positive_class = classes[1]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            pos_label=positive_class,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            pos_label=positive_class,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            pos_label=positive_class,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
    }

    results = {}

    # -----------------------------------
    # Evaluate Feature Removal
    # -----------------------------------

    for removed_feature in [None] + features:
        if removed_feature is None:
            current_features = features
            name = "All Features"
        else:
            current_features = [
                feature for feature in features if feature != removed_feature
            ]
            name = f"Without {removed_feature}"

        if not current_features:
            continue

        X_current = df[current_features]

        scores = cross_validate(
            clone(model),
            X_current,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            error_score="raise",
        )

        # -----------------------------------
        # Metrics
        # -----------------------------------

        row = {}

        for metric, column in [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("roc_auc", "ROC AUC"),
        ]:
            train_values = scores[f"train_{metric}"]
            test_values = scores[f"test_{metric}"]

            train_mean = float(np.mean(train_values))
            train_std = float(np.std(train_values))

            test_mean = float(np.mean(test_values))
            test_std = float(np.std(test_values))

            gap = train_mean - test_mean

            row[column] = (
                f"Train: {train_mean:.3f} ± {train_std:.3f}\n"
                f"Test: {test_mean:.3f} ± {test_std:.3f}\n"
                f"Gap: {gap:.3f}"
            )

            row[f"_Test {column} Mean"] = test_mean

        # -----------------------------------
        # Fit Time
        # -----------------------------------

        row["_Fit Time Mean"] = float(np.mean(scores["fit_time"]))
        row["_Fit Time Std"] = float(np.std(scores["fit_time"]))

        # -----------------------------------
        # Prediction Time
        # -----------------------------------

        predict_times = []

        for fold_model, (_, test_index) in zip(
            scores["estimator"],
            cv.split(X_current, y),
        ):
            X_test = X_current.iloc[test_index]

            # Warm-up
            fold_model.predict(X_test.iloc[:1])

            start = time.perf_counter()
            fold_model.predict(X_test)
            elapsed = time.perf_counter() - start

            predict_times.append(elapsed)

        row["_Predict Time Mean"] = float(np.mean(predict_times))
        row["_Predict Time Std"] = float(np.std(predict_times))

        results[name] = row

    # -----------------------------------
    # Results DataFrame
    # -----------------------------------

    result = pd.DataFrame(results).T
    result.index.name = "Feature Removal"

    result = result.sort_values(
        by=[
            "_Test F1 Mean",
            "_Test ROC AUC Mean",
            "_Predict Time Mean",
            "_Fit Time Mean",
        ],
        ascending=[False, False, True, True],
    )

    # -----------------------------------
    # Best Row
    # -----------------------------------

    best_row = result.iloc[0]

    best_fit_time = float(best_row["_Fit Time Mean"])
    best_predict_time = float(best_row["_Predict Time Mean"])

    # -----------------------------------
    # Test Metrics Loss
    # -----------------------------------

    loss_columns = []

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]:
        column = f"_Test {metric} Mean"

        best_value = float(best_row[column])
        values = result[column].astype(float)

        loss = best_value - values

        if best_value != 0:
            loss_percent = loss / abs(best_value) * 100
        else:
            loss_percent = pd.Series(
                np.where(loss == 0, 0.0, np.nan),
                index=result.index,
            )

        loss_columns.append(
            [
                f"{percent:.2f}% ({difference:.3f})"
                for percent, difference in zip(
                    loss_percent,
                    loss,
                )
            ]
        )

    result["Test Metrics Loss"] = [
        (
            f"Accuracy: {accuracy}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1: {f1}"
        )
        for accuracy, precision, recall, f1 in zip(*loss_columns)
    ]

    # -----------------------------------
    # Test ROC AUC Loss
    # -----------------------------------

    best_roc_auc = float(best_row["_Test ROC AUC Mean"])

    roc_auc_values = result["_Test ROC AUC Mean"].astype(float)

    roc_auc_loss = best_roc_auc - roc_auc_values

    if best_roc_auc != 0:
        roc_auc_loss_percent = roc_auc_loss / abs(best_roc_auc) * 100
    else:
        roc_auc_loss_percent = pd.Series(
            np.where(
                roc_auc_loss == 0,
                0.0,
                np.nan,
            ),
            index=result.index,
        )

    result["Test ROC AUC Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            roc_auc_loss_percent,
            roc_auc_loss,
        )
    ]

    # -----------------------------------
    # Fit / Predict Time
    # -----------------------------------

    time_values = []

    for index in result.index:
        fit_mean = cast(
            float,
            result.at[index, "_Fit Time Mean"],
        )
        fit_std = cast(
            float,
            result.at[index, "_Fit Time Std"],
        )

        predict_mean = cast(
            float,
            result.at[index, "_Predict Time Mean"],
        )
        predict_std = cast(
            float,
            result.at[index, "_Predict Time Std"],
        )

        time_values.append(
            f"Fit: {fit_mean:.3f} ± {fit_std:.3f}\n"
            f"Predict: {predict_mean:.3f} ± {predict_std:.3f}"
        )

    result["Fit / Predict Time (s)"] = time_values

    # -----------------------------------
    # Speed vs Best
    # -----------------------------------

    speed_values = []

    for index in result.index:
        fit_time = float(result.at[index, "_Fit Time Mean"])
        predict_time = float(result.at[index, "_Predict Time Mean"])

        if fit_time == 0 or best_fit_time == 0:
            fit_speed = 1.0 if fit_time == best_fit_time else np.nan
        elif fit_time <= best_fit_time:
            fit_speed = best_fit_time / fit_time
        else:
            fit_speed = -(fit_time / best_fit_time)

        if predict_time == 0 or best_predict_time == 0:
            predict_speed = 1.0 if predict_time == best_predict_time else np.nan
        elif predict_time <= best_predict_time:
            predict_speed = best_predict_time / predict_time
        else:
            predict_speed = -(predict_time / best_predict_time)

        speed_values.append(f"Predict: {predict_speed:.2f}x\n" f"Fit: {fit_speed:.2f}x")

    result["Speed vs Best"] = speed_values

    # -----------------------------------
    # Final Columns
    # -----------------------------------

    result = result[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "Test Metrics Loss",
            "Test ROC AUC Loss",
            "Fit / Predict Time (s)",
            "Speed vs Best",
        ]
    ]

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = SingleFeatureRemovalTemplate(
        dataset_name=dataset_name,
        results=result,
    )

    return template.save(output=output)
