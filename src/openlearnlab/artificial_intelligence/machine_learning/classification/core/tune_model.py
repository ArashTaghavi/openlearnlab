import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from ..reporting.templates import TuneModelTemplate


def tune_model(
    df: pd.DataFrame,
    target: str,
    model,
    param_grid: dict,
    dataset_name: str,
    output: Path,
    features,
    cv: int = 5,
    scoring=None,
    random_state: int = 44,
) -> Path:

    if scoring is None:
        scoring = {
            "accuracy": "accuracy",
            "f1": "f1_weighted",
        }

    X = df[features]
    y = df[target]

    kfold = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    # -----------------------------------
    # Grid Search
    # -----------------------------------

    grid_search = GridSearchCV(
        estimator=clone(model),
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=kfold,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_

    tuned_model = clone(model).set_params(**best_params)

    # -----------------------------------
    # Base / Tuned Models
    # -----------------------------------

    models = {
        "Base": clone(model),
        "Tuned": tuned_model,
    }

    results = {}

    for name, current_model in models.items():

        scores = cross_validate(
            current_model,
            X,
            y,
            cv=kfold,
            scoring=scoring,
            return_train_score=True,
        )

        # -----------------------------------
        # Mean Values
        # -----------------------------------

        train_accuracy_mean = scores["train_accuracy"].mean()
        test_accuracy_mean = scores["test_accuracy"].mean()

        train_f1_mean = scores["train_f1"].mean()
        test_f1_mean = scores["test_f1"].mean()

        fit_time_mean = scores["fit_time"].mean()
        fit_time_std = scores["fit_time"].std()

        # -----------------------------------
        # Prediction Time
        # -----------------------------------

        predict_times = []

        for train_index, test_index in kfold.split(X, y):

            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]

            fold_model = clone(current_model)

            fold_model.fit(
                X_train,
                y_train,
            )

            # Warm-up
            fold_model.predict(X_test.iloc[:1])

            start = time.perf_counter()

            fold_model.predict(X_test)

            elapsed = time.perf_counter() - start

            predict_times.append(elapsed)

        predict_time_mean = float(np.mean(predict_times))
        predict_time_std = float(np.std(predict_times))

        # -----------------------------------
        # Gaps
        # -----------------------------------

        accuracy_gap = train_accuracy_mean - test_accuracy_mean
        f1_gap = train_f1_mean - test_f1_mean

        # -----------------------------------
        # Result
        # -----------------------------------

        results[name] = {
            "Accuracy": (
                f"Train: {train_accuracy_mean:.3f} ± "
                f"{scores['train_accuracy'].std():.3f}\n"
                f"Test: {test_accuracy_mean:.3f} ± "
                f"{scores['test_accuracy'].std():.3f}\n"
                f"Gap: {accuracy_gap:.3f}"
            ),
            "F1 Weighted": (
                f"Train: {train_f1_mean:.3f} ± "
                f"{scores['train_f1'].std():.3f}\n"
                f"Test: {test_f1_mean:.3f} ± "
                f"{scores['test_f1'].std():.3f}\n"
                f"Gap: {f1_gap:.3f}"
            ),
            "_Test Accuracy Mean": test_accuracy_mean,
            "_Test F1 Mean": test_f1_mean,
            "_Fit Time Mean": fit_time_mean,
            "_Fit Time Std": fit_time_std,
            "_Predict Time Mean": predict_time_mean,
            "_Predict Time Std": predict_time_std,
        }

    result = pd.DataFrame(results).T
    result.index.name = "Model"

    # -----------------------------------
    # Sort by Test F1 / Fit Time
    # -----------------------------------

    result = result.sort_values(
        by=[
            "_Test F1 Mean",
            "_Fit Time Mean",
        ],
        ascending=[
            False,
            True,
        ],
    )

    # -----------------------------------
    # Best Model
    # -----------------------------------

    best_test_accuracy = float(result.iloc[0]["_Test Accuracy Mean"])
    best_test_f1 = float(result.iloc[0]["_Test F1 Mean"])
    best_fit_time = float(result.iloc[0]["_Fit Time Mean"])
    best_predict_time = float(result.iloc[0]["_Predict Time Mean"])

    # -----------------------------------
    # Test Accuracy Loss
    # -----------------------------------

    accuracy_values = result["_Test Accuracy Mean"].astype(float)

    accuracy_loss = best_test_accuracy - accuracy_values
    accuracy_loss_percent = accuracy_loss / best_test_accuracy * 100

    result["Test Accuracy Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            accuracy_loss_percent,
            accuracy_loss,
        )
    ]

    # -----------------------------------
    # Test F1 Loss
    # -----------------------------------

    f1_values = result["_Test F1 Mean"].astype(float)

    f1_loss = best_test_f1 - f1_values
    f1_loss_percent = f1_loss / best_test_f1 * 100

    result["Test F1 Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            f1_loss_percent,
            f1_loss,
        )
    ]

    # -----------------------------------
    # Fit / Predict Time
    # -----------------------------------

    result["Fit / Predict Time (s)"] = [
        (
            f"Fit: {fit_mean:.3f} ± "
            f"{fit_std:.3f}\n"
            f"Predict: {predict_mean:.3f} ± "
            f"{predict_std:.3f}"
        )
        for (
            fit_mean,
            fit_std,
            predict_mean,
            predict_std,
        ) in zip(
            result["_Fit Time Mean"],
            result["_Fit Time Std"],
            result["_Predict Time Mean"],
            result["_Predict Time Std"],
        )
    ]

    # -----------------------------------
    # Speed vs Best
    # -----------------------------------

    speed_values = []

    for index in result.index:

        fit_time = float(result.loc[index, "_Fit Time Mean"])

        predict_time = float(result.loc[index, "_Predict Time Mean"])

        if fit_time <= best_fit_time:
            fit_speed = best_fit_time / fit_time
        else:
            fit_speed = -(fit_time / best_fit_time)

        if predict_time <= best_predict_time:
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
            "F1 Weighted",
            "Test Accuracy Loss",
            "Test F1 Loss",
            "Fit / Predict Time (s)",
            "Speed vs Best",
        ]
    ]

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = TuneModelTemplate(
        dataset_name=dataset_name,
        results=result,
        best_params=best_params,
    )

    return template.save(output=output)
