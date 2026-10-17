import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from ..reporting.templates import ModelComparisonTemplate


def compare_models(
    df: pd.DataFrame,
    features,
    target: str,
    models: dict,
    dataset_name: str,
    output: Path,
    random_state,
):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    X = df[features]
    y = df[target]

    results = {}

    for name, model in models.items():

        # -----------------------------------
        # Cross Validation
        # -----------------------------------

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
            },
            return_train_score=True,
            return_estimator=True,
        )

        train_r2_mean = scores["train_r2"].mean()
        test_r2_mean = scores["test_r2"].mean()

        train_rmse_mean = -scores["train_rmse"].mean()
        test_rmse_mean = -scores["test_rmse"].mean()

        fit_time_mean = scores["fit_time"].mean()

        # -----------------------------------
        # Prediction Time
        # Reuse fitted CV estimators
        # -----------------------------------

        predict_times = []

        for fold_model, (_, test_index) in zip(
            scores["estimator"],
            cv.split(X),
        ):
            X_test = X.iloc[test_index]

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

        r2_gap = train_r2_mean - test_r2_mean
        rmse_gap = test_rmse_mean - train_rmse_mean

        results[name] = {
            "R2": (
                f"Train: {train_r2_mean:.3f} ± {scores['train_r2'].std():.3f}\n"
                f"Test: {test_r2_mean:.3f} ± {scores['test_r2'].std():.3f}\n"
                f"Gap: {r2_gap:.3f}"
            ),
            "RMSE": (
                f"Train: {train_rmse_mean:.3f} ± {scores['train_rmse'].std():.3f}\n"
                f"Test: {test_rmse_mean:.3f} ± {scores['test_rmse'].std():.3f}\n"
                f"Gap: {rmse_gap:.3f}"
            ),
            # Internal values
            "_Test R2 Mean": test_r2_mean,
            "_Test RMSE Mean": test_rmse_mean,
            "_Fit Time Mean": fit_time_mean,
            "_Fit Time Std": scores["fit_time"].std(),
            "_Predict Time Mean": predict_time_mean,
            "_Predict Time Std": predict_time_std,
        }

    # -----------------------------------
    # Results DataFrame
    # -----------------------------------

    result = pd.DataFrame(results).T

    result.index.name = "Model"

    result = result.sort_values(
        by=[
            "_Test R2 Mean",
            "_Fit Time Mean",
        ],
        ascending=[
            False,
            True,
        ],
    )

    # -----------------------------------
    # Best Model = Highest Test R2
    # -----------------------------------

    best_test_r2 = float(result.iloc[0]["_Test R2 Mean"])
    best_test_rmse = float(result.iloc[0]["_Test RMSE Mean"])
    best_fit_time = float(result.iloc[0]["_Fit Time Mean"])
    best_predict_time = float(result.iloc[0]["_Predict Time Mean"])

    # -----------------------------------
    # Test R2 Relative Loss
    # -----------------------------------

    r2_values = result["_Test R2 Mean"].astype(float)

    r2_loss = best_test_r2 - r2_values
    r2_loss_percent = r2_loss / best_test_r2 * 100

    result["Test R2 Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            r2_loss_percent,
            r2_loss,
        )
    ]

    # -----------------------------------
    # Test RMSE Relative Loss
    # -----------------------------------

    rmse_values = result["_Test RMSE Mean"].astype(float)

    rmse_loss = rmse_values - best_test_rmse
    rmse_loss_percent = rmse_loss / best_test_rmse * 100

    result["Test RMSE Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            rmse_loss_percent,
            rmse_loss,
        )
    ]

    # -----------------------------------
    # Fit / Predict Time
    # -----------------------------------

    time_values = []

    for index in result.index:

        fit_mean = cast(float, result.at[index, "_Fit Time Mean"])
        fit_std = cast(float, result.at[index, "_Fit Time Std"])

        predict_mean = cast(float, result.at[index, "_Predict Time Mean"])
        predict_std = cast(float, result.at[index, "_Predict Time Std"])

        time_values.append(
            f"Fit: {fit_mean:.3f} ± {fit_std:.3f}\n"
            f"Predict: {predict_mean:.3f} ± {predict_std:.3f}"
        )

    result["Fit / Predict Time (s)"] = time_values

    # -----------------------------------
    # Fit / Predict Speed vs Best
    # -----------------------------------

    speed_values = []

    for index in result.index:

        fit_time = cast(float, result.at[index, "_Fit Time Mean"])
        predict_time = cast(float, result.at[index, "_Predict Time Mean"])

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
    # Final Column Order
    # -----------------------------------

    result = result[
        [
            "R2",
            "RMSE",
            "Test R2 Loss",
            "Test RMSE Loss",
            "Fit / Predict Time (s)",
            "Speed vs Best",
        ]
    ]

    template = ModelComparisonTemplate(
        dataset_name=dataset_name,
        results=result,
        best_model=str(result.index[0]),
        n_samples=len(df),
        n_features=len(features),
        n_folds=cv.get_n_splits(),
    )

    return template.save(output=output)
