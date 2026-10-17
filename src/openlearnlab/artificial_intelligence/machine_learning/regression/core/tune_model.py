import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

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
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
        }

    X = df[features]
    y = df[target]

    kfold = KFold(
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
        refit="r2",
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

        train_r2_mean = scores["train_r2"].mean()
        test_r2_mean = scores["test_r2"].mean()

        train_rmse_mean = -scores["train_rmse"].mean()
        test_rmse_mean = -scores["test_rmse"].mean()

        fit_time_mean = scores["fit_time"].mean()
        fit_time_std = scores["fit_time"].std()

        # -----------------------------------
        # Prediction Time
        # -----------------------------------

        predict_times = []

        for train_index, test_index in kfold.split(X):

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

        r2_gap = train_r2_mean - test_r2_mean
        rmse_gap = test_rmse_mean - train_rmse_mean

        # -----------------------------------
        # Result
        # -----------------------------------

        results[name] = {
            "R2": (
                f"Train: {train_r2_mean:.3f} ± "
                f"{scores['train_r2'].std():.3f}\n"
                f"Test: {test_r2_mean:.3f} ± "
                f"{scores['test_r2'].std():.3f}\n"
                f"Gap: {r2_gap:.3f}"
            ),
            "RMSE": (
                f"Train: {train_rmse_mean:.3f} ± "
                f"{scores['train_rmse'].std():.3f}\n"
                f"Test: {test_rmse_mean:.3f} ± "
                f"{scores['test_rmse'].std():.3f}\n"
                f"Gap: {rmse_gap:.3f}"
            ),
            "_Test R2 Mean": test_r2_mean,
            "_Test RMSE Mean": test_rmse_mean,
            "_Fit Time Mean": fit_time_mean,
            "_Fit Time Std": fit_time_std,
            "_Predict Time Mean": predict_time_mean,
            "_Predict Time Std": predict_time_std,
        }

    result = pd.DataFrame(results).T
    result.index.name = "Model"

    # -----------------------------------
    # Sort by Test R2 / Fit Time
    # -----------------------------------

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
    # Best Model
    # -----------------------------------

    best_test_r2 = float(result.iloc[0]["_Test R2 Mean"])
    best_test_rmse = float(result.iloc[0]["_Test RMSE Mean"])
    best_fit_time = float(result.iloc[0]["_Fit Time Mean"])
    best_predict_time = float(result.iloc[0]["_Predict Time Mean"])

    # -----------------------------------
    # Test R2 Loss
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
    # Test RMSE Loss
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
            "R2",
            "RMSE",
            "Test R2 Loss",
            "Test RMSE Loss",
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
