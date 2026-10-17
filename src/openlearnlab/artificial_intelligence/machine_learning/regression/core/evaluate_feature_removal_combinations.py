import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

from ..reporting.templates import FeatureRemovalCombinationsTemplate


def evaluate_feature_removal_combinations(
    df: pd.DataFrame,
    features: list,
    candidates: list,
    target: str,
    model,
    dataset_name: str,
    output: Path,
    random_state: int,
) -> Path:
    features = list(features)
    candidates = list(candidates)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    y = df[target]

    if model is None:
        model = LinearRegression()

    # -----------------------------------
    # Removal Combinations
    # -----------------------------------

    removal_combinations = [()]

    for n_remove in range(1, len(candidates) + 1):
        removal_combinations.extend(combinations(candidates, n_remove))

    results = []

    # -----------------------------------
    # Evaluate Combinations
    # -----------------------------------

    for removed_features in removal_combinations:

        current_features = [
            feature for feature in features if feature not in removed_features
        ]

        if not current_features:
            continue

        X = df[current_features]

        scores = cross_validate(
            clone(model),
            X,
            y,
            cv=cv,
            scoring={
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
            },
            return_train_score=True,
        )

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

        for train_index, test_index in cv.split(X):

            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]

            fold_model = clone(model)

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

        results.append(
            {
                "Removed Count": len(removed_features),
                "Removed Features": (
                    ", ".join(removed_features) if removed_features else "None"
                ),
                "Remaining Features": ", ".join(current_features),
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
        )

    # -----------------------------------
    # Results DataFrame
    # -----------------------------------

    result = pd.DataFrame(results)

    # -----------------------------------
    # Sort
    # Best R2 first
    # More removed features second
    # -----------------------------------

    result = result.sort_values(
        by=[
            "_Test R2 Mean",
            "Removed Count",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    # -----------------------------------
    # Best Row
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

    for _, row in result.iterrows():

        fit_time = float(row["_Fit Time Mean"])
        predict_time = float(row["_Predict Time Mean"])

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
            "Removed Count",
            "Removed Features",
            "Remaining Features",
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

    template = FeatureRemovalCombinationsTemplate(
        dataset_name=dataset_name,
        results=result,
    )

    return template.save(output=output)
