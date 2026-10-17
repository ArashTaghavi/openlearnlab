from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict, cross_validate

from ..reporting.templates import RefineAndEvaluateTemplate


def refine_and_evaluate(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    dataset_name: str,
    output: Path,
    random_state: int,
) -> Path:
    if len(df) < 12:
        raise ValueError("Refinement requires at least 12 observations.")

    features = list(features)

    if not features:
        raise ValueError("At least one feature is required.")

    if target in features:
        raise ValueError("Target must not be included in features.")

    if not df.index.is_unique:
        raise ValueError("Refinement requires a unique DataFrame index.")

    if not np.all(np.isfinite(df[target].to_numpy(dtype=float))):
        raise ValueError("Refinement requires finite target values.")

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # -----------------------------------
    # Detect High-Error Records
    # -----------------------------------

    oof_pred = np.asarray(
        cross_val_predict(
            clone(model),
            df[features],
            df[target],
            cv=cv,
        ),
        dtype=float,
    )

    if not np.all(np.isfinite(oof_pred)):
        raise ValueError("Model produced non-finite OOF predictions.")

    y_true = df[target].to_numpy(dtype=float)
    abs_errors = np.abs(y_true - oof_pred)

    q1 = float(np.quantile(abs_errors, 0.25))
    q3 = float(np.quantile(abs_errors, 0.75))

    threshold = q3 + 1.5 * (q3 - q1)
    high_error_mask = abs_errors > threshold

    refined_df = df.iloc[np.flatnonzero(~high_error_mask)].copy()
    removed_df = df.iloc[np.flatnonzero(high_error_mask)].copy()

    if len(refined_df) < 5:
        raise ValueError("Too few observations remain after refinement.")

    # -----------------------------------
    # Evaluate Exactly Like compare_models
    # -----------------------------------

    def evaluate_dataset(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        scores = cross_validate(
            clone(model),
            data[features],
            data[target],
            cv=KFold(
                n_splits=5,
                shuffle=True,
                random_state=random_state,
            ),
            scoring={
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
            },
            return_train_score=True,
        )

        return pd.DataFrame(
            {
                "Train R²": scores["train_r2"],
                "Test R²": scores["test_r2"],
                "Train RMSE": -scores["train_rmse"],
                "Test RMSE": -scores["test_rmse"],
                "Train MAE": -scores["train_mae"],
                "Test MAE": -scores["test_mae"],
            }
        )

    # -----------------------------------
    # Two Scenarios
    # -----------------------------------

    scenario_definitions = [
        {
            "name": "Original Model",
            "training": "Original Dataset",
            "evaluation": "5-Fold CV",
            "data": df,
        },
        {
            "name": "Refined Model",
            "training": "Refined Dataset",
            "evaluation": "5-Fold CV",
            "data": refined_df,
        },
    ]

    comparison_rows = []
    scenario_results = []
    fold_results = []

    for scenario in scenario_definitions:
        data = scenario["data"]
        folds = evaluate_dataset(data)

        folds.insert(0, "Scenario", scenario["name"])
        folds.insert(1, "Fold", np.arange(1, len(folds) + 1))

        fold_results.append(folds)

        # Out-of-fold predictions for diagnostic plots
        predictions = np.asarray(
            cross_val_predict(
                clone(model),
                data[features],
                data[target],
                cv=KFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=random_state,
                ),
            ),
            dtype=float,
        )

        actual = data[target].to_numpy(dtype=float)
        residuals = actual - predictions

        comparison_rows.append(
            {
                "Scenario": scenario["name"],
                "Training Data": scenario["training"],
                "Evaluation Data": scenario["evaluation"],
                "Test Records": len(data),
                "R²": (
                    float(
                        1
                        - np.sum(residuals**2) / np.sum((actual - np.mean(actual)) ** 2)
                    )
                    if np.sum((actual - np.mean(actual)) ** 2) > 0
                    else float("nan")
                ),
                "RMSE": float(np.sqrt(np.mean(residuals**2))),
                "MAE": float(np.mean(np.abs(residuals))),
            }
        )

        scenario_results.append(
            {
                "name": scenario["name"],
                "training": scenario["training"],
                "evaluation": scenario["evaluation"],
                "y_true": actual.copy(),
                "y_pred": predictions.copy(),
                "fold_metrics": folds.copy(),
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    all_fold_results = pd.concat(
        fold_results,
        ignore_index=True,
    )

    # -----------------------------------
    # Export
    # -----------------------------------

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    refined_df.to_csv(
        output / "refined_train.csv",
        index=True,
    )

    removed_df.to_csv(
        output / "removed_train_records.csv",
        index=True,
    )

    comparison.to_csv(
        output / "refinement_comparison.csv",
        index=False,
    )

    all_fold_results.to_csv(
        output / "refinement_fold_metrics.csv",
        index=False,
    )

    # -----------------------------------
    # Report
    # -----------------------------------

    template = RefineAndEvaluateTemplate(
        dataset_name=dataset_name,
        comparison=comparison,
        scenarios=scenario_results,
        n_samples=len(df),
        n_train=len(df),
        n_test=len(df),
        n_removed=len(removed_df),
        n_retained=len(refined_df),
        n_typical_test=len(refined_df),
        n_folds=cv.get_n_splits(),
    )

    return template.save(output=output)
