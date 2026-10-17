from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

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
    if len(df) < 5:
        raise ValueError("Diagnostics requires at least 5 observations.")

    features = list(features)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    X = df[features]
    y = df[target]

    # -----------------------------------
    # Out-of-Fold Predictions
    # -----------------------------------

    y_pred = cross_val_predict(
        clone(model),
        X,
        y,
        cv=cv,
    )

    y_true = y.to_numpy(dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Diagnostics requires finite target values and predictions.")

    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    # -----------------------------------
    # Overall Metrics
    # -----------------------------------

    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(abs_residuals))

    # -----------------------------------
    # Automatic High Error Detection
    # -----------------------------------

    q1 = float(np.quantile(abs_residuals, 0.25))
    q3 = float(np.quantile(abs_residuals, 0.75))

    iqr = q3 - q1
    error_threshold = q3 + 1.5 * iqr

    high_error_mask = abs_residuals > error_threshold
    high_error_positions = np.flatnonzero(high_error_mask)

    # -----------------------------------
    # Error Contribution
    # -----------------------------------

    squared_errors = residuals**2
    total_sse = float(np.sum(squared_errors))

    error_records = df[features].copy()

    error_records["Actual"] = y_true
    error_records["Predicted"] = y_pred
    error_records["Residual"] = residuals
    error_records["Absolute Error"] = abs_residuals
    error_records["Squared Error"] = squared_errors

    if total_sse > 0:
        error_records["SSE Share (%)"] = 100 * squared_errors / total_sse
    else:
        error_records["SSE Share (%)"] = 0.0

    critical_errors = error_records.iloc[high_error_positions].copy()

    critical_errors = critical_errors.sort_values(
        "Squared Error",
        ascending=False,
        kind="stable",
    )

    critical_errors["Cumulative SSE (%)"] = critical_errors["SSE Share (%)"].cumsum()

    high_error_sse_share = (
        float(np.sum(squared_errors[high_error_mask]) / total_sse)
        if total_sse > 0
        else 0.0
    )

    # -----------------------------------
    # Diagnostic Insights
    # -----------------------------------

    insights: list[tuple[str, str]] = []

    mean_error = float(np.mean(residuals))

    if rmse > 0 and abs(mean_error) > 0.1 * rmse:
        direction = "underestimates" if mean_error > 0 else "overestimates"

        insights.append(
            (
                "Possible Prediction Bias",
                f"The model tends to {direction} the target on average.",
            )
        )

    if np.std(y_pred) > 0 and np.std(residuals) > 0:
        correlation = float(np.corrcoef(y_pred, residuals)[0, 1])

        if abs(correlation) > 0.3:
            insights.append(
                (
                    "Residual Pattern",
                    (
                        "Residuals show a linear relationship with predictions. "
                        "The model may have systematic prediction errors."
                    ),
                )
            )

    if np.any(high_error_mask) and high_error_sse_share > 0.5:
        insights.append(
            (
                "Concentrated Prediction Errors",
                (
                    "A small group of observations contributes "
                    "more than half of the total squared error. "
                    "Inspect these observations for unusual patterns."
                ),
            )
        )

    if np.std(y_pred) > 0 and np.std(abs_residuals) > 0:
        error_corr = float(np.corrcoef(y_pred, abs_residuals)[0, 1])

        if error_corr > 0.3:
            insights.append(
                (
                    "Increasing Error Magnitude",
                    (
                        "Larger predicted values tend to have larger errors. "
                        "Error variance may not be constant."
                    ),
                )
            )

    if not insights:
        insights.append(
            (
                "No Strong Diagnostic Signals",
                (
                    "The selected checks did not identify strong patterns. "
                    "This does not guarantee that the model is free of problems."
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
        residuals=residuals,
        r2=r2,
        rmse=rmse,
        mae=mae,
        insights=insights,
        n_samples=len(df),
        n_features=len(features),
        n_folds=cv.get_n_splits(),
        critical_errors=critical_errors,
    )

    return template.save(output=output)
