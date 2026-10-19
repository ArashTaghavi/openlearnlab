from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

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
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    X = df[features]
    y = df[target]

    # Out-of-fold predictions on the supplied training data.
    y_pred = cross_val_predict(clone(model), X, y, cv=cv)
    y_true = y.to_numpy(dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Diagnostics requires finite target values and predictions.")

    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    squared_errors = residuals**2

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # Detect outliers ONLY in the target, not in individual input features.
    q1 = float(np.quantile(y_true, 0.25))
    q3 = float(np.quantile(y_true, 0.75))
    target_iqr = q3 - q1
    target_lower_bound = q1 - 1.5 * target_iqr
    target_upper_bound = q3 + 1.5 * target_iqr
    outlier_mask = (y_true < target_lower_bound) | (y_true > target_upper_bound)

    # High-error detection is separate from target-outlier detection.
    high_error_percentage = 50.0
    error_threshold = mae * (1 + high_error_percentage / 100)
    high_error_mask = abs_residuals > error_threshold
    if mae == 0:
        high_error_mask[:] = False

    overlapping_mask = outlier_mask & high_error_mask
    error_to_mae_ratio = np.divide(
        abs_residuals,
        mae,
        out=np.zeros_like(abs_residuals),
        where=mae > 0,
    )

    total_absolute_error = float(np.sum(abs_residuals))
    total_sse = float(np.sum(squared_errors))
    mae_contribution = (
        100 * abs_residuals / total_absolute_error
        if total_absolute_error > 0
        else np.zeros(len(df))
    )
    mse_contribution = (
        100 * squared_errors / total_sse if total_sse > 0 else np.zeros(len(df))
    )

    high_error_mae_share = float(np.sum(mae_contribution[high_error_mask]))
    high_error_mse_share = float(np.sum(mse_contribution[high_error_mask]))
    outlier_mae_share = float(np.sum(mae_contribution[outlier_mask]))
    outlier_mse_share = float(np.sum(mse_contribution[outlier_mask]))

    error_records = df[features].copy()
    error_records["Actual"] = y_true
    error_records["Predicted"] = y_pred
    error_records["Residual"] = residuals
    error_records["Absolute Error"] = abs_residuals
    error_records["Squared Error"] = squared_errors
    error_records["Error / MAE"] = error_to_mae_ratio
    error_records["MAE Share (%)"] = mae_contribution
    error_records["SSE Share (%)"] = mse_contribution
    error_records["Is Outlier"] = outlier_mask
    error_records["Is High Error"] = high_error_mask
    error_records["Is Overlapping"] = overlapping_mask

    outlier_records = error_records.loc[outlier_mask].copy()
    outlier_records = outlier_records.sort_values(
        "Squared Error", ascending=False, kind="stable"
    )
    high_error_records = error_records.loc[high_error_mask].copy()
    high_error_records = high_error_records.sort_values(
        "Squared Error", ascending=False, kind="stable"
    )
    high_error_records["Cumulative SSE (%)"] = high_error_records[
        "SSE Share (%)"
    ].cumsum()
    overlapping_records = error_records.loc[overlapping_mask].copy()
    overlapping_records = overlapping_records.sort_values(
        "Squared Error", ascending=False, kind="stable"
    )

    target_error_analysis = pd.DataFrame()
    if len(np.unique(y_true)) >= 2:
        target_bins = pd.qcut(
            y_true, q=min(5, len(np.unique(y_true))), duplicates="drop"
        )
        target_error_analysis = (
            pd.DataFrame(
                {
                    "Target Range": target_bins,
                    "Absolute Error": abs_residuals,
                    "Squared Error": squared_errors,
                }
            )
            .groupby("Target Range", observed=True)
            .agg(
                Observations=("Absolute Error", "size"),
                MAE=("Absolute Error", "mean"),
                MSE=("Squared Error", "mean"),
            )
            .reset_index()
        )
        target_error_analysis["RMSE"] = np.sqrt(target_error_analysis["MSE"])

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    feature_error_analysis = []
    for feature in numeric_features:
        values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(values) & np.isfinite(abs_residuals)
        if (
            valid_mask.sum() < 3
            or np.std(values[valid_mask]) == 0
            or np.std(abs_residuals[valid_mask]) == 0
        ):
            continue
        correlation = float(
            np.corrcoef(values[valid_mask], abs_residuals[valid_mask])[0, 1]
        )
        feature_error_analysis.append(
            {
                "Feature": feature,
                "Error Correlation": correlation,
                "Absolute Correlation": abs(correlation),
            }
        )
    feature_error_analysis = pd.DataFrame(
        feature_error_analysis,
        columns=["Feature", "Error Correlation", "Absolute Correlation"],
    ).sort_values("Absolute Correlation", ascending=False, kind="stable")

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
                    "Residuals show a linear relationship with predictions. "
                    "The model may have systematic prediction errors.",
                )
            )
    if np.any(high_error_mask) and high_error_mse_share > 50:
        insights.append(
            (
                "Concentrated Prediction Errors",
                f"{int(high_error_mask.sum())} high-error observations contribute "
                f"{high_error_mse_share:.1f}% of the total squared error. "
                "Inspect these observations.",
            )
        )
    if np.any(outlier_mask):
        insights.append(
            (
                "Target Outliers Detected",
                f"{int(outlier_mask.sum())} observations have target values "
                "outside the target's 1.5 × IQR bounds.",
            )
        )
    if np.any(overlapping_mask):
        insights.append(
            (
                "Target Outliers with High Prediction Error",
                f"{int(overlapping_mask.sum())} observations are both target "
                "outliers and high-error predictions. Inspect these records.",
            )
        )
    if np.std(y_pred) > 0 and np.std(abs_residuals) > 0:
        error_corr = float(np.corrcoef(y_pred, abs_residuals)[0, 1])
        if error_corr > 0.3:
            insights.append(
                (
                    "Increasing Error Magnitude",
                    "Larger predicted values tend to have larger errors. "
                    "Error variance may not be constant.",
                )
            )
    if not feature_error_analysis.empty:
        top_feature = feature_error_analysis.iloc[0]
        if top_feature["Absolute Correlation"] > 0.3:
            insights.append(
                (
                    "Feature-Dependent Prediction Error",
                    f"Prediction error magnitude is associated with "
                    f"'{top_feature['Feature']}' "
                    f"(correlation: {top_feature['Error Correlation']:.2f}).",
                )
            )
    if not insights:
        insights.append(
            (
                "No Strong Diagnostic Signals",
                "The selected checks did not identify strong patterns. "
                "This does not guarantee that the model is free of problems.",
            )
        )

    template = DiagnosticsTemplate(
        dataset_name=dataset_name,
        y_true=y_true,
        y_pred=y_pred,
        residuals=residuals,
        r2=r2,
        rmse=rmse,
        mae=mae,
        mse=mse,
        insights=insights,
        n_samples=len(df),
        n_features=len(features),
        n_folds=cv.get_n_splits(),
        critical_errors=high_error_records,
        outlier_records=outlier_records,
        high_error_records=high_error_records,
        overlapping_records=overlapping_records,
        error_records=error_records,
        target_error_analysis=target_error_analysis,
        feature_error_analysis=feature_error_analysis,
        error_threshold=error_threshold,
        high_error_percentage=high_error_percentage,
        high_error_mae_share=high_error_mae_share,
        high_error_mse_share=high_error_mse_share,
        outlier_mae_share=outlier_mae_share,
        outlier_mse_share=outlier_mse_share,
        model_name=model_name,
    )
    return template.save(output=output, suffix_file_name=model_name)
