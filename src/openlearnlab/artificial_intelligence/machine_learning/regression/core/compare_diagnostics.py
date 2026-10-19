from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict

from ..reporting.templates import CompareDiagnosticsTemplate


def compare_diagnostics(
    df: pd.DataFrame,
    features,
    target: str,
    models: dict,
    dataset_name: str,
    output: Path,
    random_state: int = 42,
) -> Path:
    features = list(features)

    if not models:
        raise ValueError("At least one model is required.")

    X = df[features]
    y = df[target]

    y_true = y.to_numpy(dtype=float)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for model_name, model in models.items():
        y_pred = np.asarray(
            cross_val_predict(
                clone(model),
                X,
                y,
                cv=cv,
            ),
            dtype=float,
        ).reshape(-1)

        if (
            len(y_pred) != len(y_true)
            or not np.isfinite(y_pred).all()
            or not np.isfinite(y_true).all()
        ):
            raise ValueError(f"Invalid predictions for model: {model_name}")

        absolute_errors = np.abs(y_true - y_pred)
        squared_errors = absolute_errors**2

        mae = float(absolute_errors.mean())
        threshold = 1.5 * mae

        high_error_mask = absolute_errors > threshold

        if mae == 0:
            high_error_mask[:] = False

        n_high_error = int(high_error_mask.sum())

        total_absolute_error = float(absolute_errors.sum())
        total_squared_error = float(squared_errors.sum())

        absolute_share = (
            100 * absolute_errors[high_error_mask].sum() / total_absolute_error
            if total_absolute_error > 0
            else 0.0
        )

        squared_share = (
            100 * squared_errors[high_error_mask].sum() / total_squared_error
            if total_squared_error > 0
            else 0.0
        )

        rows.append(
            {
                "Model": model_name,
                "High-Error Records": n_high_error,
                "Dataset Share (%)": 100 * n_high_error / len(df),
                "Absolute Error Share (%)": absolute_share,
                "Squared Error Share (%)": squared_share,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        "High-Error Records",
        ascending=False,
        kind="stable",
    )

    template = CompareDiagnosticsTemplate(
        dataset_name=dataset_name,
        results=results,
    )

    return template.save(output=output)
