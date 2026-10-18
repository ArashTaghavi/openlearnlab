from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..base.reporting.templates import CorrelationTemplate


def correlation(
    df: pd.DataFrame,
    dataset_name: str,
    output: Path,
    threshold: float,
    features: list[str],
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
) -> Path:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Method must be pearson, spearman or kendall.")

    # Correlation Matrix
    corr = df[features].corr(method=method)
    corr_values = corr.to_numpy(dtype=np.float64)

    # High Correlation Pairs
    pairs = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            value = float(corr_values[i, j])

            if np.isnan(value) or abs(value) < threshold:
                continue

            pairs.append(
                {
                    "Feature 1": features[i],
                    "Feature 2": features[j],
                    "Correlation": round(value, 4),
                    "Abs Correlation": round(abs(value), 4),
                    "Direction": "Positive" if value > 0 else "Negative",
                }
            )

    pairs_df = (
        pd.DataFrame(
            pairs,
            columns=[
                "Feature 1",
                "Feature 2",
                "Correlation",
                "Abs Correlation",
                "Direction",
            ],
        )
        .sort_values(
            by="Abs Correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # Suspicious Features
    suspicious_features = set(pairs_df["Feature 1"]) | set(pairs_df["Feature 2"])

    template = CorrelationTemplate(
        dataset_name=dataset_name,
        correlation_matrix=corr,
        high_correlation_pairs=pairs_df,
        suspicious_features=suspicious_features,
        threshold=threshold,
        method=method,
    )

    return template.save(output=output)
