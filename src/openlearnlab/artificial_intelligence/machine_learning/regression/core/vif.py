import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    if features is None:
        features = df.select_dtypes(include="number").columns.to_list()

    if not features:
        raise ValueError("No numeric features found.")

    X_vif = df[features].astype(float).copy()

    if not np.isfinite(X_vif.to_numpy()).all():
        raise ValueError("Features contain NaN or infinite values.")

    # Add intercept
    X_vif.insert(0, "const", 1.0)

    X_values = X_vif.to_numpy(dtype=float)

    vif_df = pd.DataFrame(
        {
            "Feature": features,
            "VIF": [
                variance_inflation_factor(
                    X_values,
                    X_vif.columns.get_loc(feature),
                )
                for feature in features
            ],
        }
    )

    return vif_df.sort_values(
        "VIF",
        ascending=False,
    ).reset_index(drop=True)
