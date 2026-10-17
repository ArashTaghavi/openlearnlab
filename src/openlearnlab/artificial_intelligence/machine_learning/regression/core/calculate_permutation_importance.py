from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ..reporting.templates import PermutationImportanceTemplate


def calculate_permutation_importance(
    df: pd.DataFrame,
    target: str,
    model,
    dataset_name: str,
    output: Path,
    random_state: int,
) -> Path:

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    if model is None:
        model = LinearRegression()

    model.fit(X_train, y_train)

    # -----------------------------------
    # Permutation Importance - RMSE
    # -----------------------------------

    perm_rmse = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )

    importance_rmse = pd.DataFrame(
        {
            "feature": X_test.columns,
            "RMSE increase": perm_rmse["importances_mean"],
            "std": perm_rmse["importances_std"],
        }
    ).sort_values(
        by="RMSE increase",
        ascending=False,
    )

    # -----------------------------------
    # Permutation Importance - R2
    # -----------------------------------

    perm_r2 = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
    )

    importance_r2 = pd.DataFrame(
        {
            "feature": X_test.columns,
            "R2 decrease": perm_r2["importances_mean"],
            "std": perm_r2["importances_std"],
        }
    ).sort_values(
        by="R2 decrease",
        ascending=False,
    )

    # -----------------------------------
    # Format: Mean ± Std
    # -----------------------------------

    importance_r2_output = importance_r2.copy()

    importance_r2_output["R2 decrease"] = (
        importance_r2_output["R2 decrease"].map("{:.4f}".format)
        + " ± "
        + importance_r2_output["std"].map("{:.4f}".format)
    )

    importance_r2_output = importance_r2_output.drop(columns="std")

    importance_rmse_output = importance_rmse.copy()

    importance_rmse_output["RMSE increase"] = (
        importance_rmse_output["RMSE increase"].map("{:.4f}".format)
        + " ± "
        + importance_rmse_output["std"].map("{:.4f}".format)
    )

    importance_rmse_output = importance_rmse_output.drop(columns="std")

    # -----------------------------------
    # Final Results
    # -----------------------------------

    result = importance_r2_output.merge(
        importance_rmse_output,
        on="feature",
    )

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = PermutationImportanceTemplate(
        dataset_name=dataset_name,
        results=result,
        importance_r2=importance_r2,
        importance_rmse=importance_rmse,
    )

    return template.save(output=output)
