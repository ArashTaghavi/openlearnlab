from pathlib import Path

import pandas as pd
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import KFold, cross_validate

from ..reporting.templates import RegularizationComparisonTemplate


def compare_regularization(
    df: pd.DataFrame,
    target: str,
    dataset_name: str,
    output: Path,
    features=None,
    alphas=None,
    l1_ratio: float = 0.5,
    random_state: int = 44,
) -> Path:

    if features is None:
        features = df.drop(columns=[target]).columns.to_list()
    else:
        features = list(features)

    if alphas is None:
        alphas = [0.01, 0.1, 1, 10, 100]

    X = df[features]
    y = df[target]

    # -----------------------------------
    # Cross Validation
    # -----------------------------------

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    # -----------------------------------
    # Models
    # -----------------------------------

    models = {
        "Linear Regression": LinearRegression(),
    }

    for alpha in alphas:
        models[f"Ridge alpha={alpha}"] = Ridge(
            alpha=alpha,
        )

        models[f"Lasso alpha={alpha}"] = Lasso(
            alpha=alpha,
            max_iter=10000,
        )

        models[f"ElasticNet alpha={alpha}"] = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
        )

    # -----------------------------------
    # Evaluation
    # -----------------------------------

    results = []

    for name, model in models.items():

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
        )

        train_r2_mean = scores["train_r2"].mean()
        test_r2_mean = scores["test_r2"].mean()

        train_rmse_mean = -scores["train_rmse"].mean()
        test_rmse_mean = -scores["test_rmse"].mean()

        results.append(
            {
                "Model": name,
                "Train R2": (
                    f"{train_r2_mean:.4f} ± " f"{scores['train_r2'].std():.4f}"
                ),
                "Test R2": (f"{test_r2_mean:.4f} ± " f"{scores['test_r2'].std():.4f}"),
                "Train RMSE": (
                    f"{train_rmse_mean:.4f} ± " f"{scores['train_rmse'].std():.4f}"
                ),
                "Test RMSE": (
                    f"{test_rmse_mean:.4f} ± " f"{scores['test_rmse'].std():.4f}"
                ),
                "_Test R2 Mean": test_r2_mean,
            }
        )

    # -----------------------------------
    # Results
    # -----------------------------------

    result = (
        pd.DataFrame(results)
        .sort_values(
            by="_Test R2 Mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result = result.drop(columns=["_Test R2 Mean"])

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = RegularizationComparisonTemplate(
        dataset_name=dataset_name,
        results=result,
    )

    return template.save(output=output)
