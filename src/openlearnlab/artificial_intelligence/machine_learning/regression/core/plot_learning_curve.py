from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, learning_curve

from ..reporting.templates import LearningCurveTemplate


def plot_learning_curve(
    df: pd.DataFrame,
    target: str,
    dataset_name: str,
    output: Path,
    features,
    cv: int = 5,
    random_state: int = 44,
    model=None,
    train_sizes=None,
) -> Path:

    if model is None:
        model = LinearRegression()

    if train_sizes is None:
        train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

    X = df[features]
    y = df[target]

    # -----------------------------------
    # Cross Validation
    # -----------------------------------

    kfold = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    # -----------------------------------
    # Learning Curve
    # -----------------------------------

    sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=kfold,
        train_sizes=train_sizes,
        scoring="r2",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    r2_gap = train_mean - test_mean

    # -----------------------------------
    # Results
    # -----------------------------------

    result = pd.DataFrame(
        {
            "Train Size": sizes,
            "Train R2": [
                f"{mean:.4f} ± {std:.4f}" for mean, std in zip(train_mean, train_std)
            ],
            "Test R2": [
                f"{mean:.4f} ± {std:.4f}" for mean, std in zip(test_mean, test_std)
            ],
            "R2 Gap": r2_gap,
        }
    )

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = LearningCurveTemplate(
        dataset_name=dataset_name,
        results=result,
        sizes=sizes,
        train_mean=train_mean,
        train_std=train_std,
        test_mean=test_mean,
        test_std=test_std,
    )

    return template.save(output=output)
