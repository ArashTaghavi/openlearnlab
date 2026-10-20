from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve

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
        model = LogisticRegression(max_iter=1000)

    if train_sizes is None:
        train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

    X = df[features]
    y = df[target]

    # -----------------------------------
    # Cross Validation
    # -----------------------------------

    kfold = StratifiedKFold(
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
        scoring="f1_weighted",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    f1_gap = train_mean - test_mean

    # -----------------------------------
    # Results
    # -----------------------------------


from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve

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
        model = LogisticRegression(max_iter=1000)

    if train_sizes is None:
        train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

    X = df[features]
    y = df[target]

    # -----------------------------------
    # Cross Validation
    # -----------------------------------

    kfold = StratifiedKFold(
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
        scoring="f1_weighted",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    f1_gap = train_mean - test_mean

    # -----------------------------------
    # Results
    # -----------------------------------

    result = pd.DataFrame(
        {
            "Train Size": sizes,
            "Train F1 Weighted": [
                f"{mean:.4f} ± {std:.4f}" for mean, std in zip(train_mean, train_std)
            ],
            "Test F1 Weighted": [
                f"{mean:.4f} ± {std:.4f}" for mean, std in zip(test_mean, test_std)
            ],
            "F1 Gap": [f"{gap:.4f}" for gap in f1_gap],
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
