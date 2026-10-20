from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
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

    if X.empty or X.shape[1] == 0:
        raise ValueError("At least one feature is required.")

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if y.nunique() != 2:
        raise ValueError("This analysis currently supports binary classification only.")

    if y.value_counts().min() < 2:
        raise ValueError("Each class must have at least 2 records.")

    if model is None:
        model = LogisticRegression(max_iter=1000)

    classes = np.unique(y.to_numpy())
    positive_class = classes[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both training and test sets must contain both classes.")

    model.fit(X_train, y_train)

    # -----------------------------------
    # Classification Scoring
    # -----------------------------------

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            pos_label=positive_class,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            pos_label=positive_class,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            pos_label=positive_class,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
    }

    # -----------------------------------
    # Permutation Importance
    # -----------------------------------

    permutation_results = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )

    # -----------------------------------
    # Importance DataFrames
    # -----------------------------------

    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC AUC",
    }

    importance_frames = {}
    formatted_frames = []

    for metric, name in metric_names.items():
        importance = permutation_results[metric]

        column_name = f"{name} Decrease"

        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_test.columns,
                    column_name: importance.importances_mean,
                    "std": importance.importances_std,
                }
            )
            .sort_values(
                by=column_name,
                ascending=False,
            )
            .reset_index(drop=True)
        )

        importance_frames[metric] = importance_df

        # -----------------------------------
        # Format: Mean ± Std
        # -----------------------------------

        formatted_df = importance_df.copy()

        formatted_df[column_name] = (
            formatted_df[column_name].map("{:.4f}".format)
            + " ± "
            + formatted_df["std"].map("{:.4f}".format)
        )

        formatted_df = formatted_df.drop(columns="std")

        formatted_frames.append(formatted_df)

    # -----------------------------------
    # Final Results
    # -----------------------------------

    result = formatted_frames[0]

    for frame in formatted_frames[1:]:
        result = result.merge(
            frame,
            on="feature",
        )

    # Sort by F1 importance, then ROC AUC importance
    result = (
        result.set_index("feature")
        .loc[importance_frames["f1"]["feature"]]
        .reset_index()
    )

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = PermutationImportanceTemplate(
        dataset_name=dataset_name,
        results=result,
        importance_accuracy=importance_frames["accuracy"],
        importance_precision=importance_frames["precision"],
        importance_recall=importance_frames["recall"],
        importance_f1=importance_frames["f1"],
        importance_roc_auc=importance_frames["roc_auc"],
    )

    return template.save(output=output)
