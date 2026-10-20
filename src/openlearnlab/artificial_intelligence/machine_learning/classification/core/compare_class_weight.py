from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from ..reporting.templates import CompareClassWeightTemplate


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = pd.Series(y).unique()

        class_counts = pd.Series(y).value_counts()

        scale_pos_weight = class_counts.loc[0] / class_counts.loc[1]

        self.estimator_ = clone(self.estimator)

        self.estimator_.set_params(
            scale_pos_weight=scale_pos_weight,
        )

        self.estimator_.fit(X, y)

        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def compare_class_weight(
    df: pd.DataFrame,
    model,
    features: list,
    target: str,
    dataset_name: str,
    output: Path,
    cv: int = 5,
    random_state: int = 42,
) -> Path:

    params = model.get_params()

    if "class_weight" in params:
        weight_parameter = "class_weight"

    elif "scale_pos_weight" in params:
        weight_parameter = "scale_pos_weight"

    else:
        raise ValueError(
            f"{model.__class__.__name__} does not support "
            "class_weight or scale_pos_weight."
        )

    X = df[features]
    y = df[target]

    # -----------------------------------
    # Class Distribution
    # -----------------------------------

    class_distribution = (
        y.value_counts().sort_index().rename_axis("Class").reset_index(name="Count")
    )

    class_distribution["Percentage"] = class_distribution["Count"] / len(y) * 100

    # -----------------------------------
    # Cross Validation
    # -----------------------------------

    kfold = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = []

    for class_weight in [None, "balanced"]:
        current_model = clone(model)

        if weight_parameter == "class_weight":
            current_model.set_params(
                class_weight=class_weight,
            )

        elif class_weight is None:
            current_model.set_params(
                scale_pos_weight=1,
            )

        else:
            current_model = BalancedXGBClassifier(
                estimator=current_model,
            )

        scores = cross_validate(
            current_model,
            X,
            y,
            cv=kfold,
            scoring=scoring,
            n_jobs=-1,
        )

        results.append(
            {
                "Class Weight": str(class_weight),
                "CV Accuracy": (
                    f"{scores['test_accuracy'].mean():.4f} ± "
                    f"{scores['test_accuracy'].std():.4f}"
                ),
                "CV Precision": (
                    f"{scores['test_precision'].mean():.4f} ± "
                    f"{scores['test_precision'].std():.4f}"
                ),
                "CV Recall": (
                    f"{scores['test_recall'].mean():.4f} ± "
                    f"{scores['test_recall'].std():.4f}"
                ),
                "CV F1": (
                    f"{scores['test_f1'].mean():.4f} ± "
                    f"{scores['test_f1'].std():.4f}"
                ),
                "CV ROC AUC": (
                    f"{scores['test_roc_auc'].mean():.4f} ± "
                    f"{scores['test_roc_auc'].std():.4f}"
                ),
            }
        )

    result = pd.DataFrame(results)

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = CompareClassWeightTemplate(
        dataset_name=dataset_name,
        results=result,
        class_distribution=class_distribution,
    )

    return template.save(output=output)
