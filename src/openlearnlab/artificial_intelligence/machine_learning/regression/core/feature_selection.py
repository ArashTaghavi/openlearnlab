from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from ..reporting.templates import FeatureSelectionTemplate


def feature_selection(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    dataset_name: str,
    output: Path,
    scoring: str = "r2",
    cv: int = 5,
    random_state: int = 44,
) -> Path:

    # -----------------------------------
    # Features
    # -----------------------------------

    if features is None:
        features = df.drop(columns=[target]).columns.to_list()
    else:
        features = list(features)

    if model is None:
        model = LinearRegression()

    X = df[features]
    y = df[target]

    # -----------------------------------
    # Cross Validation
    # -----------------------------------

    cv_strategy = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    # -----------------------------------
    # Importance Getter
    # -----------------------------------

    importance_getter = "auto"

    if isinstance(model, Pipeline):
        final_step = model.steps[-1][0]
        final_model = model.steps[-1][1]

        if hasattr(final_model, "feature_importances_") or (
            "Forest" in type(final_model).__name__
        ):
            importance_getter = f"named_steps.{final_step}.feature_importances_"
        else:
            importance_getter = f"named_steps.{final_step}.coef_"

    # -----------------------------------
    # RFECV
    # -----------------------------------

    selector = RFECV(
        estimator=clone(model),
        step=1,
        cv=cv_strategy,
        scoring=scoring,
        importance_getter=importance_getter,
        n_jobs=-1,
    )

    selector.fit(X, y)

    # -----------------------------------
    # Feature Selection Results
    # -----------------------------------

    result = pd.DataFrame(
        {
            "Feature": features,
            "Selected": selector.support_,
            "Ranking": selector.ranking_,
        }
    )

    result = result.sort_values(
        by=["Selected", "Ranking"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # -----------------------------------
    # Summary
    # -----------------------------------

    summary = {
        "Total Features": len(features),
        "Selected Features": int(selector.n_features_),
        "Removed Features": (len(features) - int(selector.n_features_)),
        "Scoring": scoring,
        "CV Folds": cv,
    }

    # -----------------------------------
    # RFECV Performance
    # -----------------------------------

    cv_results = selector.cv_results_

    mean_scores = np.asarray(
        cv_results["mean_test_score"],
        dtype=float,
    )

    std_scores = np.asarray(
        cv_results["std_test_score"],
        dtype=float,
    )

    feature_counts = np.arange(
        1,
        len(features) + 1,
    )

    performance = pd.DataFrame(
        {
            "Number of Features": feature_counts,
            "Mean CV Score": mean_scores,
            "Std CV Score": std_scores,
        }
    )

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = FeatureSelectionTemplate(
        dataset_name=dataset_name,
        results=result,
        summary=summary,
        performance=performance,
        selected_count=int(selector.n_features_),
    )

    return template.save(output=output)
