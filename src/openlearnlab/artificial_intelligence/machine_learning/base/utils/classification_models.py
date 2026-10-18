from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def classification_models(
    random_state: int = 42,
    selected_models: list[str] | None = None,
):
    models = {
        # Baseline
        "Dummy": DummyClassifier(
            strategy="most_frequent",
        ),
        # Linear
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeClassifier()),
            ]
        ),
        "SGD": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SGDClassifier(random_state=random_state)),
            ]
        ),
        "Polynomial Logistic Regression": Pipeline(
            [
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        # Distance / Kernel
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "SVC": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(random_state=random_state)),
            ]
        ),
        # Naive Bayes
        "Gaussian NB": GaussianNB(),
        # Tree
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
        ),
        # Bagging
        "Random Forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            random_state=random_state,
            n_jobs=-1,
        ),
        # Boosting
        "AdaBoost": AdaBoostClassifier(
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        ),
        # "CatBoost": CatBoostClassifier(
        #     random_state=random_state,
        #     verbose=0,
        # ),
    }

    if selected_models is not None:
        models = {name: models[name] for name in selected_models}

    return models
