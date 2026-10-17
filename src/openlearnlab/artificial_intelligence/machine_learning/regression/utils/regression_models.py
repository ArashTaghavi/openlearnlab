from __future__ import annotations

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def regression_models(random_state: int = 42, selected_models: list[str] | None = None):

    models = {
        # Baseline
        "Dummy": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        # Linear
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso()),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet()),
            ]
        ),
        "Polynomial Regression": Pipeline(
            [
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        # Distance / Kernel
        "KNN": Pipeline([("scaler", StandardScaler()), ("KNN", KNeighborsRegressor())]),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR()),
            ]
        ),
        # Tree
        "Decision Tree": DecisionTreeRegressor(
            random_state=random_state,
        ),
        # Bagging
        "Random Forest": RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            random_state=random_state,
            n_jobs=-1,
        ),
        # Boosting
        "AdaBoost": AdaBoostRegressor(
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=random_state,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        ),
        # "CatBoost": CatBoostRegressor(
        #     random_state=random_state,
        #     verbose=0,
        # ),
    }

    if selected_models is not None:
        models = {name: models[name] for name in selected_models}
    return models
