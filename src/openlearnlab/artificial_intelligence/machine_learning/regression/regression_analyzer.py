from pathlib import Path

import pandas as pd

from .core import calculate_permutation_importance, plot_learning_curve
from .core import compare_feature_sets as get_compare_feature_sets
from .core import compare_models as get_compare_models
from .core import (
    compare_regularization as get_compare_regularization,
)
from .core import correlation as get_correlation
from .core import diagnostics as get_diagnostics
from .core import (
    evaluate_feature_removal_combinations as get_evaluate_feature_removal_combinations,
)
from .core import evaluate_single_feature_removal as get_evaluate_single_feature_removal
from .core import feature_extraction as get_feature_extraction
from .core import feature_selection as get_feature_selection
from .core import info as get_info
from .core import refine_and_evaluate as get_refine_and_evaluate
from .core import tune_model as get_tune_model


class RegressionAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Unnamed Dataset",
        output_dir: str = "reports",
        title: str = "OpenLearnLab Report",
    ):
        self.df = df
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.title = title

    def _set_data(
        self,
        df: pd.DataFrame,
        target: str,
    ):
        if target is not None and target not in df.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame.")

        self.df = df.copy()
        self.target = target

    def info(self) -> Path:
        return get_info(
            df=self.df,
            dataset_name=self.dataset_name,
            output=self.output_dir / "info",
        )

    def correlation(
        self,
        features,
        threshold=0.8,
    ) -> Path:
        return get_correlation(
            df=self.df,
            threshold=threshold,
            features=features,
            dataset_name=self.dataset_name,
            output=self.output_dir / "correlation",
        )

    def compare_models(
        self,
        features,
        target: str,
        models: dict,
        random_state=44,
    ) -> Path:
        return get_compare_models(
            df=self.df,
            features=features,
            target=target,
            models=models,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "compare_models",
        )

    def diagnostics(
        self,
        features,
        target: str,
        model,
        random_state: int = 44,
    ) -> Path:
        return get_diagnostics(
            df=self.df,
            features=features,
            target=target,
            model=model,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "diagnostics",
        )

    def refine_and_evaluate(
        self,
        features,
        target: str,
        model,
        random_state: int = 44,
    ) -> Path:
        return get_refine_and_evaluate(
            df=self.df,
            features=features,
            target=target,
            model=model,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "refine_and_evaluate",
        )

    def evaluate_single_feature_removal(
        self,
        features,
        target: str,
        model,
        random_state: int = 44,
    ) -> Path:
        return get_evaluate_single_feature_removal(
            df=self.df,
            features=features,
            target=target,
            model=model,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "evaluate_single_feature_removal",
        )

    def evaluate_feature_removal_combinations(
        self,
        features,
        candidates,
        target: str,
        model,
        random_state: int = 44,
    ) -> Path:
        return get_evaluate_feature_removal_combinations(
            df=self.df,
            features=features,
            candidates=candidates,
            target=target,
            model=model,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "evaluate_feature_removal_combinations",
        )

    def permutation_importance(
        self,
        target: str,
        model=None,
        random_state: int = 44,
    ) -> Path:
        return calculate_permutation_importance(
            df=self.df,
            target=target,
            model=model,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "permutation_importance",
        )

    def compare_feature_sets(
        self,
        target: str,
        model,
        feature_sets: dict,
        cv: int = 5,
        random_state: int = 44,
    ) -> Path:
        return get_compare_feature_sets(
            df=self.df,
            target=target,
            model=model,
            feature_sets=feature_sets,
            cv=cv,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "compare_feature_sets",
        )

    def feature_selection(
        self,
        target: str,
        model=None,
        features=None,
        scoring: str = "r2",
        cv: int = 5,
        random_state: int = 44,
    ) -> Path:
        return get_feature_selection(
            df=self.df,
            features=features,
            target=target,
            model=model,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "feature_selection",
        )

    def feature_extraction(
        self,
        target: str,
        features=None,
        variance_threshold: float = 0.95,
    ) -> Path:
        return get_feature_extraction(
            df=self.df,
            target=target,
            features=features,
            variance_threshold=variance_threshold,
            dataset_name=self.dataset_name,
            output=self.output_dir / "feature_extraction",
        )

    def tune_model(
        self,
        target: str,
        model,
        param_grid: dict,
        features=None,
        cv: int = 5,
        scoring=None,
        random_state: int = 44,
    ) -> Path:
        return get_tune_model(
            df=self.df,
            target=target,
            model=model,
            param_grid=param_grid,
            features=features,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "tune_model",
        )

    def learning_curve(
        self,
        target: str,
        cv: int = 5,
        random_state: int = 44,
        features=None,
        model=None,
        train_sizes=None,
    ) -> Path:
        return plot_learning_curve(
            df=self.df,
            target=target,
            cv=cv,
            random_state=random_state,
            features=features,
            model=model,
            train_sizes=train_sizes,
            dataset_name=self.dataset_name,
            output=self.output_dir / "learning_curve",
        )

    def compare_regularization(
        self,
        target: str,
        features=None,
        alphas=None,
        l1_ratio: float = 0.5,
        random_state: int = 44,
    ) -> Path:
        return get_compare_regularization(
            df=self.df,
            target=target,
            features=features,
            alphas=alphas,
            l1_ratio=l1_ratio,
            random_state=random_state,
            dataset_name=self.dataset_name,
            output=self.output_dir / "compare_regularization",
        )

    def reinit(self, df: pd.DataFrame, target: str):
        self._set_data(df, target)
