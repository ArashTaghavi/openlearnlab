from pathlib import Path

import pandas as pd

from ..base import BaseAnalyzer

# from .core import calculate_permutation_importance, plot_learning_curve
# from .core import compare_diagnostics as get_compare_diagnostics
# from .core import compare_feature_extraction as get_compare_feature_extraction
# from .core import compare_feature_sets as get_compare_feature_sets
from .core import compare_models as get_compare_models
from .core import diagnostics as get_diagnostics
from .core import (
    label_correction_analysis as get_label_correction_analysis,
)
from .core import label_quality_analysis as get_label_quality_analysis
from .core import similarity_analysis as get_similarity_analysis
from .core import threshold_analysis as get_threshold_analysis

# from .core import (
#     evaluate_feature_removal_combinations as get_evaluate_feature_removal_combinations,
# )
# from .core import evaluate_single_feature_removal as get_evaluate_single_feature_removal
# from .core import feature_extraction as get_feature_extraction
# from .core import feature_selection as get_feature_selection
# from .core import refine_and_compare as get_refine_and_compare
# from .core import tune_model as get_tune_model


class ClassificationAnalyzer(BaseAnalyzer):
    RANDOM_STATE = 44

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Unnamed Dataset",
        output_dir: str = "reports",
        title: str = "OpenLearnLab Report",
    ):
        super().__init__(
            df=df,
            dataset_name=dataset_name,
            output_dir=output_dir,
        )

        self.output_dir = Path(output_dir)
        self.title = title

    def compare_models(
        self,
        features,
        target: str,
        models: dict,
    ) -> Path:
        return get_compare_models(
            df=self.df,
            features=features,
            target=target,
            models=models,
            random_state=self.RANDOM_STATE,
            dataset_name=self.dataset_name,
            output=self.output_dir / "compare_models",
        )

    def diagnostics(
        self,
        features,
        target: str,
        model,
        model_name,
    ) -> Path:
        return get_diagnostics(
            df=self.df,
            features=features,
            target=target,
            model=model,
            model_name=model_name,
            random_state=self.RANDOM_STATE,
            dataset_name=self.dataset_name,
            output=self.output_dir / "diagnostics",
        )

    def threshold_analysis(
        self,
        features,
        target: str,
        model,
        model_name,
        metric: str = "recall",
        average: int | str = 1,
    ) -> Path:
        return get_threshold_analysis(
            df=self.df,
            features=features,
            target=target,
            model=model,
            model_name=model_name,
            random_state=self.RANDOM_STATE,
            dataset_name=self.dataset_name,
            output=self.output_dir / "threshold_analysis",
            metric=metric,
            average=average,
        )

    def label_quality_analysis(
        self,
        features,
        target: str,
        model,
        model_name,
        n_neighbors: int = 10,
        min_confidence: float = 90.0,
        min_disagreement: float = 80.0,
    ) -> Path:
        return get_label_quality_analysis(
            df=self.df,
            features=features,
            target=target,
            model=model,
            model_name=model_name,
            dataset_name=self.dataset_name,
            output=self.output_dir / "label_quality_analysis",
            random_state=self.RANDOM_STATE,
            n_neighbors=n_neighbors,
            min_confidence=min_confidence,
            min_disagreement=min_disagreement,
        )

    # def compare_diagnostics(
    #     self,
    #     models: dict,
    #     features,
    #     target: str,
    # ) -> Path:
    #     return get_compare_diagnostics(
    #         df=self.df,
    #         features=features,
    #         target=target,
    #         models=models,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "diagnostics/compare_diagnostics",
    #         random_state=self.RANDOM_STATE,
    #     )

    # def refine_and_compare(
    #     self,
    #     features,
    #     target: str,
    #     model,
    # ) -> Path:
    #     return get_refine_and_compare(
    #         df=self.df,
    #         features=features,
    #         target=target,
    #         model=model,
    #         random_state=self.RANDOM_STATE,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "refine_and_compare",
    #     )

    # def evaluate_single_feature_removal(
    #     self,
    #     features,
    #     target: str,
    #     model,
    # ) -> Path:
    #     return get_evaluate_single_feature_removal(
    #         df=self.df,
    #         features=features,
    #         target=target,
    #         model=model,
    #         random_state=self.RANDOM_STATE,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "evaluate_single_feature_removal",
    #     )

    # def evaluate_feature_removal_combinations(
    #     self,
    #     features,
    #     candidates,
    #     target: str,
    #     model,
    # ) -> Path:
    #     return get_evaluate_feature_removal_combinations(
    #         df=self.df,
    #         features=features,
    #         candidates=candidates,
    #         target=target,
    #         model=model,
    #         random_state=self.RANDOM_STATE,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "evaluate_feature_removal_combinations",
    #     )

    # def permutation_importance(
    #     self,
    #     target: str,
    #     model=None,
    # ) -> Path:
    #     return calculate_permutation_importance(
    #         df=self.df,
    #         target=target,
    #         model=model,
    #         random_state=self.RANDOM_STATE,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "permutation_importance",
    #     )

    # def compare_feature_sets(
    #     self,
    #     target: str,
    #     model,
    #     feature_sets: dict,
    #     cv: int = 5,
    # ) -> Path:
    #     return get_compare_feature_sets(
    #         df=self.df,
    #         target=target,
    #         model=model,
    #         feature_sets=feature_sets,
    #         cv=cv,
    #         random_state=self.RANDOM_STATE,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "compare_feature_sets",
    #     )

    # def feature_selection(
    #     self,
    #     target: str,
    #     features,
    #     model=None,
    #     scoring: str = "f1_weighted",
    #     cv: int = 5,
    # ) -> Path:
    #     return get_feature_selection(
    #         df=self.df,
    #         features=features,
    #         target=target,
    #         model=model,
    #         scoring=scoring,
    #         cv=cv,
    #         random_state=self.RANDOM_STATE,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "feature_selection",
    #     )

    # def feature_extraction(
    #     self,
    #     target: str,
    #     features,
    #     variance_threshold: float = 0.95,
    # ) -> Path:
    #     return get_feature_extraction(
    #         df=self.df,
    #         target=target,
    #         features=features,
    #         variance_threshold=variance_threshold,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "feature_extraction",
    #     )

    # def compare_feature_extraction(
    #     self,
    #     models: dict,
    #     target: str,
    #     features,
    #     variance_threshold: float = 0.95,
    # ) -> Path:
    #     return get_compare_feature_extraction(
    #         df=self.df,
    #         features=features,
    #         target=target,
    #         models=models,
    #         variance_threshold=variance_threshold,
    #         dataset_name=self.dataset_name,
    #         random_state=self.RANDOM_STATE,
    #         output=self.output_dir / "compare_feature_extraction",
    #     )

    # def tune_model(
    #     self,
    #     target: str,
    #     model,
    #     param_grid: dict,
    #     features,
    #     cv: int = 5,
    #     scoring: str = "f1_weighted",
    # ) -> Path:
    #     return get_tune_model(
    #         df=self.df,
    #         target=target,
    #         model=model,
    #         param_grid=param_grid,
    #         features=features,
    #         cv=cv,
    #         scoring=scoring,
    #         random_state=self.RANDOM_STATE,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "tune_model",
    #     )

    # def learning_curve(
    #     self,
    #     target: str,
    #     features,
    #     cv: int = 5,
    #     model=None,
    #     train_sizes=None,
    # ) -> Path:
    #     return plot_learning_curve(
    #         df=self.df,
    #         target=target,
    #         cv=cv,
    #         random_state=self.RANDOM_STATE,
    #         features=features,
    #         model=model,
    #         train_sizes=train_sizes,
    #         dataset_name=self.dataset_name,
    #         output=self.output_dir / "learning_curve",
    #     )
