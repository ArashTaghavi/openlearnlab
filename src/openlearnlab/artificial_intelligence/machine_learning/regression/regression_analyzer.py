from pathlib import Path

import pandas as pd

from .core import compare_models as get_compare_models
from .core import correlation as get_correlation
from .core import diagnostics as get_diagnostics
from .core import info as get_info
from .core import refine_and_evaluate as get_refine_and_evaluate


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

    def reinit(self, df: pd.DataFrame, target: str):
        self._set_data(df, target)
