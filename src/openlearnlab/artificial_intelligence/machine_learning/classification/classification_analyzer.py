from pathlib import Path

import pandas as pd

from ..base import BaseAnalyzer
from .core import compare_models as get_compare_models
from .core import diagnostics as get_diagnostics


class ClassificationAnalyzer(BaseAnalyzer):
    RANDOM_STATE = 44

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Unnamed Dataset",
        output_dir: str = "reports",
        title: str = "OpenLearnLab Report",
    ):
        super().__init__(df=df, dataset_name=dataset_name, output_dir=output_dir)
        self.output_dir = Path(output_dir)
        self.title = title

    def compare_models(
        self,
        features,
        target: str,
        models: dict,
        random_state=RANDOM_STATE,
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
        random_state=RANDOM_STATE,
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
