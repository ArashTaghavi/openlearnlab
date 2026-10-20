from pathlib import Path

import pandas as pd

from ..base import BaseAnalyzer
from .core import compare_models as get_compare_models


class ClusteringAnalyzer(BaseAnalyzer):
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
        models: dict,
        max_noise_ratio: float = 30.0,
        silhouette_sample_size: int | None = 5000,
    ) -> Path:
        return get_compare_models(
            df=self.df,
            features=features,
            models=models,
            dataset_name=self.dataset_name,
            output=self.output_dir / "compare_models",
            max_noise_ratio=max_noise_ratio,
            silhouette_sample_size=silhouette_sample_size,
            random_state=self.RANDOM_STATE,
        )
