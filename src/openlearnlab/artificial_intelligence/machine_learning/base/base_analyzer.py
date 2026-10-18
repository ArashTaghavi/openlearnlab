from pathlib import Path
from typing import Literal

import pandas as pd

from .core import correlation as get_correlation
from .core import info as get_info


class BaseAnalyzer:
    def __init__(self, df: pd.DataFrame, dataset_name: str, output_dir: str):
        self.df = df
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)

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
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
    ) -> Path:
        return get_correlation(
            df=self.df,
            threshold=threshold,
            features=features,
            method=method,
            dataset_name=self.dataset_name,
            output=self.output_dir / "correlation",
        )
