from pathlib import Path

import pandas as pd

from .core.correlation import correlation as get_correlation
from .core.info import info as get_info


class RegressionAnalyzer:
    def __init__(self, df: pd.DataFrame,dataset_name:str='Unnamed Dataset',output_dir:str='reports',title:str='OpenLearnLab Report'):
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
    ) -> Path:
        return get_correlation(
            df=self.df,
            dataset_name=self.dataset_name,
            output=self.output_dir / "correlation",
        )

    def reinit(self, df: pd.DataFrame, target: str):
        self._set_data(df, target)
