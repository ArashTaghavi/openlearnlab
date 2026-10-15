from pathlib import Path

import pandas as pd

from ..reporting.templates.info_template import InfoTemplate


def info(
    df: pd.DataFrame,
    dataset_name: str,
    output: Path,
):
    shape = df.shape

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]

    duplicated = int(df.duplicated().sum())
    columns = df.columns.to_list()

    invalid_values = {
        "",
        "null",
        "none",
        "nan",
        "na",
        "n/a",
        "?",
        "-",
        "--",
        "undefined",
    }

    invalid_data = {}

    for column in df.columns:
        invalid_count = df[column].isin(invalid_values).sum()

        if invalid_count > 0:
            invalid_data[column] = int(invalid_count)

    df_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Non-Null Count": df.notna().sum().values,
            "Null Count": df.isna().sum().values,
            "Dtype": df.dtypes.astype(str).values,
        }
    )

    df_describe = (
        df.describe(include="all").T if len(df.columns) > 0 else pd.DataFrame()
    )

    template = InfoTemplate(
        dataset_name=dataset_name,
        shape=shape,
        duplicated=duplicated,
        null_counts=null_counts.to_dict(),
        columns=columns,
        invalid_data=invalid_data,
        df_info=df_info,
        df_describe=df_describe,
        df_head=df.head(),
    )

    return template.save(output=output)
