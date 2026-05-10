from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataChatDataset:
    """Dataset plus metadata exposed to the generated analysis code."""

    name: str
    dataframe: pd.DataFrame
    description: str = ""

    def prompt_description(self, sample_rows: int = 5, max_columns: int = 30) -> str:
        """Return a prompt-ready description of the dataset and its columns."""
        dtypes = "\n".join(
            f"- {column}: {dtype}" for column, dtype in self.dataframe.dtypes.astype(str).items()
        )
        sample = self.dataframe.head(sample_rows).to_string(index=False, max_cols=max_columns)
        return f"""
Dataset name: {self.name}
Description: {self.description or "No description provided."}
Shape: {self.dataframe.shape[0]} rows x {self.dataframe.shape[1]} columns
Columns and dtypes:
{dtypes}
Sample rows:
{sample}
"""
