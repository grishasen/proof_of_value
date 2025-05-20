import logging
from typing import Dict, Any

import polars as pl

from value_dashboard.utils.logger import get_logger

T_DIGEST_COMPRESSION = 500
logger = get_logger(__name__, logging.DEBUG)


def df_to_dict(df: pl.DataFrame, key_col: str, value_col: str) -> Dict[Any, Any]:
    """
    Get a Python dict from two columns of a DataFrame
    If the key column is not unique, the last row is used
    """
    return dict(df.select(key_col, value_col).iter_rows())
