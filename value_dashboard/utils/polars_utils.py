import logging
from typing import Dict, Any, List

from fastdigest import TDigest, merge_all
import numpy as np
import polars as pl

from value_dashboard.utils.logger import get_logger

T_DIGEST_COMPRESSION = 1000
logger = get_logger(__name__, logging.DEBUG)


def df_to_dict(df: pl.DataFrame, key_col: str, value_col: str) -> Dict[Any, Any]:
    """
    Get a Python dict from two columns of a DataFrame
    If the key column is not unique, the last row is used
    """
    return dict(df.select(key_col, value_col).iter_rows())


def schema_with_unique_counts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Return a Polars DataFrame schema with the number of unique values for each string column.

    Parameters
    ----------
    df : pl.DataFrame
        The input Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        A DataFrame with columns: 'column', 'dtype', and 'unique_count'.
    """

    schema = df.schema
    records = []
    for col, dtype in schema.items():
        if dtype == pl.Utf8:
            count = df[col].n_unique()
            records.append({
                "Column": col,
                "Data Type": str(dtype),
                "Unique Count": count
            })
        else:
            records.append({
                "Column": col,
                "Data Type": str(dtype),
                "Unique Count": "N/A"
            })

    return pl.DataFrame(records)


def build_tdigest(args: List[pl.Series]) -> bytes:
    arr = np.array(args[0].drop_nulls(), dtype=np.float64)
    if arr.size == 0:
        sketch = TDigest.from_values([0.0], T_DIGEST_COMPRESSION)
    else:
        sketch = TDigest.from_values(arr, T_DIGEST_COMPRESSION)
    return sketch.to_dict()


def merge_tdigests(args: List[pl.Series]
                   ) -> bytes:
    sketch_bytes_list = args[0].to_list()[0]
    partial_digests = []
    for b in sketch_bytes_list:
        partial_digests.append(TDigest.from_dict(b))
    merged = merge_all(partial_digests)
    return merged.to_dict()

def estimate_quantile(args: List[pl.Series], quantile: float) -> float:
    sketch_bytes_list = args[0].to_list()[0]
    partial_digests = []
    for b in sketch_bytes_list:
        partial_digests.append(TDigest.from_dict(b))
    merged = merge_all(partial_digests)
    return merged.quantile(quantile)
