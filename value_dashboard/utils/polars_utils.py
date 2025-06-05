import logging
from typing import Dict, Any, List

import datasketches
import numpy as np
import polars as pl

from value_dashboard.utils.logger import get_logger

T_DIGEST_COMPRESSION = 500
REQ_SKETCH_ACCURACY = 24
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
            unique_count = df[col].n_unique()
            mode = df[col].mode().to_list()
            unique = df[col].unique().to_list()
            records.append({
                "Column": col,
                "Data Type": str(dtype),
                "Unique Count": unique_count,
                "Mode": str(mode),
                "Values": str(unique) if unique_count < 10 else '...'
            })
        elif dtype.is_numeric():
            records.append({
                "Column": col,
                "Data Type": str(dtype),
                "Unique Count": "N/A",
                "Mode": "N/A",
                "Values": "Min = " + f'{df[col].min():.4f}' + " Max = " + f'{df[col].max():.4f}' + " Mean = "
                          + f'{df[col].mean():.4f}' + " Median = " + f'{df[col].median():.4f}'
            })
        else:
            records.append({
                "Column": col,
                "Data Type": str(dtype),
                "Unique Count": "N/A",
                "Mode": "N/A",
                "Values": "Min = " + f'{df[col].min()}' + " Max = " + f'{df[col].max()}'
            })

    return pl.DataFrame(records)


def build_digest(args: List[pl.Series]) -> bytes:
    arr = np.array(args[0].drop_nulls(), dtype=np.float64)
    sketch = datasketches.tdigest_double(k=T_DIGEST_COMPRESSION)
    sketch.update(arr)
    return sketch.serialize()


def merge_digests(args: List[pl.Series]
                  ) -> bytes:
    sketch_bytes_list = args[0].to_list()
    merged = datasketches.tdigest_double.deserialize(sketch_bytes_list[0])
    for b in sketch_bytes_list[1:]:
        other = datasketches.tdigest_double.deserialize(b)
        merged.merge(other)
    return merged.serialize()


def estimate_quantile(args: List[pl.Series], quantile: float) -> float:
    sketch_bytes_list = args[0].to_list()
    merged = datasketches.tdigest_double.deserialize(sketch_bytes_list[0])
    for b in sketch_bytes_list[1:]:
        other = datasketches.tdigest_double.deserialize(b)
        merged.merge(other)
    return merged.get_quantile(quantile)
