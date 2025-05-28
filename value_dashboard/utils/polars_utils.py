import logging
from typing import Dict, Any, List

import datasketches
import numpy as np
import polars as pl

from value_dashboard.utils.logger import get_logger

T_DIGEST_COMPRESSION = 500
REQ_SKETCH_K = 12
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


def build_req_floats_sketch(args: List[pl.Series]) -> bytes:
    arr = np.array(args[0].drop_nulls(), dtype=np.float32)
    sketch = datasketches.req_floats_sketch(k=REQ_SKETCH_K, is_hra=False)
    sketch.update(arr)
    return sketch.serialize()


def merge_req_floats_sketches(args: List[pl.Series]
                              ) -> bytes:
    sketch_bytes_list = args[0].to_list()[0]
    merged = datasketches.req_floats_sketch.deserialize(sketch_bytes_list[0])
    for b in sketch_bytes_list[1:]:
        other = datasketches.req_floats_sketch.deserialize(b)
        merged.merge(other)
    return merged.serialize()


def estimate_quantile(args: List[pl.Series], quantile: float) -> float:
    sketch_bytes_list = args[0].to_list()[0]
    merged = datasketches.req_floats_sketch.deserialize(sketch_bytes_list[0])
    for b in sketch_bytes_list[1:]:
        other = datasketches.req_floats_sketch.deserialize(b)
        merged.merge(other)
    return merged.get_quantile(quantile)
