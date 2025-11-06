import logging
from typing import Dict, Any, List

import datasketches
import numpy as np
import polars as pl
from polars.datatypes._parse import NoneType

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
        if col.lower().endswith('id'):
            records.append({
                "Column": col,
                "Data Type": str(dtype),
                "Unique Count": "N/A",
                "Mode": "N/A",
                "Values": ''
            })
        elif dtype == pl.Utf8:
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
        elif dtype == NoneType:
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
    """
    Build and serialize a t-digest from a Polars Series of numeric values.

    Parameters
    ----------
    args : List[pl.Series]
        The first element (args[0]) must be a Polars Series of numeric values.
        Nulls are dropped and values are converted to float64 before updating
        the t-digest.

    Returns
    -------
    bytes
        Serialized t-digest (datasketches.tdigest_double) representing the
        distribution of the input values.

    Notes
    -----
    - Uses the compression parameter `T_DIGEST_COMPRESSION`.
    - Bulk updating with a NumPy array is efficient for large inputs.
    """
    arr = np.array(args[0].drop_nulls(), dtype=np.float64)
    sketch = datasketches.tdigest_double(k=T_DIGEST_COMPRESSION)
    sketch.update(arr)
    return sketch.serialize()


def merge_digests(args: List[pl.Series]
                  ) -> bytes:
    """
    Merge a collection of serialized t-digests into a single serialized digest.

    Parameters
    ----------
    args : List[pl.Series]
        The first element (args[0]) must be a Polars Series where each entry is
        a bytes object corresponding to a serialized t-digest
        (datasketches.tdigest_double).

    Returns
    -------
    bytes
        Serialized t-digest resulting from merging all input digests.

    Edge Cases
    ----------
    - If the input series is empty, returns a serialized digest initialized with
      a single dummy value (0.0) to preserve downstream type/shape expectations.

    Notes
    -----
    - Merging is associative, enabling scalable, distributed aggregation.
    """
    sketch_bytes_list = args[0].to_list()
    if not sketch_bytes_list:
        sketch = datasketches.tdigest_double(k=T_DIGEST_COMPRESSION)
        sketch.update(0.0)
        return sketch.serialize()
    merged = datasketches.tdigest_double.deserialize(sketch_bytes_list[0])
    for b in sketch_bytes_list[1:]:
        other = datasketches.tdigest_double.deserialize(b)
        merged.merge(other)
    return merged.serialize()


def estimate_quantile(args: List[pl.Series], quantile: float) -> float:
    """
    Estimate a single quantile from a collection of serialized t-digests.

    Parameters
    ----------
    args : List[pl.Series]
        The first element (args[0]) must be a Polars Series where each entry is
        a bytes object of a serialized t-digest (datasketches.tdigest_double).
    quantile : float
        The desired quantile in the closed interval [0.0, 1.0].
        For example, 0.5 for the median.

    Returns
    -------
    float
        The estimated quantile value. Returns 0.0 if the input series is empty.

    Notes
    -----
    - All digests are deserialized and merged prior to querying the quantile.
    """
    sketch_bytes_list = args[0].to_list()
    if not sketch_bytes_list:
        return 0.0
    merged = datasketches.tdigest_double.deserialize(sketch_bytes_list[0])
    for b in sketch_bytes_list[1:]:
        other = datasketches.tdigest_double.deserialize(b)
        merged.merge(other)
    return merged.get_quantile(quantile)


def estimate_quantiles_arr(args: List[pl.Series], quantiles: List[float]) -> List[float]:
    """
    Estimate multiple quantiles from a collection of serialized t-digests.

    Parameters
    ----------
    args : List[pl.Series]
        The first element (args[0]) must be a Polars Series where each entry is
        a bytes object of a serialized t-digest (datasketches.tdigest_double).
    quantiles : List[float]
        A list of desired quantiles in the interval [0.0, 1.0].

    Returns
    -------
    List[float]
        A list of estimated quantile values corresponding to `quantiles`.
        Returns [0.0] if the input series is empty.

    Notes
    -----
    - Performs a single merge pass, then queries all quantiles.
    """
    sketch_bytes_list = args[0].to_list()
    if not sketch_bytes_list:
        return [0.0]
    merged = datasketches.tdigest_double.deserialize(sketch_bytes_list[0])
    for b in sketch_bytes_list[1:]:
        other = datasketches.tdigest_double.deserialize(b)
        merged.merge(other)
    return [merged.get_quantile(quantile) for quantile in quantiles]


def digest_to_histogram(tdigest: bytes, bins: int = 30, value_range: tuple[float, float] = None):
    """
    Approximate a histogram from a serialized t-digest.

    Parameters
    ----------
    tdigest : bytes
        Serialized t-digest (datasketches.tdigest_double).
    bins : int, optional
        Number of histogram bins, by default 30.
    value_range : tuple[float, float], optional
        The inclusive range (min, max) for the histogram. If None, uses the
        digest's 0th and 100th percentiles (min/max) as bounds.

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        A tuple of (bin_edges, bin_counts), where:
        - bin_edges : np.ndarray of shape (bins + 1,)
            The edges defining each histogram bin.
        - bin_counts : List[float]
            Approximate probability mass per bin computed via CDF differences.
            If `value_range` covers the full support, these masses sum to ~1.0.

    Notes
    -----
    - Uses CDF differences over adjacent bin edges to estimate per-bin mass.
    - If counts (not probabilities) are required, multiply by the total sample
      size used to build the digest (tracked externally).
    """
    tdigest = datasketches.tdigest_double.deserialize(tdigest)
    if value_range is None:
        value_range = (tdigest.get_quantile(0), tdigest.get_quantile(1))
    bin_edges = np.linspace(*value_range, bins + 1)
    bin_counts = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        count = tdigest.get_cdf([right])[0] - tdigest.get_cdf([left])[0]
        bin_counts.append(count)
    return bin_edges, bin_counts
