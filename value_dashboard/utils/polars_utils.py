import logging
from typing import Dict, Any, List

import numpy as np
import polars as pl
from polars import Series
from pytdigest import TDigest

from value_dashboard.utils.logger import get_logger

T_DIGEST_COMPRESSION = 500
logger = get_logger(__name__, logging.DEBUG)


def df_to_dict(df: pl.DataFrame, key_col: str, value_col: str) -> Dict[Any, Any]:
    """
    Get a Python dict from two columns of a DataFrame
    If the key column is not unique, the last row is used
    """
    return dict(df.select(key_col, value_col).iter_rows())


def tdigest_pos_neg(args: List[Series]) -> Series:
    """
    TDigest helps in merging distributions, allowing for accurate computation of overall metrics like mean,
    variance, median, and ROC AUC over larger periods.
    Combining ROC AUC
    ROC AUC measures the ability of a classifier to distinguish between classes by considering the true positive rate (TPR)
    and false positive rate (FPR) across all thresholds. ROC AUC depends on the joint distribution of scores for positive
    and negative classes. ROC AUC is not a linear metric; therefore, you cannot average daily ROC AUCs to get a monthly ROC AUC.
    Use the merged positive and negative T-Digests to compute the ROC AUC over the period.
    More info: https://github.com/tdunning/t-digest
    Parameters:
    ----------
    args : a 2-element list of Series with boolean outcome or propensities
    Returns:
    -------
        The t-digest structure.
    """
    df = pl.DataFrame(args)
    tdigest_pos_df = df.filter(pl.col('column_0') == True)
    if tdigest_pos_df.shape[0] > 0:
        tdigest_pos = TDigest.compute(tdigest_pos_df.select('column_1').to_series().to_numpy(),
                                      compression=T_DIGEST_COMPRESSION)
    else:
        tdigest_pos = TDigest.compute(0.0, compression=T_DIGEST_COMPRESSION)

    tdigest_neg_df = df.filter(pl.col('column_0') == False)
    if tdigest_neg_df.shape[0] > 0:
        tdigest_neg = TDigest.compute(tdigest_neg_df.select('column_1').to_series().to_numpy(),
                                      compression=T_DIGEST_COMPRESSION)
    else:
        tdigest_neg = TDigest.compute(0.0, compression=T_DIGEST_COMPRESSION)

    return Series([{
        'tdigest_positives': {'tdigest': tdigest_pos.get_centroids().tolist()},
        'tdigest_negatives': {'tdigest': tdigest_neg.get_centroids().tolist()}
    }], dtype=pl.Struct)


def tdigest(args: List[Series]) -> Series:
    """
    TDigest helps in merging distributions, allowing for accurate computation of overall metrics like mean,
    variance, median, and ROC AUC over larger periods.
    Combining ROC AUC
    ROC AUC measures the ability of a classifier to distinguish between classes by considering the true positive rate (TPR)
    and false positive rate (FPR) across all thresholds. ROC AUC depends on the joint distribution of scores for positive
    and negative classes. ROC AUC is not a linear metric; therefore, you cannot average daily ROC AUCs to get a monthly ROC AUC.
    Use the merged positive and negative T-Digests to compute the ROC AUC over the period.
    More info: https://github.com/tdunning/t-digest
    Parameters:
    ----------
    args : a 1-element list of Series with values
    Returns:
    -------
        The t-digest structure.
    """
    df = pl.DataFrame(args)
    if df.shape[0] > 0:
        tdigest = TDigest.compute(df.select('column_0').to_series().to_numpy(), compression=T_DIGEST_COMPRESSION)
    else:
        tdigest = TDigest.compute(0.0, compression=T_DIGEST_COMPRESSION)
    return Series(
        [
            {
                'tdigest': tdigest.get_centroids().tolist()
            }
        ],
        dtype=pl.Struct
    )


def merge_tdigests(args: List[Series]) -> pl.Struct:
    """
    Merge t-digests into one
    Parameters:
    ----------
    args : a 1-element list of Series with values
    Returns:
    -------
        The t-digest structure.
    """
    df = pl.DataFrame(args)
    tdigests = []
    for row in df.iter_rows():
        tdigests.append(TDigest.of_centroids(np.array(row[0]['tdigest']), compression=T_DIGEST_COMPRESSION))
    tdigest = TDigest.combine(tdigests)
    tdigest.force_merge()
    return Series(
        [
            {
                'tdigest': tdigest.get_centroids().tolist()
            }
        ],
        dtype=pl.Struct
    )


def estimate_quantile(args: List[Series], quantile: float) -> pl.Struct:
    """
    Estimate quantile on t-digest column
    Parameters:
    ----------
    args : a 1-element list of Series with t-digest structures
    Returns:
    -------
        The t-digest structure.
    """
    # logger.debug("start estimate_quantile: " + str(quantile))
    df = pl.DataFrame(args)
    tdigests = []
    for row in df.iter_rows():
        tdigests.append(TDigest.of_centroids(np.array(row[0]['tdigest']), compression=T_DIGEST_COMPRESSION))
    tdigest = TDigest.combine(tdigests)
    tdigest.force_merge()
    inv_cdf = tdigest.inverse_cdf(quantile=quantile)
    # logger.debug("end estimate_quantile: " + str(quantile) + "  " + str(inv_cdf))
    return pl.Series([inv_cdf],
                     dtype=pl.Float64)
