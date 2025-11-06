import traceback
from typing import List, Sequence

import datasketches
import numpy as np
import polars as pl
import polars_ds as pds
from polars import Series
from polars import selectors as cs
from polars_ds import weighted_mean
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from value_dashboard.metrics.constants import NAME, CUSTOMER_ID, INTERACTION_ID, RANK, OUTCOME, PROPENSITY, \
    FINAL_PROPENSITY, ACTION_ID
from value_dashboard.utils.config import get_config
from value_dashboard.utils.polars_utils import merge_digests, build_digest
from value_dashboard.utils.py_utils import strtobool, stable_dedup
from value_dashboard.utils.timer import timed


def personalization(args: List[Series]) -> pl.Float64:
    """
    Personalization measures recommendation similarity across users.
    A high score indicates good personalization (user's lists of recommendations are different).
    A low score indicates poor personalization (user's lists of recommendations are very similar).
    A model is "personalizing" well if the set of recommendations for each user is different.
    Inspired by implementation at https://github.com/statisticianinstilettos/recmetrics.git (unfortunately could not use it in polars context)
    Parameters:
    ----------
    args : a 2-elements list of Series with CustomerID and Action Name
    Returns:
    -------
        The personalization score for all recommendations.
    """
    if len(args) < 2:
        return 0.0
    df = pl.DataFrame(args, schema=[CUSTOMER_ID, NAME])
    height = df.height
    if height < 5000:
        pass
    elif height < 10000:
        df = df.slice(round(height / 2))
    else:
        df = df.slice(round(height / 2), 5000)
    h = FeatureHasher(input_type="string", n_features=2 ** 16)
    df = (
        df
        .group_by([CUSTOMER_ID])
        .agg([
            pl.col(NAME).alias("ActionNames")
        ])
    )

    predicted = df.get_column("ActionNames").to_list()
    rec_matrix_sparse = h.transform(predicted)
    similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)
    dim = similarity.shape[0]
    if dim == 1:
        return 0.0
    personalization_score = (similarity.sum() - dim) / (dim * (dim - 1))
    return 1 - personalization_score


def personalization_optimized(args: List[Series]) -> pl.Float64:
    """
    Personalization measures recommendation similarity across users (Optimized).
    Avoids computing the full N x N similarity matrix.

    Parameters:
    ----------
    args : a 2-elements list of Series with CustomerID and Action Name

    Returns:
    -------
    The personalization score for all recommendations.
    """
    if len(args) < 2:
        return 0.0

    df = pl.DataFrame(args, schema=[CUSTOMER_ID, NAME])
    if df.height < 2:
        return 0.0
    df = pl.DataFrame(args, schema=[CUSTOMER_ID, NAME])
    height = df.height
    if height < 50000:
        pass
    elif height < 100000:
        df = df.slice(round(height / 2))
    else:
        df = df.slice(round(height / 2), 50000)

    df_grouped = (
        df
        .group_by(CUSTOMER_ID)
        .agg(pl.col(NAME).alias("ActionNames"))
    )

    predicted = df_grouped.get_column("ActionNames").to_list()
    dim = len(predicted)

    if dim <= 1:
        return 0.0

    h = FeatureHasher(input_type="string", n_features=2 ** 16)
    rec_matrix_sparse = h.transform(predicted)
    rec_matrix_normalized = normalize(rec_matrix_sparse, norm='l2', axis=1)
    total_similarity_sum = np.sum(np.square(rec_matrix_normalized.sum(axis=0)))
    off_diagonal_similarity_sum = total_similarity_sum - dim
    if dim > 1:
        avg_off_diagonal_similarity = off_diagonal_similarity_sum / (dim * (dim - 1))
    else:
        avg_off_diagonal_similarity = 0.0

    personalization_score = 1.0 - avg_off_diagonal_similarity

    return personalization_score


def novelty(args: Sequence[Series]) -> float:
    """
    Compute a novelty score for a recommendation set.

    The function approximates the information-theoretic novelty of recommended items
    by aggregating their (self-)information weighted by frequency and normalizing by
    the number of unique users and a proxy of recommendation list length.

    For performance on very large groups, the function subsamples the input:
    - if `height < 50_000`: use full sample,
    - if `50_000 <= height < 100_000`: use the second half,
    - if `height >= 100_000`: use a 50k slice from the middle.

    Parameters
    ----------
    args : Sequence[polars.Series]
        Sequence expected to contain three aligned series (in this order) representing:
        `[CUSTOMER_ID, INTERACTION_ID, NAME]`. This signature is designed to be used
        with `pl.map_groups(exprs=[...], function=novelty, ...)`.

    Returns
    -------
    float
        A scalar novelty score in `[0, +inf)`. Returns `0.0` if there are no users,
        the maximum recommendation length is `0`, or inputs are insufficient.

    Notes
    -----
    - The per-item "self information" proxy is computed as:
      `ActionCount * -(log2(ActionCount / unique_users) + 1e-10)`.
    - The final score is normalized by `(unique_users * max_rec_length)` to make values
      comparable across groups of different sizes.
    - Assumes global constants for column names: `CUSTOMER_ID`, `INTERACTION_ID`, `NAME`.

    """
    if len(args) < 2:
        return 0.0
    df = pl.DataFrame(args, schema=[CUSTOMER_ID, INTERACTION_ID, NAME])
    height = df.height
    if height < 50000:
        pass
    elif height < 100000:
        df = df.slice(round(height / 2))
    else:
        df = df.slice(round(height / 2), 50000)
    u = df.n_unique(subset=[CUSTOMER_ID])
    if u == 0:
        return 0.0
    item_counts = (
        df.group_by(NAME)
        .agg(pl.len().alias("ActionCount"))
        .with_columns(
            (pl.col("ActionCount") * -((pl.col("ActionCount") / u).log(base=2) + 1e-10)).alias("total_self_info")
        )
    )
    total_self_info = item_counts["total_self_info"].sum()
    n = (
        df.group_by(INTERACTION_ID)
        .agg(pl.len().alias("RecLength"))
        ["RecLength"].max()
    )
    if n == 0:
        return 0.0
    novelty_score = total_self_info / (u * n)
    return float(novelty_score)


def binary_metrics_tdigest(args: List[Series]) -> pl.Struct:
    """
    Compute binary classification metrics from **t-digests** of scores.

    This function expects per-group **collections of t-digest blobs** (positives and
    negatives). It merges each collection, reconstructs the CDFs, and computes:
    ROC AUC, Average Precision (AP), TPR/FPR arrays, precision/recall curves, and
    the positive fraction.

    Parameters
    ----------
    args : List[polars.Series]
        A two-element list of series: `[positives_tdigests, negatives_tdigests]`.
        Each series contains binary blobs representing serialized t-digests of scores
        that will be merged (via `merge_digests`) and deserialized using
        `datasketches.tdigest_double.deserialize`.

    Returns
    -------
    pl.Struct
        A dict-like structure with keys:
        - `roc_auc` : float
        - `average_precision` : float
        - `tpr` : list[float]            (sorted by FPR ascending)
        - `fpr` : list[float]            (sorted ascending)
        - `precision` : list[float]      (monotonically non-increasing w.r.t. recall)
        - `recall` : list[float]         (descending thresholds)
        - `pos_fraction` : float         (positives / (positives + negatives))

    Notes
    -----
    - The ROC curve is computed from the complementary CDF (1 - CDF(threshold)).
    - Precision is made non-increasing by applying `np.maximum.accumulate` in reverse
      to ensure the interpolated AP follows the standard definition.
    - If either positive or negative digest set is empty, returns zeros.

    """
    pos_series, neg_series = args[0], args[1]
    if pos_series.len() == 0 or neg_series.len() == 0:
        return {
            'roc_auc': 0.0,
            'average_precision': 0.0,
            'tpr': [0.0],
            'fpr': [0.0],
            'precision': [0.0],
            'recall': [0.0],
            'pos_fraction': 0.0
        }

    thresholds = [round(float(t), 4) for t in np.linspace(0, 1, num=101)]

    all_pos = args[0]
    all_neg = args[1]

    pos_sk = datasketches.tdigest_double.deserialize(merge_digests([all_pos]))
    neg_sk = datasketches.tdigest_double.deserialize(merge_digests([all_neg]))

    if pos_sk.is_empty() or neg_sk.is_empty():
        return {
            'roc_auc': 0.0,
            'average_precision': 0.0,
            'tpr': [0.0],
            'fpr': [0.0],
            'precision': [0.0],
            'recall': [0.0],
            'pos_fraction': 0.0
        }

    pos_count = pos_sk.get_total_weight()
    neg_count = neg_sk.get_total_weight()

    cdf_pos = np.array(pos_sk.get_cdf(thresholds))
    cdf_neg = np.array(neg_sk.get_cdf(thresholds))
    tpr = 1.0 - cdf_pos
    fpr = 1.0 - cdf_neg

    idx = np.argsort(fpr)
    fpr_sorted = fpr[idx]
    tpr_sorted = tpr[idx]
    roc_auc = np.trapz(tpr_sorted, fpr_sorted)

    recall = tpr[::-1]
    fpr_desc = fpr[::-1]
    tp = pos_count * recall
    fp = neg_count * fpr_desc
    pos_fraction = pos_count / (pos_count + neg_count)

    eps = 1e-10
    precision = tp / (tp + fp + eps)

    precision = np.maximum.accumulate(precision[::-1])[::-1]

    if recall[0] != 0.0:
        recall = np.concatenate(([0.0], recall))
        precision = np.concatenate(([1.0], precision))

    delta_r = recall[1:] - recall[:-1]
    average_precision = np.sum(delta_r * precision[1:])

    return {
        'roc_auc': float(roc_auc),
        'average_precision': float(average_precision),
        'tpr': tpr_sorted.tolist(),
        'fpr': fpr_sorted.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'pos_fraction': pos_fraction
    }


def calibration_tdigest(args: List[Series]) -> pl.Struct:
    """
    Build calibration data from **t-digests** of positive/negative score distributions.

    Partitions the score range into bins, estimates the average **predicted propensity**
    per bin by querying t-digests (via quantiles within each bin), and computes the
    **observed positive rate** per bin based on CDF delta mass.

    Parameters
    ----------
    args : List[polars.Series]
        A two-element list `[positives_tdigests, negatives_tdigests]`, where each
        element is a series of serialized t-digests to be merged for the group.

    Returns
    -------
    pl.Struct
        Dict with:
        - `calibration_bin` : list[float]
            Bin centers over the [0, 1] score range.
        - `calibration_proba` : list[float]
            Estimated mean predicted propensity in each bin.
        - `calibration_rate` : list[float]
            Observed positive rate in each bin.

    Notes
    -----
    - Uses a denser binning for low propensities: 10 bins over [0, 0.1), then 16 bins
      from [0.1, 1], for a total of 26 bins.
    - The per-bin predicted propensity is a weighted mean of bin-centered quantiles
      sampled from positive and negative t-digests, weighted by the estimated counts
      in the bin from each distribution.
    - Returns `[0.0]` defaults if any digest set is empty.

    """
    pos_series, neg_series = args[0], args[1]
    if pos_series.len() == 0 or neg_series.len() == 0:
        return {'calibration_bin': [0.0], 'calibration_proba': [0.0], 'calibration_rate': [0.0]}

    thresholds = [round(float(t), 4) for t in np.linspace(0, 0.1, num=10, endpoint=False)] + [round(float(t), 4) for t
                                                                                              in np.linspace(0.1, 1,
                                                                                                             num=17)]

    all_pos = args[0]
    all_neg = args[1]

    pos_sk = datasketches.tdigest_double.deserialize(merge_digests([all_pos]))
    neg_sk = datasketches.tdigest_double.deserialize(merge_digests([all_neg]))

    if pos_sk.is_empty() or neg_sk.is_empty():
        return {'calibration_bin': [0.0], 'calibration_proba': [0.0], 'calibration_rate': [0.0]}

    pos_count = pos_sk.get_total_weight()
    neg_count = neg_sk.get_total_weight()
    cdf_pos = np.array(pos_sk.get_cdf(thresholds))
    cdf_neg = np.array(neg_sk.get_cdf(thresholds))
    delta_pos = cdf_pos[1:] - cdf_pos[:-1]
    delta_neg = cdf_neg[1:] - cdf_neg[:-1]

    pos_in_bin = pos_count * delta_pos
    neg_in_bin = neg_count * delta_neg
    total_in_bin = pos_in_bin + neg_in_bin
    positives_rate = pos_in_bin / total_in_bin

    n_quantiles = 10
    results_bin = []
    results_proba = []
    results_rate = []

    for i in range(len(thresholds) - 1):
        a, b = thresholds[i], thresholds[i + 1]
        cdf_a_pos = pos_sk.get_cdf([a])[0]
        cdf_b_pos = pos_sk.get_cdf([b])[0]
        cdf_a_neg = neg_sk.get_cdf([a])[0]
        cdf_b_neg = neg_sk.get_cdf([b])[0]

        if (cdf_b_pos - cdf_a_pos) == 0 and (cdf_b_neg - cdf_a_neg) == 0:
            mean_propensity = (a + b) / 2
        else:
            q_pos = np.linspace(cdf_a_pos, cdf_b_pos, n_quantiles, endpoint=False)
            if q_pos.size > 0:
                pos_vals = [pos_sk.get_quantile(q) for q in q_pos]
                pos_bin_mean = np.mean(pos_vals)
            else:
                pos_bin_mean = 0.0

            q_neg = np.linspace(cdf_a_neg, cdf_b_neg, n_quantiles, endpoint=False)
            if q_neg.size > 0:
                neg_vals = [neg_sk.get_quantile(q) for q in q_neg]
                neg_bin_mean = np.mean(neg_vals)
            else:
                neg_bin_mean = 0.0

            total = pos_in_bin[i] + neg_in_bin[i]
            if total > 0:
                mean_propensity = (pos_bin_mean * pos_in_bin[i] + neg_bin_mean * neg_in_bin[i]) / total
            else:
                mean_propensity = (a + b) / 2

        results_bin.append((a + b) / 2)
        results_proba.append(float(mean_propensity))
        results_rate.append(float(positives_rate[i]) if total_in_bin[i] > 0 else 0.0)

    return {'calibration_bin': results_bin, 'calibration_proba': results_proba, 'calibration_rate': results_rate}


@timed
def model_ml_scores(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    """
    Compute ML scoring diagnostics (personalization, novelty, ROC/AUC, calibration) by group.

    Filters IH to negative/positive outcomes, creates a boolean outcome label, and
    aggregates per group the following:
      - `Count` of qualifying rows,
      - `personalization` score via `personalization_optimized([CUSTOMER_ID, NAME])`,
      - `novelty` score via `novelty([CUSTOMER_ID, INTERACTION_ID, NAME])`,
      - **If `use_t_digest=True`**:
          * t-digests for `PROPENSITY` and `FINAL_PROPENSITY` split by outcome boolean,
      - **Else**:
          * a struct of binary metrics from `pds.query_binary_metrics("Outcome_Boolean", PROPENSITY, threshold=0)`.

    Parameters
    ----------
    ih : pl.LazyFrame
        Interaction History lazy frame containing at least:
        `OUTCOME`, `CUSTOMER_ID`, `INTERACTION_ID`, `ACTION_ID`, `RANK`, `NAME`,
        and score columns: `PROPENSITY`, `FINAL_PROPENSITY`.
    config : dict
        Configuration with keys:
        - `group_by` : list[str]
        - `positive_model_response` : list[Any]
        - `negative_model_response` : list[Any]
        - `use_t_digest` : bool or str, optional (default True)
        - `filter` : Any, optional Polars predicate (non-string) to pre-filter IH.
    streaming : bool, default False
        If `background=True`, selects the collection engine (`"streaming"` vs `"auto"`).
    background : bool, default False
        If True, collects and returns an eager DataFrame; otherwise returns a LazyFrame.

    Returns
    -------
    pl.LazyFrame or pl.DataFrame
        Aggregated diagnostics per deduplicated `group_by + global_filters` keys.

    Notes
    -----
    - Keeps only rows with outcomes in the union of negative and positive responses.
    - Deduplicates multiple rows per `(INTERACTION_ID, ACTION_ID, RANK)` by keeping the
      row with the maximum `Outcome_Boolean` inside that interaction key.
    - Ensures each output group has at least one positive (`.filter(pl.any("Outcome_Boolean").over(grp_by))`).
    - When `use_t_digest=False`, the `metrics` struct is un-nested into top-level columns.

    """
    grp_by = stable_dedup(config['group_by'] + get_config()["metrics"]["global_filters"])
    negative_model_response = config['negative_model_response']
    positive_model_response = config['positive_model_response']
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else True

    if "filter" in config:
        filter_exp_cmp = config["filter"]
        if not isinstance(filter_exp_cmp, str):
            ih = ih.filter(config["filter"])

    common_aggs = [
        pl.len().alias('Count'),
        pl.map_groups(
            exprs=[CUSTOMER_ID, NAME],
            function=personalization_optimized,
            return_dtype=pl.Float64,
            returns_scalar=True
        ).alias("personalization"),
        pl.map_groups(
            exprs=[CUSTOMER_ID, INTERACTION_ID, NAME],
            function=novelty,
            return_dtype=pl.Float64,
            returns_scalar=True
        ).alias("novelty")
    ]
    if use_t_digest:
        t_digest_aggs = [
            pl.map_groups(
                exprs=[pl.col(PROPENSITY)
                       .filter(pl.col("Outcome_Boolean") == True)],
                function=lambda s: build_digest(s),
                return_dtype=pl.Binary,
                returns_scalar=True
            ).alias('tdigest_positives'),
            pl.map_groups(
                exprs=[pl.col(PROPENSITY)
                       .filter(pl.col("Outcome_Boolean") == False)],
                function=lambda s: build_digest(s),
                return_dtype=pl.Binary,
                returns_scalar=True
            ).alias('tdigest_negatives'),
            pl.map_groups(
                exprs=[pl.col(FINAL_PROPENSITY)
                       .filter(pl.col("Outcome_Boolean") == True)],
                function=lambda s: build_digest(s),
                return_dtype=pl.Binary,
                returns_scalar=True
            ).alias('tdigest_finalprop_positives'),
            pl.map_groups(
                exprs=[pl.col(FINAL_PROPENSITY)
                       .filter(pl.col("Outcome_Boolean") == False)],
                function=lambda s: build_digest(s),
                return_dtype=pl.Binary,
                returns_scalar=True
            ).alias('tdigest_finalprop_negatives')
        ]
        agg_exprs = common_aggs + t_digest_aggs
    else:
        extra_aggs = [pds.query_binary_metrics(
            "Outcome_Boolean",
            PROPENSITY,
            threshold=0).alias("metrics")]
        agg_exprs = common_aggs + extra_aggs

    try:
        ml_data = (
            ih
            .filter(
                (pl.col(OUTCOME).is_in(negative_model_response + positive_model_response))
            )
            .with_columns([
                pl.when(pl.col(OUTCOME).is_in(positive_model_response)).
                then(True).otherwise(False).alias('Outcome_Boolean')
            ])
            .filter(pl.any("Outcome_Boolean").over(grp_by))
            .filter(pl.col('Outcome_Boolean') == pl.col('Outcome_Boolean').max().over(INTERACTION_ID, ACTION_ID, RANK))
            .group_by(grp_by)
            .agg(agg_exprs)
        )
        if not use_t_digest:
            ml_data = ml_data.unnest(pl.col("metrics"))

        if background:
            return ml_data.collect(background=background, engine="streaming" if streaming else "auto")
        else:
            return ml_data
    except Exception as e:
        print("An exception occurred: ", e)
        traceback.print_exc()
        raise e


@timed
def compact_model_ml_scores_data(model_roc_auc_data: pl.DataFrame,
                                 config: dict) -> pl.DataFrame:
    """
    Compact ML scoring diagnostics to higher-level groups.

    Filters out empty groups (`Count <= 0`), then re-aggregates by the deduplicated
    union of `config['group_by']` and global filters:
      - **If `use_t_digest=False`**: computes weighted means for the numeric `scores`
        columns provided in config (weights = `Count`).
      - **If `use_t_digest=True`**: computes weighted means for `personalization` and
        `novelty`, and **merges** any `tdigest_*` columns by group.

    Parameters
    ----------
    model_roc_auc_data : pl.DataFrame
        Eager DataFrame produced by `model_ml_scores(...).collect(...)`.
    config : dict
        Configuration with keys:
        - `group_by` : list[str]
        - `scores` : list[str]
            Names of metric columns to aggregate when `use_t_digest=False`.
        - `use_t_digest` : bool or str, optional (default True)

    Returns
    -------
    pl.DataFrame
        Compacted DataFrame with:
        - Summed `Count`,
        - Weighted means for requested metrics (or personalization/novelty),
        - Merged `tdigest_*` columns when enabled,
        - Grouping keys preserved.

    Notes
    -----
    - `tdigest_columns` are determined by checking for columns starting with `"tdigest"`.
    - Merging of t-digests uses `merge_digests` over series of binary blobs per group.
    - All weighted means use `weighted_mean(value, Count)` to respect support size.

    """
    auc_data = model_roc_auc_data.filter(pl.col("Count") > 0)
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else True

    grp_by = config['group_by'] + get_config()["metrics"]["global_filters"]
    scores = config["scores"]
    tdigest_columns = [col for col in auc_data.collect_schema().names() if col.startswith("tdigest")]
    grp_by = stable_dedup(grp_by)
    auc_data = (
        auc_data
        .group_by(grp_by)
        .agg(
            (
                [
                    weighted_mean(pl.col(scores), pl.col("Count")).name.suffix("_a")
                ]
                if not use_t_digest else
                [
                    weighted_mean(pl.col(['personalization', 'novelty']), pl.col("Count")).name.suffix("_a")
                ]
            )
            +
            [
                pl.col("Count").sum().alias("Count_a"),
                pl.col(grp_by).first().name.suffix("_a")
            ]
            +
            (
                [
                    pl.map_groups(
                        exprs=[pl.col(f'{c}')],
                        function=lambda s: merge_digests(s),
                        return_dtype=pl.Binary,
                        returns_scalar=True
                    ).alias(f'{c}_a') for c in tdigest_columns
                ] if use_t_digest else []
            )
        )
        .select(cs.ends_with("_a"))
        .rename(lambda column_name: column_name.removesuffix('_a'))
    )

    return auc_data
