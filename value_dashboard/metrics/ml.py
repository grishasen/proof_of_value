import traceback
from typing import List

import numpy as np
import polars as pl
import polars_ds as pds
from polars import Series
from polars import selectors as cs
from polars_ds import weighted_mean
from polars_tdigest import estimate_quantile, merge_tdigests, tdigest
from scipy.interpolate import interp1d
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from value_dashboard.metrics.constants import NAME, CUSTOMER_ID, INTERACTION_ID, RANK, OUTCOME
from value_dashboard.utils.polars_utils import T_DIGEST_COMPRESSION
from value_dashboard.utils.string_utils import strtobool
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


def novelty(args: List[Series]) -> pl.Float64:
    """
    Computes the novelty for a list of recommendations
    Inspired by implementation at https://github.com/statisticianinstilettos/recmetrics.git (unfortunately could not use it in polars context)
    Parameters
    ----------
    args : a 2-elements list of Series with CustomerID and Action Name

    pop: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    u: integer
        The number of users in the training data
    n: integer
        The length of recommended lists per user
    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level
    ----------
    Metric Defintion:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
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
    item_counts = (
        df.group_by(NAME)
        .agg(pl.len().alias("ActionCount"))
    )
    item_counts = item_counts.with_columns(
        (
                pl.col("ActionCount")
                * -((pl.col("ActionCount") / u).log(base=2) + 1e-10)
        ).alias("total_self_info")
    )
    total_self_info = item_counts.select(pl.col("total_self_info").sum()).item()
    rec_lengths = (
        df.group_by(INTERACTION_ID)
        .agg(pl.len().alias("RecLength"))
    )
    n = rec_lengths.select(pl.col("RecLength").max()).item()
    novelty_score = total_self_info / (u * n)
    return float(novelty_score)


def novelty_optimized(args: List[Series]) -> pl.Float64:
    """
    Computes novelty based on item popularity within the recommendation set.
    Optimized by removing sampling and using total recommendations for normalization.

    Parameters
    ----------
    args : a 3-elements list of Series with CustomerID, InteractionID, and Action Name

    Returns
    ----------
    novelty: float
        The novelty score. Higher means more niche items recommended.
    """
    if len(args) < 3:
        return 0.0

    df = pl.DataFrame(args, schema=[CUSTOMER_ID, INTERACTION_ID, NAME])
    total_recommendations = df.height

    if total_recommendations == 0:
        return 0.0

    u = df.n_unique(subset=[CUSTOMER_ID])
    if u == 0:
        return 0.0

    item_counts = (
        df.group_by(NAME)
        .agg(pl.len().alias("ActionCount"))
        .with_columns(
            (pl.col("ActionCount") / u).alias("P_i")
        )
        .with_columns(
            (
                    pl.col("ActionCount")
                    * -(pl.col("P_i").log(base=2.0) + 1e-10)
            ).alias("item_self_info_contribution")
        )
    )

    total_self_info = item_counts.select(
        pl.sum("item_self_info_contribution")
    ).item()
    novelty_score = total_self_info / total_recommendations

    return float(novelty_score) if np.isfinite(novelty_score) else 0.0


def binary_metrics_tdigest(args: List[Series]) -> pl.Struct:
    if (args[0].len() == 0) or (args[1].len() == 0):
        return {'roc_auc': 0.0, 'average_precision': 0.0, 'tpr': [0.0], 'fpr': [0.0]}

    a = np.linspace(0, 0.1, num=100, endpoint=False)
    b = np.linspace(0, 1, num=201)[20:]
    thresholds = np.concatenate((a, b))
    thresholds = [round(t.item(), 4) for t in np.linspace(0, 1, num=101)]

    df = pl.DataFrame(args)
    df = df.filter(pl.col('column_0').struct.field("count") > 0)
    if df.shape[0] > 0:
        df = df.filter(pl.col('column_1').struct.field("count") > 0)
    else:
        return {'roc_auc': 0.0, 'average_precision': 0.0, 'tpr': [0.0], 'fpr': [0.0], 'precision': [0.0],
                'recall': [0.0]}

    if df.shape[0] == 0:
        return {'roc_auc': 0.0, 'average_precision': 0.0, 'tpr': [0.0], 'fpr': [0.0], 'precision': [0.0],
                'recall': [0.0]}

    positives_tdigest_series = df.select(merge_tdigests(pl.col('column_0')))
    positives_tdigest = positives_tdigest_series.item()
    positive_percentiles = (df.select([estimate_quantile('column_0', t).alias(f'{t}') for t in thresholds])
                            .unpivot(cs.numeric())
                            .with_columns(pl.col("variable").cast(pl.Float64))
                            )

    positive_percentiles = dict(positive_percentiles.iter_rows())

    negatives_tdigest_series = df.select(merge_tdigests(pl.col('column_1')))
    negatives_tdigest = negatives_tdigest_series.item()

    negative_percentiles = (df.select([estimate_quantile('column_1', t).alias(f'{t}') for t in thresholds])
                            .unpivot(cs.numeric())
                            .with_columns(pl.col("variable").cast(pl.Float64))
                            )
    negative_percentiles = dict(negative_percentiles.iter_rows())

    q_p = np.array(sorted(positive_percentiles.keys()))
    s_p = np.array([positive_percentiles[q] for q in q_p])
    q_n = np.array(sorted(negative_percentiles.keys()))
    s_n = np.array([negative_percentiles[q] for q in q_n])

    s_p, idx_p = np.unique(s_p, return_index=True)
    q_p = q_p[idx_p]

    s_n, idx_n = np.unique(s_n, return_index=True)
    q_n = q_n[idx_n]

    F_p = interp1d(s_p, q_p, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
    F_n = interp1d(s_n, q_n, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))

    all_scores = np.union1d(s_p, s_n)
    tpr = 1.0 - F_p(all_scores)
    fpr = 1.0 - F_n(all_scores)

    tpr = np.maximum.accumulate(tpr[::-1])[::-1]
    pairs = np.vstack((fpr, tpr)).T
    unique_pairs = np.unique(pairs, axis=0)
    fpr, tpr = unique_pairs[:, 0], unique_pairs[:, 1]

    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    roc_auc = np.trapz(tpr_sorted, fpr_sorted)

    pos = positives_tdigest.get('count')
    neg = negatives_tdigest.get('count')
    #all_scores = np.sort(all_scores)[::-1]
    #tpr = 1.0 - F_p(all_scores)
    #fpr = 1.0 - F_n(all_scores)

    tp = pos * tpr
    fp = neg * fpr

    epsilon = 1e-10
    precision = tp / (tp + fp + epsilon)
    recall = tpr

    precision = np.maximum.accumulate(precision[::-1])[::-1]

    if recall[0] != 0.0:
        recall = np.concatenate(([0.0], recall))
        precision = np.concatenate(([1.0], precision))

    delta_recall = recall[1:] - recall[:-1]
    average_precision = np.sum(delta_recall * precision[1:])

    return {'roc_auc': roc_auc, 'average_precision': average_precision, 'tpr': tpr_sorted.tolist(),
            'fpr': fpr_sorted.tolist(), 'precision': precision.tolist(), 'recall': recall.tolist()}


@timed
def model_ml_scores(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    grp_by = config['group_by']
    negative_model_response = config['negative_model_response']
    positive_model_response = config['positive_model_response']
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else False

    if "filter" in config:
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
            tdigest(pl.col("Propensity").filter(pl.col("Outcome_Boolean") == True),
                    max_size=T_DIGEST_COMPRESSION).alias('tdigest_positives'),
            tdigest(pl.col("Propensity").filter(pl.col("Outcome_Boolean") == False),
                    max_size=T_DIGEST_COMPRESSION).alias('tdigest_negatives')
        ]
        agg_exprs = common_aggs + t_digest_aggs
    else:
        extra_aggs = [pds.query_binary_metrics(
            "Outcome_Boolean",
            "Propensity",
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
            .filter(pl.col('Outcome_Boolean') == pl.col('Outcome_Boolean').max().over(INTERACTION_ID, NAME, RANK))
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
    auc_data = model_roc_auc_data.filter(pl.col("Count") > 0)
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else False

    grp_by = config['group_by']
    scores = config["scores"]
    grp_by = list(set(grp_by))
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
                    merge_tdigests("tdigest_positives").alias("tdigest_positives_a")
                ] if use_t_digest else []
            )
            +
            (
                [
                    merge_tdigests("tdigest_negatives").alias("tdigest_negatives_a")
                ] if use_t_digest else []
            )
        )
        .select(cs.ends_with("_a"))
        .rename(lambda column_name: column_name.removesuffix('_a'))
    )

    return auc_data
