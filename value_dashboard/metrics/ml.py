import traceback
from functools import partial
from typing import List

import numpy as np
import polars as pl
import polars_ds as pds
import polars_ds.sample_and_split as pds_sample_split
from polars import Series
from polars import selectors as cs
from polars_ds import weighted_mean
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity

from value_dashboard.metrics.constants import NAME, CUSTOMER_ID, INTERACTION_ID
from value_dashboard.utils.polars_utils import df_to_dict, tdigest_pos_neg, merge_tdigests, estimate_quantile
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
    df = pl.DataFrame(args)
    n_features = df.n_unique(subset=["column_1"])
    h = FeatureHasher(n_features=n_features, input_type="string")
    df = pds_sample_split.sample(df, 10000)
    df = (
        df
        .group_by(["column_0"])
        .agg([
            pl.col("column_1").alias("ActionNames")
        ])
    )
    predicted = df.get_column("ActionNames").to_list()
    rec_matrix_sparse = csr_matrix(h.transform(predicted))
    similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)
    dim = similarity.shape[0]
    if dim == 1:
        return 0.0
    personalization_score = (similarity.sum() - dim) / (dim * (dim - 1))
    return 1 - personalization_score


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
    df = pl.DataFrame(args)
    u = df.n_unique(subset=["column_0"])
    pop = df_to_dict(
        df
        .group_by(["column_1"])
        .agg(
            [pl.count().alias('Counts')]
        ),
        "column_1",
        "Counts"
    )
    df = pds_sample_split.sample(df, 10000)
    df = (
        df
        .group_by(["column_0"])
        .agg([
            pl.col("column_1").alias("ActionNames"),
            pl.count().alias('Length'),
        ])
    )
    predicted = df.get_column("ActionNames").to_list()
    n = df.get_column("Length").median()
    mean_self_information = []
    k = 0
    for sublist in predicted:
        self_information = 0
        k += 1
        for i in sublist:
            self_information += np.sum(-np.log2(pop[i] / u))
        mean_self_information.append(self_information / n)
    novelty_score = sum(mean_self_information) / k
    return novelty_score


def binary_metrics_tdigest(args: List[Series]) -> pl.Struct:
    a = np.linspace(0, 0.1, num=100, endpoint=False)
    b = np.linspace(0, 1, num=101)[10:]
    thresholds = np.concatenate((a, b))
    thresholds = [round(t.item(), 4) for t in thresholds]

    df = pl.DataFrame(args)

    positive_percentiles = (
        df
        .select('column_0')
        .group_by(None)
        .agg(
            [
                pl.map_groups(
                    exprs=['column_0'],
                    function=partial(estimate_quantile, quantile=t),
                    return_dtype=pl.Struct,
                    returns_scalar=True).alias(f'{t}') for t in thresholds
            ]
        )
        .unpivot(cs.numeric())
        .with_columns(pl.col("variable").cast(pl.Float64))
    )

    positive_percentiles = dict(positive_percentiles.iter_rows())

    negative_percentiles = (
        df
        .select('column_1')
        .group_by(None)
        .agg(
            [
                pl.map_groups(
                    exprs=['column_1'],
                    function=partial(estimate_quantile, quantile=t),
                    return_dtype=pl.Struct,
                    returns_scalar=True).alias(f'{t}') for t in thresholds
            ]
        )
        .unpivot(cs.numeric())
        .with_columns(pl.col("variable").cast(pl.Float64))
    )
    negative_percentiles = dict(negative_percentiles.iter_rows())

    q_p = np.array(sorted(positive_percentiles.keys()))
    s_p = np.array([positive_percentiles[q] for q in q_p])
    q_n = np.array(sorted(negative_percentiles.keys()))
    s_n = np.array([negative_percentiles[q] for q in q_n])

    F_p = interp1d(s_p, q_p, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
    F_n = interp1d(s_n, q_n, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))

    all_scores = np.union1d(s_p, s_n)
    tpr = 1.0 - F_p(all_scores)
    fpr = 1.0 - F_n(all_scores)
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    roc_auc = np.trapz(tpr_sorted, fpr_sorted)

    thresholds = np.sort(all_scores)[::-1]
    TP = 1.0 - F_p(thresholds)
    FP = 1.0 - F_n(thresholds)
    precision = TP / (TP + FP + 1e-10)
    recall = TP

    recall = recall[::-1]
    precision = precision[::-1]

    if recall[0] != 0.0:
        recall = np.concatenate(([0.0], recall))
        precision = np.concatenate(([1.0], precision))

    average_precision = np.trapz(precision, recall)

    return {'roc_auc': roc_auc, 'average_precision': average_precision}


@timed
def model_ml_scores(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    grp_by = config['group_by']
    negative_model_response = config['negative_model_response']
    positive_model_response = config['positive_model_response']
    scores = config['scores']
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else False
    negative_model_response_both_classes = strtobool(config[
                                                         'negative_model_response_both_classes']) if 'negative_model_response_both_classes' in config.keys() else False
    if negative_model_response_both_classes:
        k = 2
    else:
        k = 1
    unnest_expr = pl.col("metrics")
    if use_t_digest:
        unnest_expr = pl.col("propensity_tdigest_pos_neg")

    if "filter" in config.keys():
        filter_exp = config["filter"]
        if filter_exp:
            ih_filter_expr = eval(filter_exp)
            ih = ih.filter(ih_filter_expr)
    try:
        ml_grp_by = grp_by + ['Outcome', "Propensity"] + [CUSTOMER_ID, INTERACTION_ID]
        if not NAME in ml_grp_by:
            ml_grp_by = ml_grp_by + [NAME]
        ml_data = (
            ih
            .select(ml_grp_by)
            .filter(
                (pl.col("Outcome").is_in(negative_model_response + positive_model_response))
            )
            .with_columns([
                pl.when(pl.col('Outcome').is_in(positive_model_response)).
                then(True).otherwise(False).alias('Outcome_Boolean')
            ])
            .filter(True if not negative_model_response_both_classes else (
                    pl.col('Outcome_Boolean') == pl.col('Outcome_Boolean').max().over(INTERACTION_ID, NAME)))
            .filter(pl.any("Outcome_Boolean").over(grp_by))
            .group_by(grp_by)
            .agg([
                     pl.len().alias('Count'),
                     pl.map_groups(
                         exprs=[CUSTOMER_ID, NAME],
                         function=personalization,
                         return_dtype=pl.Float64,
                         returns_scalar=True
                     ).alias("personalization"),
                     pl.map_groups(
                         exprs=[CUSTOMER_ID, NAME],
                         function=novelty,
                         return_dtype=pl.Float64,
                         returns_scalar=True
                     ).alias("novelty")
                 ]
                 +
                 ([pl.map_groups(
                     exprs=["Outcome_Boolean", "Propensity"],
                     function=tdigest_pos_neg,
                     return_dtype=pl.Struct,
                     returns_scalar=True).alias("propensity_tdigest_pos_neg")] if use_t_digest else []
                  )
                 +
                 ([pds.query_binary_metrics(
                     "Outcome_Boolean",
                     "Propensity",
                     threshold=0).alias("metrics")] if not use_t_digest else []
                  )
                 )
            .unnest([unnest_expr])
        )
        if background:
            return ml_data.collect(background=background, streaming=streaming)
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
                    pl.map_groups(
                        exprs=["tdigest_positives"],
                        function=merge_tdigests,
                        return_dtype=pl.Struct,
                        returns_scalar=True).alias("tdigest_positives_a")
                ] if use_t_digest else []
            )
            +
            (
                [
                    pl.map_groups(
                        exprs=["tdigest_negatives"],
                        function=merge_tdigests,
                        return_dtype=pl.Struct,
                        returns_scalar=True).alias("tdigest_negatives_a")
                ] if use_t_digest else []
            )
        )
        .select(cs.ends_with("_a"))
        .rename(lambda column_name: column_name.removesuffix('_a'))
    )

    return auc_data
