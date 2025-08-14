from functools import partial
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from polars import selectors as cs
from polars_ds import weighted_mean

from value_dashboard.metrics.clv import rfm_summary
from value_dashboard.metrics.constants import MODELCONTROLGROUP, PROPENSITY
from value_dashboard.metrics.ml import binary_metrics_tdigest, calibration_tdigest
from value_dashboard.utils.config import get_config
from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.polars_utils import merge_digests, estimate_quantile
from value_dashboard.utils.py_utils import strtobool
from value_dashboard.utils.stats import chi2_test, g_test, z_test, proportions_ztest
from value_dashboard.utils.timer import timed

logger = get_logger(__name__)


@timed
def calculate_reports_data(
        grouped_rep_data: Union[pd.DataFrame, pl.DataFrame], params: dict
) -> pl.DataFrame:
    report_data = None
    if params["metric"].startswith("engagement"):
        report_data = calculate_engagement_scores(grouped_rep_data, params)
    elif params["metric"].startswith("model_ml_scores"):
        report_data = calculate_model_ml_scores(grouped_rep_data, params)
    elif params["metric"].startswith("conversion"):
        report_data = calculate_conversion_scores(grouped_rep_data, params)
    elif params["metric"].startswith("descriptive"):
        report_data = calculate_descriptive_scores(grouped_rep_data, params)
    elif params["metric"].startswith("experiment"):
        report_data = calculate_experiment_scores(grouped_rep_data, params)
    elif params["metric"].startswith("clv"):
        report_data = calculate_clv_scores(grouped_rep_data, params)
    return report_data


@timed
def group_model_ml_scores_data(
        model_roc_auc_data: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(model_roc_auc_data, pd.DataFrame):
        model_roc_auc_data = pl.from_pandas(model_roc_auc_data)
    if isinstance(model_roc_auc_data, pl.DataFrame):
        model_roc_auc_data = model_roc_auc_data.clone()

    auc_data = model_roc_auc_data.filter(pl.col("Count") > 0)

    m_config = get_config()["metrics"][config["metric"]]
    use_t_digest = (
        strtobool(m_config["use_t_digest"])
        if "use_t_digest" in m_config.keys()
        else True
    )
    logger.debug("Use t-digest for scores: " + str(use_t_digest))

    grp_by = config["group_by"] + get_config()["metrics"]["global_filters"]
    scores = get_config()["metrics"].get("model_ml_scores", {}).get("scores", [])
    grp_by = list(set(grp_by))
    auc_data = (
        auc_data.group_by(grp_by)
        .agg(
            (
                [weighted_mean(pl.col(scores), pl.col("Count")).name.suffix("_a")]
                if not use_t_digest
                else [
                    weighted_mean(
                        pl.col(["personalization", "novelty"]), pl.col("Count")
                    ).name.suffix("_a")
                ]
            )
            + [
                pl.col("Count").sum().alias("Count_a"),
                pl.col(grp_by).first().name.suffix("_a"),
            ]
            + (
                [
                    pl.map_groups(
                        exprs=["tdigest_positives"],
                        function=merge_digests,
                        return_dtype=pl.Binary,
                        returns_scalar=True
                    ).alias("tdigest_positives_a")
                ]
                if use_t_digest
                else []
            )
            + (
                [
                    pl.map_groups(
                        exprs=["tdigest_negatives"],
                        function=merge_digests,
                        return_dtype=pl.Binary,
                        returns_scalar=True
                    ).alias("tdigest_negatives_a")
                ]
                if use_t_digest
                else []
            )
        )
        .select(cs.ends_with("_a"))
        .rename(lambda column_name: column_name.removesuffix("_a"))
    )

    return auc_data


@timed
def group_experiment_data(
        exp_data: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(exp_data, pd.DataFrame):
        exp_data = pl.from_pandas(exp_data)
    if isinstance(exp_data, pl.DataFrame):
        exp_data = exp_data.clone()

    m_config = get_config()["metrics"][config["metric"]]
    grp_by = (
            get_config()["metrics"]["global_filters"]
            + config["group_by"]
            + [m_config["experiment_name"]]
    )

    grp_by = list(set(grp_by))
    if grp_by:
        exp_data = (
            exp_data.group_by(grp_by + [m_config["experiment_group"]])
            .agg(
                [
                    pl.col("Count").sum(),
                    pl.col("Positives").sum(),
                    pl.col("Negatives").sum(),
                ]
            )
            .filter((pl.col("Positives") > 0) & (pl.col("Negatives") > 0))
        )

    return exp_data


def calculate_experiment_scores(
        exp_data: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(exp_data, pd.DataFrame):
        exp_data = pl.from_pandas(exp_data)
    if isinstance(exp_data, pl.DataFrame):
        exp_data = exp_data.clone()
    sort_list = []
    if "facet_row" in config.keys():
        sort_list.append(config["facet_row"])
    if "facet_column" in config.keys():
        sort_list.append(config["facet_column"])

    m_config = get_config()["metrics"][config["metric"]]
    grp_by = config["group_by"]
    grp_by = list(set(grp_by))
    if grp_by:
        exp_data = (
            exp_data.group_by(grp_by + [m_config["experiment_group"]])
            .agg(
                [
                    pl.col("Count").sum(),
                    pl.col("Positives").sum(),
                    pl.col("Negatives").sum(),
                ]
            )
            .filter((pl.col("Positives") > 0) & (pl.col("Negatives") > 0))
        )
        if exp_data.shape[0] == 0:
            return exp_data
        exp_data = (
            exp_data.sort(grp_by + [m_config["experiment_group"]], descending=True)
            .group_by(grp_by, maintain_order=True)
            .agg(
                [
                    pl.map_groups(
                        exprs=[m_config["experiment_group"], "Positives", "Negatives"],
                        function=chi2_test,
                        return_dtype=pl.Struct([pl.Field("chi2_stat", pl.Float64), pl.Field("chi2_dof", pl.Int64),
                                                pl.Field("chi2_p_val", pl.Float64),
                                                pl.Field("chi2_odds_ratio_stat", pl.Float64),
                                                pl.Field("chi2_odds_ratio_ci_low", pl.Float64),
                                                pl.Field("chi2_odds_ratio_ci_high", pl.Float64)]),
                        returns_scalar=True,
                    ).alias("chi2_stat"),
                    pl.map_groups(
                        exprs=[m_config["experiment_group"], "Positives", "Negatives"],
                        function=g_test,
                        return_dtype=pl.Struct([pl.Field("g_stat", pl.Float64), pl.Field("g_dof", pl.Int64),
                                                pl.Field("g_p_val", pl.Float64),
                                                pl.Field("g_odds_ratio_stat", pl.Float64),
                                                pl.Field("g_odds_ratio_ci_low", pl.Float64),
                                                pl.Field("g_odds_ratio_ci_high", pl.Float64)]),
                        returns_scalar=True,
                    ).alias("g_stat"),
                    pl.map_groups(
                        exprs=[m_config["experiment_group"], "Positives", "Count"],
                        function=z_test,
                        return_dtype=pl.Struct([pl.Field("z_score", pl.Float64), pl.Field("z_p_val", pl.Float64)]),
                        returns_scalar=True,
                    ).alias("z_stat"),
                    pl.col("Count").sum(),
                    pl.col("Positives").sum(),
                    pl.col("Negatives").sum(),
                ]
            )
            .unnest(["chi2_stat", "g_stat", "z_stat"])
        )

    if sort_list:
        exp_data = exp_data.sort(sort_list, descending=True)

    return exp_data


def calculate_model_ml_scores(
        model_roc_auc_data: Union[pl.DataFrame, pd.DataFrame],
        config: dict,
        drop_fpr_tpr=True
) -> pl.DataFrame:
    if isinstance(model_roc_auc_data, pd.DataFrame):
        model_roc_auc_data = pl.from_pandas(model_roc_auc_data)
    if isinstance(model_roc_auc_data, pl.DataFrame):
        model_roc_auc_data = model_roc_auc_data.clone()

    grp_by = config.get("group_by", None)
    if not grp_by:
        grp_by = ['ALL']
        model_roc_auc_data = model_roc_auc_data.with_columns(
            pl.lit('ALL').alias('ALL')
        )
    m_config = get_config()["metrics"][config["metric"]]
    scores = m_config["scores"]
    use_t_digest = (
        strtobool(m_config["use_t_digest"])
        if "use_t_digest" in m_config.keys()
        else True
    )
    logger.debug("Use t-digest for scores: " + str(use_t_digest))

    property = 'tdigest' if config.get('property', PROPENSITY) == PROPENSITY else 'tdigest_finalprop'

    auc_data = (
        model_roc_auc_data
        .group_by(grp_by)
        .agg(
            (
                [weighted_mean(pl.col(scores), pl.col("Count")).name.suffix("_a")]
                if not use_t_digest
                else [
                    weighted_mean(
                        pl.col(["personalization", "novelty"]), pl.col("Count")
                    ).name.suffix("_a")
                ]
            )
            + [
                pl.col("Count").sum().alias("Count_a")
            ]
            + ([pl.col(grp_by).first().name.suffix("_a")] if grp_by else [])
            + (
                [
                    pl.map_groups(
                        exprs=[f"{property}_positives",
                               f"{property}_negatives"],
                        function=binary_metrics_tdigest,
                        return_dtype=pl.Struct(
                            [pl.Field("roc_auc", pl.Float64), pl.Field("average_precision", pl.Float64),
                             pl.Field("tpr", pl.List(pl.Float64)), pl.Field("fpr", pl.List(pl.Float64)),
                             pl.Field("precision", pl.List(pl.Float64)), pl.Field("recall", pl.List(pl.Float64)),
                             pl.Field("pos_fraction", pl.Float64)]),
                        returns_scalar=True,
                    ).alias("roc_auc_tdigest_a")
                ]
                if use_t_digest
                else []
            )
            + (
                [
                    pl.map_groups(
                        exprs=[f"{property}_positives",
                               f"{property}_negatives"],
                        function=calibration_tdigest,
                        return_dtype=pl.Struct([pl.Field("calibration_bin", pl.List(pl.Float64)),
                                                pl.Field("calibration_proba", pl.List(pl.Float64)),
                                                pl.Field("calibration_rate", pl.List(pl.Float64))]),
                        returns_scalar=True,
                    ).alias("calibration_tdigest_a")
                ]
                if use_t_digest
                else []
            )
        )
        .select(cs.ends_with("_a"))
        .rename(lambda column_name: column_name.removesuffix("_a"))
        .unnest(["roc_auc_tdigest", 'calibration_tdigest'] if use_t_digest else [])
        .sort(grp_by, descending=False)
    )
    if drop_fpr_tpr:
        auc_data = (
            auc_data
            .drop(
                cs.by_name(
                    ['fpr', 'tpr', 'precision', 'recall', 'calibration_bin', 'calibration_proba', 'calibration_rate'],
                    require_all=False),
                strict=False
            )
        )

    return auc_data


def calculate_engagement_scores(
        ih_analysis: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(ih_analysis, pd.DataFrame):
        ih_analysis = pl.from_pandas(ih_analysis)

    if isinstance(ih_analysis, pl.DataFrame):
        ih_analysis = ih_analysis.clone()

    sort_list = []
    if "x" in config.keys():
        sort_list.append(config["x"])
    if "color" in config.keys():
        sort_list.append(config["color"])
    if "facet_row" in config.keys():
        sort_list.append(config["facet_row"])
    if "facet_column" in config.keys():
        sort_list.append(config["facet_column"])

    data_copy = ih_analysis.filter(pl.col("Negatives") > 0)
    column_name_map = {"z_score": "Lift_Z_Score", "z_p_val": "Lift_P_Val"}

    grp_by = config.get("group_by", None)
    if not grp_by:
        grp_by = ['ALL']
        data_copy = data_copy.with_columns(
            pl.lit('ALL').alias('ALL')
        )
    ret_dtype = pl.Struct({"z_score": pl.Float64, "z_p_val": pl.Float64})
    if grp_by:
        data_copy = (
            data_copy.with_columns(
                [
                    pl.when(pl.col(MODELCONTROLGROUP) == "Control")
                    .then(pl.lit("Control"))
                    .otherwise(pl.lit("Test"))
                    .alias(MODELCONTROLGROUP)
                ]
            )
            .group_by(
                grp_by if MODELCONTROLGROUP in grp_by else grp_by + [MODELCONTROLGROUP]
            )
            .agg(
                pl.sum("Count").alias("Count"),
                pl.sum("Negatives").alias("Negatives"),
                pl.sum("Positives").alias("Positives"),
            )
            .with_columns([(pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives"))).alias("CTR")])
            .sort(
                grp_by if MODELCONTROLGROUP in grp_by else grp_by + [MODELCONTROLGROUP]
            )
            .group_by(grp_by, maintain_order=True)
            .agg(
                pl.col("CTR").last().alias("TestCTR"),
                pl.col("CTR").first().alias("ControlCTR"),
                pl.sum("Negatives").alias("Negatives"),
                pl.sum("Positives").alias("Positives"),
                pl.sum("Count").alias("Count"),
                pl.col("Negatives").last().alias("Negatives_Test"),
                pl.col("Positives").last().alias("Positives_Test"),
                pl.col("Negatives").first().alias("Negatives_Control"),
                pl.col("Positives").first().alias("Positives_Control"),
            )
            .with_columns(
                (
                        (pl.col("TestCTR") - pl.col("ControlCTR")) / pl.col("ControlCTR")
                ).alias("Lift"),
                (pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives"))).alias("CTR"),
            )
            .with_columns(
                [
                    (
                        (
                                (
                                        (pl.col("CTR") * (1 - pl.col("CTR")))
                                        / (pl.col("Positives") + pl.col("Negatives"))
                                )
                                ** 0.5
                        )
                    ).alias("StdErr"),
                    pl.col("Lift").replace({np.inf: 0.0}),
                ]
            )
            .with_columns(pl.col("Lift").fill_nan(0.0))
            .with_columns(
                pl.struct(
                    [
                        "Positives_Control",
                        "Positives_Test",
                        "Negatives_Control",
                        "Negatives_Test",
                    ]
                )
                .map_elements(
                    lambda x: proportions_ztest(
                        x["Positives_Test"],
                        x["Positives_Control"],
                        x["Negatives_Test"],
                        x["Negatives_Control"],
                    ),
                    return_dtype=ret_dtype,
                    returns_scalar=True
                )
                .alias("Z_Test")
            )
            .unnest("Z_Test")
            .drop(
                [
                    "TestCTR",
                    "ControlCTR",
                    "Positives_Control",
                    "Positives_Test",
                    "Negatives_Control",
                    "Negatives_Test",
                ]
            )
            .rename(lambda column_name: column_name_map.get(column_name, column_name))
            .sort(grp_by, descending=False)
        )

    return data_copy


def calculate_conversion_scores(
        ih_analysis: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(ih_analysis, pd.DataFrame):
        ih_analysis = pl.from_pandas(ih_analysis)
    if isinstance(ih_analysis, pl.DataFrame):
        ih_analysis = ih_analysis.clone()

    grp_by = config["group_by"]
    metric = config["metric"]
    copy_data = (
        ih_analysis.group_by(grp_by)
        .agg(
            [
                pl.col("Negatives").sum(),
                pl.col("Positives").sum(),
                pl.col("Count").sum(),
                pl.col("Revenue").sum(),
                pl.col('Touchpoints').sum()
            ]
        )
        .with_columns(
            [(pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives"))).alias("ConversionRate"),
             (pl.col("Touchpoints") / pl.col("Positives")).alias("AvgTouchpoints")]
        )
        .with_columns(
            [
                (
                    (
                            (
                                    (pl.col("ConversionRate") * (1 - pl.col("ConversionRate")))
                                    / (pl.col("Positives") + pl.col("Negatives"))
                            )
                            ** 0.5
                    )
                ).alias("StdErr")
            ]
        )
        .sort(config["group_by"], descending=False)
    )

    return copy_data


@timed
def group_engagement_data(
        eng_data: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(eng_data, pd.DataFrame):
        eng_data = pl.from_pandas(eng_data)
    if isinstance(eng_data, pl.DataFrame):
        eng_data = eng_data.clone()

    grp_by = config["group_by"] + get_config()["metrics"]["global_filters"]
    grp_by = list(set(grp_by))
    if not (MODELCONTROLGROUP in grp_by):
        grp_by = grp_by + [MODELCONTROLGROUP]
    if grp_by:
        data_copy = eng_data.group_by(grp_by).agg(
            pl.sum("Negatives").alias("Negatives"),
            pl.sum("Positives").alias("Positives"),
            pl.sum("Count").alias("Count"),
        )

    return data_copy


@timed
def group_conversion_data(
        conv_data: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(conv_data, pd.DataFrame):
        conv_data = pl.from_pandas(conv_data)
    if isinstance(conv_data, pl.DataFrame):
        conv_data = conv_data.clone()

    data_copy = conv_data.filter(pl.col("Negatives") > 0)

    grp_by = config["group_by"] + get_config()["metrics"]["global_filters"]
    grp_by = list(set(grp_by))
    if grp_by:
        data_copy = data_copy.group_by(grp_by).agg(
            pl.sum("Negatives").alias("Negatives"),
            pl.sum("Positives").alias("Positives"),
            pl.sum("Revenue").alias("Revenue"),
            pl.sum("Count").alias("Count"),
        )

    return data_copy


@timed
def group_descriptive_data(
        data: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    if isinstance(data, pl.DataFrame):
        data = data.clone()
    return data


def calculate_descriptive_scores(
        data: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    if isinstance(data, pl.DataFrame):
        data = data.clone()

    m_config = get_config()["metrics"][config["metric"]]
    use_t_digest = (
        strtobool(m_config["use_t_digest"])
        if "use_t_digest" in m_config.keys()
        else True
    )
    logger.debug("Use t-digest for scores: " + str(use_t_digest))
    columns_conf = m_config["columns"]
    scores = m_config["scores"]
    num_columns = [col for col in columns_conf if (col + "_Mean") in data.columns]
    grp_by = config["group_by"]

    grouped_mean = (
        data.group_by(grp_by)
        .agg(
            [pl.col(grp_by).first().name.suffix("_a")]
            + [
                weighted_mean(pl.col(f"{c}_Mean"), pl.col(f"{c}_Count")).alias(
                    f"{c}_GroupMean_a"
                )
                for c in num_columns
            ]
        )
        .select(cs.ends_with("_a"))
        .rename(lambda column_name: column_name.removesuffix("_a"))
    )
    cdata = data.join(grouped_mean, on=grp_by)
    cdata = cdata.with_columns(
        [
            # (n_i - 1) * variance_i
            ((pl.col(f"{c}_Count") - 1) * pl.col(f"{c}_Var")).alias(
                f"{c}_n_minus1_variance"
            )
            for c in num_columns
        ]
        + [
            # n_i * (mean_i - grp_mean)^2
            (
                    pl.col(f"{c}_Count")
                    * (pl.col(f"{c}_Mean") - pl.col(f"{c}_GroupMean")) ** 2
            ).alias(f"{c}_n_mean_diff_sq")
            for c in num_columns
        ]
    )
    logger.debug("Mean and Var calculated... ")
    if not use_t_digest:
        cdata = (
            cdata.group_by(grp_by)
            .agg(
                [
                    (cs.ends_with("Count").sum()).name.suffix("_a"),
                    (cs.ends_with("Sum").sum()).name.suffix("_a"),
                    (cs.ends_with("Min").min()).name.suffix("_a"),
                    (cs.ends_with("Max").max()).name.suffix("_a"),
                    pl.col(grp_by).first().name.suffix("_a"),
                ]
                + [
                    weighted_mean(pl.col(f"{c}_Mean"), pl.col(f"{c}_Count")).alias(
                        f"{c}_Mean_a"
                    )
                    for c in num_columns
                ]
                + [
                    weighted_mean(pl.col(f"{c}_Median"), pl.col(f"{c}_Count")).alias(
                        f"{c}_Median_a"
                    )
                    for c in num_columns
                ]
                + [
                    weighted_mean(pl.col(f"{c}_Skew"), pl.col(f"{c}_Count")).alias(
                        f"{c}_Skew_a"
                    )
                    for c in num_columns
                ]
                + (
                    [
                        weighted_mean(pl.col(f"{c}_p25"), pl.col(f"{c}_Count")).alias(
                            f"{c}_p25_a"
                        )
                        for c in num_columns
                    ]
                    if "p25" in scores
                    else []
                )
                + (
                    [
                        weighted_mean(pl.col(f"{c}_p75"), pl.col(f"{c}_Count")).alias(
                            f"{c}_p75_a"
                        )
                        for c in num_columns
                    ]
                    if "p75" in scores
                    else []
                )
                + (
                    [
                        weighted_mean(pl.col(f"{c}_p95"), pl.col(f"{c}_Count")).alias(
                            f"{c}_p95_a"
                        )
                        for c in num_columns
                    ]
                    if "p95" in scores
                    else []
                )
                + (
                    [
                        weighted_mean(pl.col(f"{c}_p90"), pl.col(f"{c}_Count")).alias(
                            f"{c}_p90_a"
                        )
                        for c in num_columns
                    ]
                    if "p90" in scores
                    else []
                )
                + [
                    pl.col(f"{c}_n_minus1_variance")
                .sum()
                .alias(f"{c}_sum_n_minus1_variance_tmp_a")
                    for c in num_columns
                ]
                + [
                    pl.col(f"{c}_n_mean_diff_sq")
                .sum()
                .alias(f"{c}_sum_n_mean_diff_sq_tmp_a")
                    for c in num_columns
                ]
            )
            .select(cs.ends_with("_a"))
            .rename(lambda column_name: column_name.removesuffix("_a"))
            .with_columns(
                [
                    (
                            (
                                    pl.col(f"{c}_sum_n_minus1_variance_tmp")
                                    + pl.col(f"{c}_sum_n_mean_diff_sq_tmp")
                            )
                            / (pl.col(f"{c}_Count") - 1)
                    ).alias(f"{c}_Var")
                    for c in num_columns
                ]
            )
            .with_columns(
                [(pl.col(f"{c}_Var").sqrt()).alias(f"{c}_Std") for c in num_columns]
            )
            .select(~cs.ends_with("_tmp"))
            .sort(grp_by, descending=False)
        )
    else:
        non_tdigest_aggs = [
            (cs.ends_with("Count").sum()).name.suffix("_a"),
            (cs.ends_with("Sum").sum()).name.suffix("_a"),
            (cs.ends_with("Min").min()).name.suffix("_a"),
            (cs.ends_with("Max").max()).name.suffix("_a"),
            pl.col(grp_by).first().name.suffix("_a"),
        ]

        for c in num_columns:
            non_tdigest_aggs.extend([
                weighted_mean(pl.col(f"{c}_Mean"), pl.col(f"{c}_Count")).alias(f"{c}_Mean_a"),
                pl.col(f"{c}_n_minus1_variance").sum().alias(f"{c}_sum_n_minus1_variance_tmp_a"),
                pl.col(f"{c}_n_mean_diff_sq").sum().alias(f"{c}_sum_n_mean_diff_sq_tmp_a"),
            ])

        quantiles = [
            (0.5, "Median"),
            (0.25, "p25"),
            (0.75, "p75"),
            (0.90, "p90"),
            (0.95, "p95"),
            (0.0, "p0"),
            (1.0, "p100")
        ]
        tdigest_aggs = []
        for c in num_columns:
            for quantile, suffix in quantiles:
                tdigest_aggs.append(
                    pl.map_groups(
                        exprs=[f"{c}_tdigest"],
                        function=partial(estimate_quantile, quantile=quantile),
                        returns_scalar=True,
                        return_dtype=pl.Float64
                    ).alias(f'{c}_{suffix}_a')
                )

        cdata = cdata.group_by(grp_by).agg(non_tdigest_aggs + tdigest_aggs)
        cdata = (
            cdata.select(cs.ends_with("_a"))
            .rename(lambda column_name: column_name.removesuffix("_a"))
            .select(~cs.ends_with("_tdigest"))
            .with_columns(
                [
                    (
                            (
                                    pl.col(f"{c}_p75")
                                    + pl.col(f"{c}_p25")
                                    - 2 * pl.col(f"{c}_Median")
                            )
                            / (pl.col(f"{c}_p75") - pl.col(f"{c}_p25"))
                    ).alias(f"{c}_Skew")
                    for c in num_columns
                ]
            )
            .with_columns(
                [
                    (
                            (
                                    pl.col(f"{c}_sum_n_minus1_variance_tmp")
                                    + pl.col(f"{c}_sum_n_mean_diff_sq_tmp")
                            )
                            / (pl.col(f"{c}_Count") - 1)
                    ).alias(f"{c}_Var")
                    for c in num_columns
                ]
            )
            .with_columns(
                [(pl.col(f"{c}_Var").sqrt()).alias(f"{c}_Std") for c in num_columns]
            )
            .select(~cs.ends_with("_tmp"))
            .sort(grp_by, descending=False)
        )
    # logger.debug("End calculate_descriptive_scores ...")
    return cdata


def calculate_clv_scores(
        data: Union[pl.DataFrame, pd.DataFrame], config: dict
) -> pl.DataFrame:
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    if isinstance(data, pl.DataFrame):
        data = data.clone()
    m_config = get_config()["metrics"][config["metric"]]
    totals_frame = rfm_summary(data, m_config)
    return totals_frame


def merge_descriptive_digests(
        data: Union[pl.DataFrame, pd.DataFrame], config: dict) -> pl.DataFrame:
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    if isinstance(data, pl.DataFrame):
        data = data.clone()

    m_config = get_config()["metrics"][config["metric"]]
    use_t_digest = (
        strtobool(m_config["use_t_digest"])
        if "use_t_digest" in m_config.keys()
        else True
    )
    logger.debug("Use t-digest for scores: " + str(use_t_digest))
    columns_conf = m_config["columns"]
    num_columns = [col for col in columns_conf if (col + "_Mean") in data.columns]
    grp_by = config["group_by"]

    cdata = data.clone()
    if not use_t_digest:
        return pl.DataFrame()
    else:
        tdigest_aggs = [
            pl.map_groups(
                exprs=[pl.col(f'{c}_tdigest')],
                function=lambda s: merge_digests(s),
                return_dtype=pl.Binary,
                returns_scalar=True
            ).alias(f'{c}_tdigest_a') for c in num_columns
        ]

        cdata = cdata.group_by(grp_by).agg([pl.col(grp_by).first().name.suffix("_a")] + tdigest_aggs)
        cdata = (
            cdata.select(cs.ends_with("_a"))
            .rename(lambda column_name: column_name.removesuffix("_a"))
            .sort(grp_by, descending=False)
        )
    return cdata
