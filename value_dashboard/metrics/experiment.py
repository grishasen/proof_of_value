import traceback

import polars as pl

from value_dashboard.metrics.constants import INTERACTION_ID, OUTCOME, RANK, ACTION_ID
from value_dashboard.utils.config import get_config
from value_dashboard.utils.py_utils import stable_dedup
from value_dashboard.utils.timer import timed


@timed
def experiment(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    mand_props_grp_by = stable_dedup(get_config()["metrics"]["global_filters"] +
                                     config['group_by'] + [config['experiment_name']] + [config['experiment_group']])
    negative_model_response = config['negative_model_response']
    positive_model_response = config['positive_model_response']

    if "filter" in config:
        filter_exp_cmp = config["filter"]
        if not isinstance(filter_exp_cmp, str):
            ih = ih.filter(config["filter"])

    try:
        ih_analysis = (
            ih.filter(
                (pl.col(OUTCOME).is_in(negative_model_response + positive_model_response))
            )
            .with_columns([
                pl.when(pl.col(OUTCOME).is_in(positive_model_response)).
                then(1).otherwise(0).alias('Outcome_Binary')
            ])
            .filter(pl.col('Outcome_Binary') == pl.col('Outcome_Binary').max().over(INTERACTION_ID, ACTION_ID, RANK))
            .group_by(mand_props_grp_by)
            .agg([
                pl.len().alias('Count'),
                pl.sum("Outcome_Binary").alias("Positives")
            ])
            .filter(pl.col("Count") > 0)
            .with_columns([
                (pl.col("Count") - (pl.col("Positives"))).alias("Negatives")
            ])
        )
        if background:
            return ih_analysis.collect(background=background, engine="streaming" if streaming else "auto")
        else:
            return ih_analysis
    except Exception as e:
        traceback.print_exc()
        raise e


@timed
def compact_experiment_data(exp_data: pl.DataFrame,
                            config: dict) -> pl.DataFrame:
    grp_by = stable_dedup(config['group_by'] + [config['experiment_name']] +
                          [config['experiment_group']] + get_config()["metrics"]["global_filters"])
    if grp_by:
        exp_data = (
            exp_data
            .group_by(grp_by)
            .agg([
                pl.col("Count").sum(),
                pl.col("Positives").sum(),
                pl.col("Negatives").sum(),
            ])
            .filter(
                (pl.col("Positives") > 0) & (pl.col("Negatives") > 0)
            )
        )

    return exp_data
